import numpy as np
import torch
import soundfile as sf
import sys
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from core.synthesizer import ProSynthesizer
from core.similarity_panns import PANNsSimilarityMetric
from core.similarity_mfcc import MFCCSimilarityMetric
from core.similarity_spectral import SpectralSimilarityMetric
from core.similarity_temporal import TemporalSimilarityMetric
from core.similarity_combined import CombinedSimilarityMetric
from core.similarity_mssl import MultiScaleSpectralLoss
from core.optimizer import CMAESOptimizer
from core.preset_manager import PresetManager
from core.audio_analysis import analyze_audio_for_matching
from config import config


class WeightedMultiMetric:
    def __init__(self, metrics_dict, weights=None, device='cuda'):
        self.metrics = metrics_dict
        self.device = device

        # Set equal weights if not provided
        if weights is None:
            weights = {name: 1.0 for name in metrics_dict.keys()}
        self.weights = weights

        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        print(f"[MULTI-METRIC] Initialized with metrics: {list(self.metrics.keys())}")
        print(f"[MULTI-METRIC] Weights: {self.weights}")

    def set_target(self, target):
        for metric_name, metric in self.metrics.items():
            metric.set_target(target)

    def compute_distance(self, audio1, audio2=None):
        distances = []
        for metric_name, metric in self.metrics.items():
            # Convert to CPU/numpy if needed (Temporal, Spectral, etc need numpy)
            if metric_name in ['Temporal', 'Spectral', 'MFCC']:
                if isinstance(audio1, torch.Tensor):
                    audio_np = audio1.cpu().numpy() if audio1.is_cuda else audio1.numpy()
                else:
                    audio_np = audio1
                dist = metric.compute_distance(audio_np, audio2)
            else:
                # PANNS uses tensors
                dist = metric.compute_distance(audio1, audio2)

            weighted_dist = dist * self.weights[metric_name]
            distances.append(weighted_dist)

        return sum(distances)


def convert_to_wav(input_path):
    input_path = Path(input_path)

    # If already WAV, return as-is
    if input_path.suffix.lower() == '.wav':
        return str(input_path), False

    print(f"  Converting {input_path.suffix} to WAV format...")

    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'],
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH.\n"
            "Please install FFmpeg to use MP3/other audio formats:\n"
            "  - Windows: Download from https://ffmpeg.org/download.html\n"
            "  - Or use: winget install ffmpeg\n"
            "After installation, restart your terminal/IDE."
        )

    # Create temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()

    try:
        # Convert to WAV using FFmpeg
        # -i: input file
        # -ar 44100: sample rate 44.1kHz
        # -ac 2: stereo (will be converted to mono later)
        # -y: overwrite output file
        result = subprocess.run(
            ['ffmpeg', '-i', str(input_path),
             '-ar', '44100', '-ac', '2', '-y', temp_wav_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        print(f"  Conversion successful: {input_path.name} -> WAV")
        return temp_wav_path, True

    except subprocess.CalledProcessError as e:
        # Clean up temp file on error
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)
        raise RuntimeError(
            f"Failed to convert audio file: {input_path.name}\n"
            f"FFmpeg error: {e.stderr.decode('utf-8', errors='ignore')}"
        )


def match_audio_file(
    target_audio_path,
    progress_callback=None,
    max_evals=60000,
    target_similarity=0.95,
    sample_duration=0.5,
    num_restarts=3,
    user_stop_callback=None,
    device=None,
    stagnation_gens=50,
    optimization_method="PANNS",  # Changed from similarity_metric
    initial_params=None,
    previous_metric=None,
    target_note=60,  # MIDI note to use for synthesis (default: C4)
    log_dir="results/optimization_logs",  # Directory to save optimization logs
    restart_strategy='inverted_ipop'  # Restart strategy for CMA-ES
):
    print("\n" + "="*80)
    print("AUDIO PARAMETER MATCHING")
    print("="*80)
    print(f"Target audio: {target_audio_path}")
    print("="*80 + "\n")

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda':
        if not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available, using CPU")
            device = 'cpu'

    print(f"Using device: {device}")

    # Convert to WAV if needed
    print("\n[1/5] Loading target audio...")
    wav_path, is_temp = convert_to_wav(target_audio_path)

    # Load target audio
    target_audio, sr = sf.read(wav_path)

    # Clean up temporary file now that audio is loaded into memory
    if is_temp and wav_path and os.path.exists(wav_path):
        try:
            os.unlink(wav_path)
        except Exception as e:
            print(f"  Warning: Could not delete temp file: {e}")

    # Convert stereo to mono if needed
    if target_audio.ndim > 1:
        target_audio = target_audio.mean(axis=1)

    # Resample if needed (target should be 44.1kHz)
    if sr != config.SAMPLE_RATE:
        print(f"  Warning: Resampling from {sr} Hz to {config.SAMPLE_RATE} Hz")
        from scipy import signal
        num_samples = int(len(target_audio) * config.SAMPLE_RATE / sr)
        target_audio = signal.resample(target_audio, num_samples)

    # Trim/pad to specified duration for matching
    target_length = int(sample_duration * config.SAMPLE_RATE)
    if len(target_audio) > target_length:
        target_audio = target_audio[:target_length]
    elif len(target_audio) < target_length:
        target_audio = np.pad(target_audio, (0, target_length - len(target_audio)))

    print(f"  Target audio: {len(target_audio)} samples ({sample_duration}s), {config.SAMPLE_RATE} Hz")

    # Analyze audio to detect pitch and noise level
    print("\n[1.5/5] Analyzing target audio...")
    audio_analysis = analyze_audio_for_matching(target_audio, config.SAMPLE_RATE)

    # Override target_note with detected pitch
    if target_note == 60:  # Only override if using default
        target_note = audio_analysis['midi_note']
        print(f"  Using detected MIDI note: {target_note}")
    else:
        print(f"  Using user-specified MIDI note: {target_note} (detected: {audio_analysis['midi_note']})")

    # Initialize synthesizer
    print("\n[2/5] Initializing synthesizer...")
    synth = ProSynthesizer(batch_size=1, device=device)
    print(f"  Synthesizer ready on {device}")

    # Initialize optimization method
    print(f"\n[3/5] Initializing optimization: {optimization_method}...")

    # Parse optimization method
    if optimization_method == "PANNS":
        # Single metric: PANNS only
        metric = PANNsSimilarityMetric(device=device)
        method_display = "PANNS"
    elif optimization_method == "Temporal+PANNS_Static":
        # Multi-metric: Temporal + PANNS (static weights)
        metrics_dict = {
            'Temporal': TemporalSimilarityMetric(sample_rate=config.SAMPLE_RATE),
            'PANNS': PANNsSimilarityMetric(device=device)
        }
        metric = WeightedMultiMetric(metrics_dict, weights=None, device=device)  # Equal weights
        method_display = "Temporal+PANNS (Static)"
    elif optimization_method == "Temporal+PANNS_GradNorm":
        # Multi-metric: Temporal + PANNS (adaptive GradNorm)
        # For now, use static weights (user feedback: static works as well as GradNorm)
        metrics_dict = {
            'Temporal': TemporalSimilarityMetric(sample_rate=config.SAMPLE_RATE),
            'PANNS': PANNsSimilarityMetric(device=device)
        }
        metric = WeightedMultiMetric(metrics_dict, weights=None, device=device)
        method_display = "Temporal+PANNS (GradNorm)"
        print("  [NOTE] Using static weights (performs as well as adaptive GradNorm)")
    elif optimization_method == "TSP_Static":
        # Multi-metric: Temporal + Spectral + PANNS (static weights)
        metrics_dict = {
            'Temporal': TemporalSimilarityMetric(sample_rate=config.SAMPLE_RATE),
            'Spectral': SpectralSimilarityMetric(sample_rate=config.SAMPLE_RATE),
            'PANNS': PANNsSimilarityMetric(device=device)
        }
        metric = WeightedMultiMetric(metrics_dict, weights=None, device=device)  # Equal weights
        method_display = "Temporal+Spectral+PANNS (Static)"
    # Legacy single-metric options (for backwards compatibility)
    elif optimization_method == "MFCC":
        metric = MFCCSimilarityMetric(sample_rate=config.SAMPLE_RATE)
        method_display = "MFCC"
    elif optimization_method == "Spectral":
        metric = SpectralSimilarityMetric(sample_rate=config.SAMPLE_RATE)
        method_display = "Spectral"
    elif optimization_method == "Temporal":
        metric = TemporalSimilarityMetric(sample_rate=config.SAMPLE_RATE)
        method_display = "Temporal"
    elif optimization_method == "Combined":
        metric = CombinedSimilarityMetric(sample_rate=config.SAMPLE_RATE)
        method_display = "Combined"
    elif optimization_method == "MSSL":
        # Multi-Scale Spectral Loss (linear magnitudes)
        metric = MultiScaleSpectralLoss(sample_rate=config.SAMPLE_RATE, use_log=False)
        method_display = "Multi-Scale Spectral Loss (Linear)"
    elif optimization_method == "MSSL_Log":
        # Multi-Scale Spectral Loss (log-scaled magnitudes)
        metric = MultiScaleSpectralLoss(sample_rate=config.SAMPLE_RATE, use_log=True)
        method_display = "Multi-Scale Spectral Loss (Log)"
    else:
        raise ValueError(f"Unknown optimization method: {optimization_method}")

    target_tensor = torch.from_numpy(target_audio).float().to(device)
    metric.set_target(target_tensor)
    print(f"  {method_display} initialized")

    # Define objective function
    def objective_function(params):
        # All params are synthesis parameters (including note_hold_time)
        # note is passed separately as target_note
        synth_params = params.copy()  # Copy to avoid modifying optimizer's state

        # Lock noise_level to 0 if target has no noise
        # noise_level is at index 31 (counting from 0, excluding 'note')
        if audio_analysis['lock_noise']:
            synth_params[31] = 0.0  # Force noise_level to 0

        # Use provided target_note
        note_midi = target_note

        # Synthesize audio (note_hold_time is included in params, duration will be calculated)
        synth_params_tensor = torch.from_numpy(synth_params).float().reshape(1, -1).to(config.DEVICE)
        audio_tensor = synth.synthesize_batch(
            synth_params_tensor,
            duration=sample_duration,
            note=int(note_midi),
            velocity=0.8
        )

        # Compute distance
        distance = metric.compute_distance(audio_tensor[0], target_tensor)
        return distance

    # Run optimization
    print("\n[4/5] Running CMA-ES optimization...")
    print(f"  Config: {max_evals} max evals, {num_restarts} restarts, "
          f"{sample_duration}s samples")
    print(f"  Stops at: {max_evals} evals OR {target_similarity:.1%} similarity (whichever first)")
    if initial_params is not None:
        print(f"  Starting from: Current preset (continue optimization)")
    print("  This may take 30 seconds to 5 minutes...\n")

    optimizer = CMAESOptimizer(
        objective_function=objective_function,
        seed=42,
        progress_callback=progress_callback,
        max_evals=max_evals,
        target_similarity=target_similarity,
        num_restarts=num_restarts,
        user_stop_callback=user_stop_callback,
        stagnation_gens=stagnation_gens,
        initial_params=initial_params,
        log_dir=log_dir,
        restart_strategy=restart_strategy
    )

    # Set metadata for logging
    if log_dir is not None:
        optimizer.metadata = {
            'target_filename': os.path.basename(target_audio_path),
            'detected_pitch_hz': float(audio_analysis['frequency']),
            'detected_midi_note': int(audio_analysis['midi_note']),
            'detected_noise_level': float(audio_analysis['noise_level']),
            'target_note_used': int(target_note),
            'sample_duration': float(sample_duration),
            'optimization_method': optimization_method
        }

    result = optimizer.optimize()

    # Auto-plot convergence if logging was enabled
    if log_dir is not None:
        try:
            import glob
            from scripts.plotting.plot_convergence import plot_convergence

            # Find the most recent log file
            log_files = glob.glob(os.path.join(log_dir, "optimization_log_*.json"))
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
                print(f"\n[PLOTTING] Generating convergence plot...")
                plot_convergence(latest_log)
                print(f"[OK] Convergence plot saved")
        except Exception as e:
            print(f"[WARNING] Could not generate plot: {e}")

    best_params = result['best_params']
    best_fitness = result['best_fitness']
    best_similarity = result['similarity']
    all_solutions = result.get('all_solutions', [])

    print(f"\n[OPTIMIZATION COMPLETE]")
    print(f"  Best similarity: {best_similarity:.2%}")
    print(f"  Best fitness: {best_fitness:.6f}")
    print(f"  Total evaluations: {result['total_evals']}")
    print(f"  Solutions found: {len(all_solutions)}")

    # Convert best params array to dict (37 params: 36 synth + note_hold_time)
    synth_params_with_hold = [p for p in config.SYNTH_PARAM_ORDER if p != 'note']
    best_params_dict = {}
    for i, param_name in enumerate(synth_params_with_hold):
        best_params_dict[param_name] = float(best_params[i])
    # Add the note that was used for matching
    best_params_dict['note'] = target_note / 127.0  # Normalize to [0,1]

    # Synthesize matched audio for preview (using best solution)
    print("\n[5/5] Generating matched audio and saving presets...")
    synth_params = best_params  # All 37 synthesis parameters (36 + note_hold_time)
    note_midi = target_note

    synth_params_tensor = torch.from_numpy(synth_params).float().reshape(1, -1).to(config.DEVICE)
    matched_audio_tensor = synth.synthesize_batch(
        synth_params_tensor,
        duration=sample_duration,  # Use same duration as optimization for consistent envelope
        note=int(note_midi),
        velocity=0.8
    )
    matched_audio = matched_audio_tensor[0].cpu().numpy()

    # Save ALL solutions as presets (sorted by similarity, highest first)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_filename = Path(target_audio_path).stem
    preset_manager = PresetManager(config.PRESETS_DIR)

    all_preset_names = []

    # Sort solutions by similarity (descending)
    sorted_solutions = sorted(all_solutions, key=lambda x: x['similarity'], reverse=True)

    for idx, solution in enumerate(sorted_solutions):
        # Convert params to dict (37 params: 36 synth + note_hold_time)
        params_dict = {}
        for i, param_name in enumerate(synth_params_with_hold):
            params_dict[param_name] = float(solution['params'][i])
        # Add the note that was used for matching
        params_dict['note'] = target_note / 127.0  # Normalize to [0,1]

        # Create preset name with similarity, index, and method
        similarity_pct = int(solution['similarity'] * 100)
        method_short = optimization_method.replace("+", "").replace("_", "")  # Shorten for filename
        preset_name = f"matched_{target_filename}_{timestamp}_{similarity_pct}pct_{idx+1}_{method_short}"

        # Save preset
        preset_manager.save_preset(
            preset_name,
            params_dict,
            metadata={
                'description': f'Matched from {target_filename}',
                'similarity': f"{solution['similarity']:.2%}",
                'target_file': str(target_audio_path),
                'timestamp': timestamp,
                'fitness': f"{solution['fitness']:.6f}"
            }
        )

        all_preset_names.append(preset_name)
        print(f"  Saved: {preset_name}.json ({solution['similarity']:.2%} similarity)")

    # Best preset is first in sorted list
    best_preset_name = all_preset_names[0] if all_preset_names else None

    print(f"\n[SAVED] {len(all_preset_names)} preset(s)")
    print("="*80 + "\n")

    return {
        'params_dict': best_params_dict,
        'similarity': best_similarity,
        'preset_name': best_preset_name,
        'matched_audio': matched_audio,
        'all_preset_names': all_preset_names,
        'total_evals': result['total_evals'],
        'history': result.get('history', [])
    }


if __name__ == "__main__":
    # Test standalone
    if len(sys.argv) < 2:
        print("Usage: python match_audio.py <target_audio.wav>")
        sys.exit(1)

    target_path = sys.argv[1]
    result = match_audio_file(target_path)

    print(f"\nMatch complete!")
    print(f"Similarity: {result['similarity']:.2%}")
    print(f"Preset: {result['preset_name']}")
