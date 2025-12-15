import torch
import torch.nn.functional as F
import numpy as np
from scipy import signal
from dataclasses import dataclass

try:
    from profiler import get_profiler
except ImportError:
    from .profiler import get_profiler


# OPTIMIZED: Use SciPy's lfilter - optimized C code, much faster!
def _biquad_filter_kernel_scipy(x, b0, b1, b2, a1, a2, z1_in, z2_in):
    batch_size, chunk_len = x.shape
    device = x.device

    # Move to CPU for SciPy (optimized C is faster than GPU for IIR)
    x_cpu = x.cpu().numpy()
    b0_cpu = b0.cpu().numpy().flatten()  # Flatten to 1D
    b1_cpu = b1.cpu().numpy().flatten()
    b2_cpu = b2.cpu().numpy().flatten()
    a1_cpu = a1.cpu().numpy().flatten()
    a2_cpu = a2.cpu().numpy().flatten()

    # Convert state format
    z1_cpu = z1_in.cpu().numpy()
    z2_cpu = z2_in.cpu().numpy()

    # Process each batch element with SciPy's fast lfilter
    y_cpu = np.zeros_like(x_cpu)
    z1_out_cpu = np.zeros((batch_size, 1))
    z2_out_cpu = np.zeros((batch_size, 1))

    for i in range(batch_size):
        # Coefficients for this batch element
        b = np.array([b0_cpu[i] if batch_size > 1 else b0_cpu[0],
                      b1_cpu[i] if batch_size > 1 else b1_cpu[0],
                      b2_cpu[i] if batch_size > 1 else b2_cpu[0]])
        a = np.array([1.0,
                      a1_cpu[i] if batch_size > 1 else a1_cpu[0],
                      a2_cpu[i] if batch_size > 1 else a2_cpu[0]])

        # Initial state in SciPy format
        zi = np.array([z1_cpu[i, 0], z2_cpu[i, 0]])

        # Apply filter - THIS IS FAST (C code)
        y_cpu[i, :], zf = signal.lfilter(b, a, x_cpu[i, :], zi=zi)

        # Store final states
        z1_out_cpu[i, 0] = zf[0]
        z2_out_cpu[i, 0] = zf[1]

    # Move back to GPU
    y = torch.from_numpy(y_cpu).float().to(device)
    z1_out = torch.from_numpy(z1_out_cpu).float().to(device)
    z2_out = torch.from_numpy(z2_out_cpu).float().to(device)

    return y, z1_out, z2_out

# Non-JIT version (JIT requires source code, incompatible with PyInstaller)
def _biquad_filter_kernel(x, b0, b1, b2, a1, a2, z1_in, z2_in):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor]
    batch_size = x.shape[0]
    chunk_len = x.shape[1]
    z1 = z1_in
    z2 = z2_in
    y = torch.zeros_like(x)

    for t in range(chunk_len):
        x_t = x[:, t:t+1]
        y_t = b0 * x_t + z1
        z1 = b1 * x_t - a1 * y_t + z2
        z2 = b2 * x_t - a2 * y_t
        y[:, t:t+1] = y_t

    return y, z1, z2


@dataclass
class SynthParams:
    # Oscillators
    osc1_waveform: float = 0.5
    osc1_octave: float = 0.5
    osc1_detune: float = 0.5
    osc1_level: float = 0.8
    osc2_waveform: float = 0.5
    osc2_octave: float = 0.625
    osc2_detune: float = 0.52
    osc2_level: float = 0.6
    sub_level: float = 0.3
    # Filter
    filter_cutoff: float = 0.7
    filter_resonance: float = 0.3
    filter_type: float = 0.0
    filter_env_amount: float = 0.5
    # Amp envelope
    amp_attack: float = 0.1
    amp_decay: float = 0.15
    amp_sustain: float = 0.7
    amp_release: float = 0.25
    # Filter envelope
    filter_attack: float = 0.05
    filter_decay: float = 0.4
    filter_sustain: float = 0.3
    filter_release: float = 0.5
    # Modulation
    lfo1_rate: float = 0.15
    lfo1_to_filter: float = 0.3
    # Effects
    distortion: float = 0.0
    delay_mix: float = 0.0
    delay_time: float = 0.3  # NEW: Parameterized delay time
    delay_feedback: float = 0.3  # NEW: Parameterized delay feedback
    # Advanced modulation
    lfo1_to_pitch: float = 0.0  # NEW: LFO to pitch (vibrato)
    filter_keytrack: float = 0.5  # NEW: Filter key tracking
    noise_level: float = 0.0  # NEW: Noise oscillator level
    # Advanced synthesis
    osc_fm_amount: float = 0.0  # NEW: FM synthesis amount
    unison_detune: float = 0.0  # NEW: Unison detune


class ProSynthesizer:

    def __init__(self, batch_size=256, sample_rate=44100, device='cuda', active_params=None):
        print(f"[SYNTH DEBUG] Creating synthesizer batch_size={batch_size}, device={device}")
        self.device = device
        self.batch_size = batch_size
        self.sr = sample_rate
        self.param_count = 32  # Updated from 25 to 32
        self.profiler = get_profiler()

        # Parameter names
        self.param_names = [f.name for f in SynthParams.__dataclass_fields__.values()]

        # Active parameters configuration
        if active_params is None:
            # Use all parameters
            self.active_params = set(self.param_names)
        else:
            self.active_params = set(active_params)

        # Determine which features are enabled based on active params
        self._setup_feature_flags()

        # Pre-allocate filter states for efficiency
        self.filter_z1 = None
        self.filter_z2 = None
        print(f"[SYNTH DEBUG] Synthesizer ready with {len(self.active_params)} active params")
        print(f"[SYNTH DEBUG] Active features: osc2={self.use_osc2}, sub={self.use_sub}, filter={self.use_filter}, lfo={self.use_lfo}, effects={self.use_effects}")

    def _setup_feature_flags(self):
        # Check which features have active parameters
        self.use_osc2 = any(p.startswith('osc2_') for p in self.active_params)
        self.use_sub = 'sub_level' in self.active_params
        self.use_filter = any(p.startswith('filter_') or p.startswith('filt_') for p in self.active_params)
        self.use_lfo = any(p.startswith('lfo') for p in self.active_params)
        self.use_effects = ('distortion' in self.active_params or 'delay_mix' in self.active_params)

    def get_parameter_count(self):
        return self.param_count

    def get_parameter_names(self):
        return self.param_names

    def create_voice(self, voice_id=0, realtime_mode=True):
        return SynthVoice(self, voice_id=voice_id, realtime_mode=realtime_mode)

    def synthesize_batch(self, param_array, duration=None, note=60, velocity=0.8):
        self.profiler.start("synth_total")

        self.profiler.start("synth_setup")
        param_array = param_array.to(self.device)
        batch_size = param_array.shape[0]

        # Parse and scale parameters correctly
        params = self._parse_parameters(param_array, note)

        # Extract note_hold_time and add to params
        if param_array.shape[1] >= 37:
            # Extract note_hold_time (37th parameter at index 36)
            try:
                from ..config import config
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from config import config

            note_hold_param = param_array[:, 36]  # Keep as tensor
            # Convert to seconds: [0.1, 2.0]
            note_hold_time_seconds = 0.1 + note_hold_param * 1.9
            params['note_hold_time'] = note_hold_time_seconds
        else:
            # Default note_hold_time if not provided
            params['note_hold_time'] = torch.full((batch_size,), 0.25, device=self.device)

        # Calculate duration from note_hold_time if not provided
        if duration is None:
            # Total duration = attack + decay + note_hold_time + release
            attack = params['amp_attack'][0].item()
            decay = params['amp_decay'][0].item()
            release = params['amp_release'][0].item()
            note_hold = params['note_hold_time'][0].item()
            duration = attack + decay + note_hold + release

        num_samples = int(duration * self.sr)
        self.profiler.end("synth_setup")

        # STEP 1: Generate oscillators (with PolyBLEP anti-aliasing)
        self.profiler.start("synth_oscillators")
        audio = self._generate_oscillators(params, batch_size, num_samples)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.profiler.end("synth_oscillators")

        # STEP 2: Apply filter (with envelope modulation) - ONLY IF ACTIVE
        if self.use_filter:
            self.profiler.start("synth_filter")
            audio = self._apply_filter(audio, params, batch_size, num_samples)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            self.profiler.end("synth_filter")

        # STEP 3: Apply amplitude envelope (scaled by velocity)
        # Always apply minimal envelope for anti-click
        self.profiler.start("synth_amp_env")
        audio = self._apply_amplitude_envelope(audio, params, batch_size, num_samples, velocity)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.profiler.end("synth_amp_env")

        # STEP 4: Apply effects - ONLY IF ACTIVE
        if self.use_effects:
            self.profiler.start("synth_effects")
            audio = self._apply_effects(audio, params)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            self.profiler.end("synth_effects")

        # STEP 5: Final normalization
        self.profiler.start("synth_normalize")
        audio = self._normalize(audio)
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.profiler.end("synth_normalize")

        # Force CUDA synchronization to ensure clean state between calls
        # Critical for preventing deadlocks during rapid synthesis in training loops
        self.profiler.start("synth_cuda_sync")
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.profiler.end("synth_cuda_sync")

        self.profiler.end("synth_total")
        return audio

    def _parse_parameters(self, param_array, note):
        # Calculate base frequency from MIDI note
        base_freq = 440.0 * (2.0 ** ((note - 69) / 12.0))

        # Oscillator parameters
        osc1_wave_param = param_array[:, 0].clamp(0.0, 1.0)  # Continuous [0, 1]
        osc1_octave = -2.0 + param_array[:, 1] * 4.0  # [-2, +2] octaves
        osc1_detune = -50.0 + param_array[:, 2] * 100.0  # [-50, +50] cents
        osc1_freq = base_freq * (2.0 ** osc1_octave) * (2.0 ** (osc1_detune / 1200.0))
        osc1_level = param_array[:, 3]

        osc2_wave_param = param_array[:, 4].clamp(0.0, 1.0)  # Continuous [0, 1]
        osc2_octave = -2.0 + param_array[:, 5] * 4.0
        osc2_detune = -50.0 + param_array[:, 6] * 100.0
        osc2_freq = base_freq * (2.0 ** osc2_octave) * (2.0 ** (osc2_detune / 1200.0))
        osc2_level = param_array[:, 7]

        sub_freq = torch.full((param_array.shape[0],), base_freq * 0.5, device=param_array.device)  # -1 octave
        sub_level = param_array[:, 8]

        # Timbre shaping parameters
        pulse_width = param_array[:, 9]  # [0, 1] - pulse width for square wave (0.5 = 50% duty cycle)
        osc_sync = param_array[:, 10]  # [0, 1] - oscillator sync amount
        ring_mod = param_array[:, 11]  # [0, 1] - ring modulation amount

        # Filter parameters (LOGARITHMIC scaling for cutoff!)
        filter_cutoff = 20.0 * (1000.0 ** param_array[:, 12])  # [20, 20000] Hz
        filter_cutoff = torch.clamp(filter_cutoff, 20.0, self.sr / 3.0)  # Stability limit
        filter_res = 0.5 + (param_array[:, 13] ** 2) * 19.5  # [0.5, 20] Q (quadratic)
        filter_type_idx = (param_array[:, 14] * 3.999).long().clamp(0, 3)  # 0=LP, 1=HP, 2=BP, 3=Notch
        filter_env_amount = param_array[:, 15]
        filter_drive = param_array[:, 16] * 10.0  # [0, 10] - saturation amount

        # Amp envelope (LOGARITHMIC scaling for times!)
        # Optimized for 1-second audio clips:
        # - Attack: [0.001, 0.3s] - quick to punchy attacks
        # - Decay: [0.01, 0.5s] - enough time to reach sustain
        # - Release: [0.001, 5.0s] - long release for pads
        amp_attack = 0.001 * (300.0 ** param_array[:, 17])  # [0.001, 0.3] sec
        amp_decay = 0.01 * (50.0 ** param_array[:, 18])     # [0.01, 0.5] sec
        amp_sustain = param_array[:, 19]  # [0, 1] - level, not time!
        amp_release = 0.001 * (5000.0 ** param_array[:, 20]) # [0.001, 5.0] sec

        # Filter envelope (LOGARITHMIC scaling)
        # Same time constraints as amp envelope
        filt_attack = 0.001 * (300.0 ** param_array[:, 21])  # [0.001, 0.3] sec
        filt_decay = 0.01 * (50.0 ** param_array[:, 22])     # [0.01, 0.5] sec
        filt_sustain = param_array[:, 23]
        filt_release = 0.001 * (300.0 ** param_array[:, 24]) # [0.001, 0.3] sec

        # LFO (LOGARITHMIC scaling)
        # Min raised to 0.5 Hz so at least half a cycle is audible in 1 second
        lfo_rate = 0.5 * (40.0 ** param_array[:, 25])  # [0.5, 20] Hz
        lfo_to_filter = param_array[:, 26]

        # Effects
        distortion = param_array[:, 27]
        delay_mix = param_array[:, 28]

        # NEW: Delay parameters (29, 30)
        delay_time = 0.01 * (100.0 ** param_array[:, 29])  # [0.01, 1.0] sec (logarithmic)
        delay_feedback = param_array[:, 30] * 0.8  # [0.0, 0.8] (capped for stability)

        # NEW: LFO to pitch (31)
        lfo_to_pitch = param_array[:, 31] * 0.5  # [0, 0.5] semitones

        # NEW: Filter key tracking (32)
        filter_keytrack = param_array[:, 32]  # [0, 1]

        # NEW: Noise level (33)
        noise_level = param_array[:, 33]  # [0, 1]

        # NEW: FM amount (34)
        osc_fm_amount = param_array[:, 34] * 4.0  # [0, 4] modulation index

        # NEW: Unison detune (35)
        unison_detune = param_array[:, 35] * 50.0  # [0, 50] cents

        return {
            'osc1_wave_param': osc1_wave_param,
            'osc1_freq': osc1_freq,
            'osc1_level': osc1_level,
            'osc2_wave_param': osc2_wave_param,
            'osc2_freq': osc2_freq,
            'osc2_level': osc2_level,
            'sub_freq': sub_freq,
            'sub_level': sub_level,
            'pulse_width': pulse_width,
            'osc_sync': osc_sync,
            'ring_mod': ring_mod,
            'filter_cutoff': filter_cutoff,
            'filter_res': filter_res,
            'filter_type_idx': filter_type_idx,
            'filter_env_amount': filter_env_amount,
            'filter_drive': filter_drive,
            'amp_attack': amp_attack,
            'amp_decay': amp_decay,
            'amp_sustain': amp_sustain,
            'amp_release': amp_release,
            'filt_attack': filt_attack,
            'filt_decay': filt_decay,
            'filt_sustain': filt_sustain,
            'filt_release': filt_release,
            'lfo_rate': lfo_rate,
            'lfo_to_filter': lfo_to_filter,
            'distortion': distortion,
            'delay_mix': delay_mix,
            'delay_time': delay_time,
            'delay_feedback': delay_feedback,
            'lfo_to_pitch': lfo_to_pitch,
            'filter_keytrack': filter_keytrack,
            'noise_level': noise_level,
            'osc_fm_amount': osc_fm_amount,
            'unison_detune': unison_detune,
            'midi_note': torch.full((param_array.shape[0],), float(note), device=param_array.device),  # For filter keytrack
        }

    def _generate_oscillators(self, params, batch_size, num_samples):
        # Apply unison detune (chorus effect) - slight random detuning
        osc1_freq_with_unison = params['osc1_freq']
        osc2_freq_with_unison = params['osc2_freq']

        unison_detune = params['unison_detune']  # [0, 50] cents
        if unison_detune.max() > 0.1:
            self.profiler.start("synth_unison")
            # Simple unison: add small periodic detune to create chorus/width
            # For single-voice synthesis, apply fixed detune amount
            # For batch synthesis, vary detune across batch elements
            # This creates stereo width / chorus effect
            if batch_size == 1:
                # Single voice: apply maximum detune amount
                detune_cents = unison_detune * 0.3  # Use 30% of max for audible effect
            else:
                # Multiple voices: spread detune across batch
                batch_offsets = torch.arange(batch_size, device=self.device, dtype=torch.float32)
                detune_pattern = torch.sin(batch_offsets * 1.234567)  # Deterministic pattern
                detune_cents = detune_pattern * 0.5 * unison_detune  # [-detune/2, +detune/2]

            detune_multiplier = 2.0 ** (detune_cents / 1200.0)  # Convert cents to frequency ratio

            osc1_freq_with_unison = params['osc1_freq'] * detune_multiplier
            if self.use_osc2:
                osc2_freq_with_unison = params['osc2_freq'] * detune_multiplier
            self.profiler.end("synth_unison")

        # Apply LFO to pitch modulation (vibrato) if enabled
        osc1_freq_modulated = osc1_freq_with_unison
        osc2_freq_modulated = osc2_freq_with_unison

        if self.use_lfo:
            lfo_to_pitch = params['lfo_to_pitch']  # [0, 0.5] semitones
            if lfo_to_pitch.max() > 0.01:
                self.profiler.start("synth_lfo_pitch")
                # Generate LFO for pitch modulation
                lfo_unipolar = self._generate_lfo(num_samples, batch_size, params['lfo_rate'])
                lfo_bipolar = (lfo_unipolar * 2.0) - 1.0  # [-1, +1]

                # Convert semitones to frequency multiplier: 2^(semitones/12)
                # lfo_to_pitch is in [0, 0.5] semitones, lfo_bipolar is [-1, +1]
                # So modulation range is ±0.5 semitones
                pitch_mod_semitones = lfo_bipolar * lfo_to_pitch.unsqueeze(1)  # (batch_size, num_samples)
                pitch_multiplier = 2.0 ** (pitch_mod_semitones / 12.0)

                # Apply to both oscillators (use unison-detuned frequencies as base)
                osc1_freq_modulated = osc1_freq_with_unison.unsqueeze(1) * pitch_multiplier  # (batch_size, num_samples)
                if self.use_osc2:
                    osc2_freq_modulated = osc2_freq_with_unison.unsqueeze(1) * pitch_multiplier

                self.profiler.end("synth_lfo_pitch")

        # Generate OSC1 (always present - it's the base tone)
        self.profiler.start("synth_osc1_phase")
        osc1_phase = self._generate_phase(osc1_freq_modulated, num_samples, batch_size)
        self.profiler.end("synth_osc1_phase")

        self.profiler.start("synth_osc1_polyblep")
        # Use base frequency for polyblep (modulation is already in phase)
        osc1_freq_for_polyblep = osc1_freq_modulated if osc1_freq_modulated.ndim == 1 else params['osc1_freq']
        osc1_audio = self._generate_polyblep_waveform(osc1_phase, osc1_freq_for_polyblep, params['osc1_wave_param'], params['pulse_width'])
        self.profiler.end("synth_osc1_polyblep")

        self.profiler.start("synth_osc1_mix")
        # OSC1 will be added in the ring modulation section if OSC2 is active
        # Otherwise add it directly here
        if not self.use_osc2:
            mixed = osc1_audio * params['osc1_level'].unsqueeze(1)
        self.profiler.end("synth_osc1_mix")

        # Generate OSC2 only if active
        if self.use_osc2:
            self.profiler.start("synth_osc2")

            # FM synthesis: OSC1 modulates OSC2's frequency
            osc_fm_amount = params['osc_fm_amount']  # [0, 4] modulation index
            if osc_fm_amount.max() > 0.01:
                self.profiler.start("synth_fm")
                # FM: Use OSC1's audio to modulate OSC2's phase
                # modulation_index controls depth: 0 = no FM, higher = more harmonics
                # Scale osc1_audio to phase modulation amount
                fm_mod_depth = osc_fm_amount.unsqueeze(1) * 2.0 * torch.pi  # (batch_size, 1) -> radians
                fm_phase_mod = osc1_audio * fm_mod_depth  # (batch_size, num_samples)

                # Generate base OSC2 phase then add FM modulation
                osc2_phase_base = self._generate_phase(osc2_freq_modulated, num_samples, batch_size)
                osc2_phase = (osc2_phase_base + fm_phase_mod) % (2.0 * torch.pi)
                self.profiler.end("synth_fm")
            else:
                # No FM - normal phase generation
                osc2_phase = self._generate_phase(osc2_freq_modulated, num_samples, batch_size)

            # Hard sync: OSC1 resets OSC2's phase at zero crossings
            # Creates harmonic-rich timbres, classic for lead sounds
            sync_amount = params['osc_sync']
            if sync_amount.max() > 0.01:
                # Detect OSC1 zero crossings (phase wraps from 2π to 0)
                osc1_phase_norm = osc1_phase / (2.0 * torch.pi)  # Normalize to [0, 1]
                osc1_wrapped = torch.diff(osc1_phase_norm, dim=1, prepend=torch.zeros(batch_size, 1, device=self.device)) < 0

                # Reset OSC2 phase at OSC1 zero crossings
                # sync_amount controls blend: 0 = no sync, 1 = full hard sync
                osc2_phase_norm = osc2_phase / (2.0 * torch.pi)
                osc2_phase_synced = osc2_phase_norm * (1.0 - osc1_wrapped.float())

                # Crossfade between normal and synced phase
                osc2_phase_final = (1.0 - sync_amount.unsqueeze(1)) * osc2_phase_norm + sync_amount.unsqueeze(1) * osc2_phase_synced
                osc2_phase = osc2_phase_final * 2.0 * torch.pi  # Convert back to radians

            # Use base frequency for polyblep (modulation is already in phase)
            osc2_freq_for_polyblep = osc2_freq_modulated if osc2_freq_modulated.ndim == 1 else params['osc2_freq']
            osc2_audio = self._generate_polyblep_waveform(osc2_phase, osc2_freq_for_polyblep, params['osc2_wave_param'], params['pulse_width'])

            # Ring modulation: Multiply OSC1 and OSC2 for metallic/bell-like tones
            # At ring_mod=0: Normal additive mixing (OSC1 + OSC2)
            # At ring_mod=1: Full ring modulation (OSC1 * OSC2) replaces both oscillators
            ring_mod_amount = params['ring_mod']
            if ring_mod_amount.max() > 0.01:
                # Ring mod signal: Multiply raw oscillators
                ring_modulated = osc1_audio * osc2_audio

                # For symmetric effect: When ring modulation is active, BOTH oscillators contribute to it
                # Apply geometric mean of levels to preserve energy and ensure symmetry
                combined_level = torch.sqrt(params['osc1_level'] * params['osc2_level'])

                # Normal signals (when ring_mod=0)
                osc1_normal = osc1_audio * params['osc1_level'].unsqueeze(1)
                osc2_normal = osc2_audio * params['osc2_level'].unsqueeze(1)

                # Ring modulated signal (when ring_mod=1)
                ring_mod_scaled = ring_modulated * combined_level.unsqueeze(1)

                # Crossfade: ring_mod=0 uses normal OSC1+OSC2, ring_mod=1 uses ring modulation
                mixed = osc1_normal * (1.0 - ring_mod_amount.unsqueeze(1)) + \
                        osc2_normal * (1.0 - ring_mod_amount.unsqueeze(1)) + \
                        ring_mod_scaled * ring_mod_amount.unsqueeze(1)
            else:
                # No ring mod - normal additive mixing
                mixed = osc1_audio * params['osc1_level'].unsqueeze(1) + \
                        osc2_audio * params['osc2_level'].unsqueeze(1)

            self.profiler.end("synth_osc2")

        # Generate SUB only if active
        if self.use_sub:
            self.profiler.start("synth_sub")
            sub_phase = self._generate_phase(params['sub_freq'], num_samples, batch_size)
            sub_audio = torch.sin(sub_phase)  # Sub is always sine
            mixed = mixed + sub_audio * params['sub_level'].unsqueeze(1)
            self.profiler.end("synth_sub")

        # Add noise if noise_level > 0
        noise_level = params['noise_level']
        if noise_level.max() > 0.01:
            self.profiler.start("synth_noise")
            # Generate white noise (random values in [-1, 1])
            noise = torch.randn(batch_size, num_samples, device=self.device)
            mixed = mixed + noise * noise_level.unsqueeze(1)
            self.profiler.end("synth_noise")

        return mixed

    def _generate_phase(self, frequencies, num_samples, batch_size):
        if frequencies.ndim == 1:
            # Constant frequency: (batch_size,)
            phase_increment = 2.0 * torch.pi * frequencies / self.sr  # (batch_size,)

            # Create sample indices: (1, num_samples)
            sample_indices = torch.arange(num_samples, device=self.device).unsqueeze(0)

            # Calculate phases: (batch_size, 1) * (1, num_samples) = (batch_size, num_samples)
            phases = (phase_increment.unsqueeze(1) * sample_indices) % (2.0 * torch.pi)
        else:
            # Time-varying frequency: (batch_size, num_samples)
            # Use cumulative sum for phase accumulation
            phase_increment = 2.0 * torch.pi * frequencies / self.sr  # (batch_size, num_samples)
            phases = torch.cumsum(phase_increment, dim=1) % (2.0 * torch.pi)

        return phases

    def _poly_blep(self, phase_norm, phase_increment):
        dt = phase_increment

        # First discontinuity (at phase = 0)
        mask1 = phase_norm < dt
        t1 = phase_norm / dt
        corr1 = t1 + t1 - t1 * t1 - 1.0

        # Second discontinuity (at phase = 1)
        mask2 = phase_norm > (1.0 - dt)
        t2 = (phase_norm - 1.0) / dt
        corr2 = t2 * t2 + t2 + t2 + 1.0

        # Combine corrections
        correction = torch.zeros_like(phase_norm)
        correction = torch.where(mask1, corr1, correction)
        correction = torch.where(mask2, corr2, correction)

        return correction

    def _generate_polyblep_waveform(self, phase, frequencies, wave_param, pulse_width):
        batch_size, num_samples = phase.shape
        phase_norm = phase / (2.0 * torch.pi)  # Normalize to [0, 1]
        phase_increment = (frequencies / self.sr).unsqueeze(1)  # (batch_size, 1)

        # Generate all waveform types
        # Sine (no anti-aliasing needed)
        sine = torch.sin(phase)

        # Sawtooth with PolyBLEP
        saw_naive = 2.0 * phase_norm - 1.0
        saw_correction = self._poly_blep(phase_norm, phase_increment)
        saw = saw_naive - saw_correction

        # Square with PolyBLEP + Pulse Width Modulation
        # pulse_width: (batch_size,) - expand to (batch_size, 1) for broadcasting
        pw = pulse_width.unsqueeze(1).clamp(0.01, 0.99)  # Clamp to avoid DC offset issues
        square_naive = torch.where(phase_norm < pw, 1.0, -1.0)
        square_corr1 = self._poly_blep(phase_norm, phase_increment)
        square_corr2 = self._poly_blep((phase_norm - pw + 1.0) % 1.0, phase_increment)
        square = square_naive + square_corr1 - square_corr2

        # Triangle (integrate square)
        # For simplicity, use band-limited triangle approximation
        tri_naive = torch.where(phase_norm < 0.5,
                                4.0 * phase_norm - 1.0,
                                -4.0 * phase_norm + 3.0)
        tri = tri_naive  # Triangle aliasing is less severe, acceptable for now

        # CONTINUOUS MORPHING between waveforms
        # Map wave_param [0, 1] to blend between 4 waveforms
        # 0.0-0.33: sine -> saw
        # 0.33-0.67: saw -> square
        # 0.67-1.0: square -> triangle

        # Scale to [0, 3] range
        scaled = wave_param * 3.0

        # Determine which two waveforms to blend
        # segment 0: sine->saw, segment 1: saw->square, segment 2: square->tri
        segment = scaled.floor().long().clamp(0, 2)  # Which segment we're in
        blend = (scaled - segment.float()).clamp(0.0, 1.0)  # Blend factor within segment

        # Expand blend to match audio dimensions: (batch_size,) -> (batch_size, num_samples)
        blend_expanded = blend.unsqueeze(1)

        # Create output by blending
        # Stack waveforms for easy indexing: (batch_size, 4, num_samples)
        all_waves = torch.stack([sine, saw, square, tri], dim=1)

        # For each batch element, blend between segment[i] and segment[i]+1
        batch_idx = torch.arange(batch_size, device=self.device)
        wave_a = all_waves[batch_idx, segment, :]
        wave_b = all_waves[batch_idx, (segment + 1).clamp(0, 3), :]

        # Linear interpolation
        output = wave_a * (1.0 - blend_expanded) + wave_b * blend_expanded

        return output

    def _generate_lfo(self, num_samples, batch_size, lfo_rate):
        # Create time array
        t = torch.arange(num_samples, device=self.device, dtype=torch.float32) / self.sr
        t_expanded = t.unsqueeze(0)  # (1, num_samples)

        # LFO phase
        lfo_phase = 2.0 * torch.pi * lfo_rate.unsqueeze(1) * t_expanded

        # Generate sine LFO, scale to [0, 1]
        lfo = (torch.sin(lfo_phase) + 1.0) / 2.0

        return lfo

    def _apply_filter(self, audio, params, batch_size, num_samples):
        # Apply filter drive/saturation before filtering
        # This adds harmonic distortion for warmer, grittier tones
        drive_amount = params['filter_drive']
        if drive_amount.max() > 0.01:
            self.profiler.start("synth_filter_drive")
            # Soft clipping using tanh for smooth saturation
            # drive_amount scales from 0 to 10, higher values = more distortion
            audio = torch.tanh(audio * (1.0 + drive_amount.unsqueeze(1)))
            self.profiler.end("synth_filter_drive")

        # Start with zero modulation
        self.profiler.start("synth_filter_mod_init")
        total_mod_octaves = torch.zeros(batch_size, num_samples, device=self.device)
        self.profiler.end("synth_filter_mod_init")

        # Add filter envelope modulation if filter envelope params are active
        if any(p in self.active_params for p in ['filter_attack', 'filter_decay', 'filter_sustain', 'filter_release', 'filter_env_amount']):
            self.profiler.start("synth_filter_envelope")
            note_hold_time = params.get('note_hold_time', None)
            filter_env = self._generate_envelope_exponential(
                num_samples, batch_size,
                params['filt_attack'], params['filt_decay'],
                params['filt_sustain'], params['filt_release'],
                note_hold_time=note_hold_time
            )  # (batch_size, num_samples) in [0, 1]

            ENV_DEPTH_OCTAVES = 4.0   # 0 to +4 octaves
            env_amount = params['filter_env_amount'].unsqueeze(1)  # (batch_size, 1)
            env_mod_octaves = filter_env * ENV_DEPTH_OCTAVES * env_amount
            total_mod_octaves = total_mod_octaves + env_mod_octaves
            self.profiler.end("synth_filter_envelope")

        # Add LFO modulation if LFO params are active
        if self.use_lfo:
            self.profiler.start("synth_filter_lfo")
            lfo_unipolar = self._generate_lfo(num_samples, batch_size, params['lfo_rate'])
            lfo_bipolar = (lfo_unipolar * 2.0) - 1.0  # (batch_size, num_samples) in [-1, +1]

            LFO_DEPTH_OCTAVES = 2.0   # ±2 octaves
            lfo_amount = params['lfo_to_filter'].unsqueeze(1)      # (batch_size, 1)
            lfo_mod_octaves = lfo_bipolar * LFO_DEPTH_OCTAVES * lfo_amount
            total_mod_octaves = total_mod_octaves + lfo_mod_octaves
            self.profiler.end("synth_filter_lfo")

        # Add keyboard tracking (filter follows keyboard pitch)
        filter_keytrack = params['filter_keytrack']  # [0, 1]
        if filter_keytrack.max() > 0.01:
            self.profiler.start("synth_filter_keytrack")
            # At keytrack=1.0, filter tracks keyboard 1:1 (same octave as note)
            # At keytrack=0.0, no tracking (fixed cutoff)
            # Reference note is C4 (MIDI 60) - no offset at this note
            midi_note = params['midi_note']  # (batch_size,)
            note_offset_semitones = midi_note - 60.0  # Semitones from C4
            note_offset_octaves = note_offset_semitones / 12.0  # Convert to octaves

            keytrack_mod_octaves = note_offset_octaves * filter_keytrack  # (batch_size,)
            total_mod_octaves = total_mod_octaves + keytrack_mod_octaves.unsqueeze(1)  # Broadcast to (batch_size, num_samples)
            self.profiler.end("synth_filter_keytrack")

        # Apply modulation in logarithmic space
        self.profiler.start("synth_filter_cutoff_calc")
        base_cutoff = params['filter_cutoff'].unsqueeze(1)  # (batch_size, 1) in Hz
        freq_multiplier = 2.0 ** total_mod_octaves           # (batch_size, num_samples)
        cutoff_modulated = base_cutoff * freq_multiplier     # (batch_size, num_samples)

        # Clamp to safe range
        cutoff_modulated = torch.clamp(cutoff_modulated, 20.0, self.sr / 3.0)
        self.profiler.end("synth_filter_cutoff_calc")

        # Apply time-varying filter with chunked coefficient updates
        self.profiler.start("synth_filter_biquad_apply")
        filtered = self._apply_biquad_filter_chunked(
            audio, cutoff_modulated, params['filter_res'], params['filter_type_idx']
        )
        self.profiler.end("synth_filter_biquad_apply")

        return filtered

    def _calculate_biquad_coefficients(self, cutoff, resonance, filter_type_idx, sample_rate):
        # Intermediate calculations
        omega = 2.0 * torch.pi * cutoff / sample_rate
        sin_omega = torch.sin(omega)
        cos_omega = torch.cos(omega)
        alpha = sin_omega / (2.0 * resonance)

        # Calculate coefficients for all filter types
        # Lowpass
        b0_lp = (1.0 - cos_omega) / 2.0
        b1_lp = 1.0 - cos_omega
        b2_lp = (1.0 - cos_omega) / 2.0
        a0_lp = 1.0 + alpha
        a1_lp = -2.0 * cos_omega
        a2_lp = 1.0 - alpha

        # Highpass
        b0_hp = (1.0 + cos_omega) / 2.0
        b1_hp = -(1.0 + cos_omega)
        b2_hp = (1.0 + cos_omega) / 2.0
        a0_hp = 1.0 + alpha
        a1_hp = -2.0 * cos_omega
        a2_hp = 1.0 - alpha

        # Bandpass
        b0_bp = alpha
        b1_bp = torch.zeros_like(alpha)
        b2_bp = -alpha
        a0_bp = 1.0 + alpha
        a1_bp = -2.0 * cos_omega
        a2_bp = 1.0 - alpha

        # Notch
        b0_nt = torch.ones_like(alpha)
        b1_nt = -2.0 * cos_omega
        b2_nt = torch.ones_like(alpha)
        a0_nt = 1.0 + alpha
        a1_nt = -2.0 * cos_omega
        a2_nt = 1.0 - alpha

        # Stack all coefficient sets: (4, batch_size)
        all_b0 = torch.stack([b0_lp, b0_hp, b0_bp, b0_nt], dim=0)
        all_b1 = torch.stack([b1_lp, b1_hp, b1_bp, b1_nt], dim=0)
        all_b2 = torch.stack([b2_lp, b2_hp, b2_bp, b2_nt], dim=0)
        all_a0 = torch.stack([a0_lp, a0_hp, a0_bp, a0_nt], dim=0)
        all_a1 = torch.stack([a1_lp, a1_hp, a1_bp, a1_nt], dim=0)
        all_a2 = torch.stack([a2_lp, a2_hp, a2_bp, a2_nt], dim=0)

        # Select based on filter type
        batch_size = cutoff.shape[0]
        batch_idx = torch.arange(batch_size, device=cutoff.device)

        b0 = all_b0[filter_type_idx, batch_idx]
        b1 = all_b1[filter_type_idx, batch_idx]
        b2 = all_b2[filter_type_idx, batch_idx]
        a0 = all_a0[filter_type_idx, batch_idx]
        a1 = all_a1[filter_type_idx, batch_idx]
        a2 = all_a2[filter_type_idx, batch_idx]

        # Normalize by a0
        b0 = b0 / a0
        b1 = b1 / a0
        b2 = b2 / a0
        a1 = a1 / a0
        a2 = a2 / a0

        return b0, b1, b2, a1, a2

    def _apply_biquad_filter_chunked(self, audio, cutoff_modulated, resonance, filter_type_idx):
        batch_size, num_samples = audio.shape

        # Chunk size: Balance between modulation accuracy and performance
        # 1024 samples = 23ms at 44.1kHz (still imperceptible for modulation)
        # OPTIMIZATION: Larger chunks = fewer coefficient calculations
        CHUNK_SIZE = 1024  # Was 256 - this reduces coeff calculations by 4x

        # Initialize filter state
        z1 = torch.zeros(batch_size, 1, device=self.device)
        z2 = torch.zeros(batch_size, 1, device=self.device)

        # Output tensor
        filtered = torch.zeros_like(audio)

        # OPTIMIZATION: Calculate coefficients ONCE if cutoff is static
        # Check if filter modulation is active (envelopes, LFOs)
        self.profiler.start("synth_biquad_coeff_calc_once")
        has_filter_modulation = (self.use_lfo or
                                any(p in self.active_params for p in ['filter_attack', 'filter_decay',
                                                                       'filter_sustain', 'filter_release',
                                                                       'filter_env_amount']))

        if not has_filter_modulation:
            # FAST PATH: No modulation - calculate coefficients ONCE
            cutoff_static = params['filter_cutoff']  # (batch_size,) - constant
            b0, b1, b2, a1, a2 = self._calculate_biquad_coefficients(
                cutoff_static, resonance, filter_type_idx, self.sr
            )
            b0 = b0.unsqueeze(1)
            b1 = b1.unsqueeze(1)
            b2 = b2.unsqueeze(1)
            a1 = a1.unsqueeze(1)
            a2 = a2.unsqueeze(1)
        self.profiler.end("synth_biquad_coeff_calc_once")

        # Process in chunks
        self.profiler.start("synth_biquad_chunk_loop")
        num_chunks = 0
        for chunk_start in range(0, num_samples, CHUNK_SIZE):
            self.profiler.start("synth_biquad_chunk_iter")
            chunk_end = min(chunk_start + CHUNK_SIZE, num_samples)

            if has_filter_modulation:
                # SLOW PATH: Time-varying cutoff - calculate coefficients per chunk
                # Use cutoff value from start of chunk
                self.profiler.start("synth_biquad_chunk_cutoff")
                cutoff_chunk = cutoff_modulated[:, chunk_start]  # (batch_size,)
                self.profiler.end("synth_biquad_chunk_cutoff")

                # Calculate coefficients for this chunk
                self.profiler.start("synth_biquad_chunk_coeff")
                b0, b1, b2, a1, a2 = self._calculate_biquad_coefficients(
                    cutoff_chunk, resonance, filter_type_idx, self.sr
                )  # All (batch_size,)
                self.profiler.end("synth_biquad_chunk_coeff")

                # Expand coefficients for broadcasting: (batch_size, 1)
                self.profiler.start("synth_biquad_chunk_expand")
                b0 = b0.unsqueeze(1)
                b1 = b1.unsqueeze(1)
                b2 = b2.unsqueeze(1)
                a1 = a1.unsqueeze(1)
                a2 = a2.unsqueeze(1)
                self.profiler.end("synth_biquad_chunk_expand")

            # Extract audio chunk
            self.profiler.start("synth_biquad_chunk_extract")
            x_chunk = audio[:, chunk_start:chunk_end]
            self.profiler.end("synth_biquad_chunk_extract")

            # Apply filter to chunk with state continuity (SciPy optimized C code)
            self.profiler.start("synth_biquad_chunk_kernel")
            y_chunk, z1, z2 = _biquad_filter_kernel_scipy(
                x_chunk, b0, b1, b2, a1, a2, z1, z2
            )
            self.profiler.end("synth_biquad_chunk_kernel")

            self.profiler.start("synth_biquad_chunk_store")
            filtered[:, chunk_start:chunk_end] = y_chunk
            self.profiler.end("synth_biquad_chunk_store")

            self.profiler.end("synth_biquad_chunk_iter")
            num_chunks += 1

        self.profiler.end("synth_biquad_chunk_loop")

        return filtered

    def _generate_envelope_exponential(self, num_samples, batch_size, attack, decay, sustain, release, note_hold_time=None):
        # Create time array: (num_samples,)
        t = torch.arange(num_samples, device=self.device, dtype=torch.float32) / self.sr
        t_expanded = t.unsqueeze(0)  # (1, num_samples)

        # Phase boundaries (in seconds): (batch_size, 1)
        t_att = attack.unsqueeze(1)
        t_dec = (attack + decay).unsqueeze(1)

        # Sustain end time: use note_hold_time if provided, else calculate from total duration
        if note_hold_time is not None:
            t_sus_end = (attack + decay + note_hold_time).unsqueeze(1)
        else:
            t_sus_end = (num_samples / self.sr - release).unsqueeze(1)

        # Initialize envelope
        envelope = torch.zeros(batch_size, num_samples, device=self.device)

        # Time constants for exponential curves
        tau_att = attack.unsqueeze(1) / 4.0  # Attack curve shape
        tau_dec = decay.unsqueeze(1) / 4.0   # Decay curve shape
        tau_rel = release.unsqueeze(1) / 4.0 # Release curve shape

        # ATTACK: exponential rise to 1.0
        # y = 1 - exp(-t/tau)
        attack_mask = t_expanded < t_att
        attack_curve = 1.0 - torch.exp(-t_expanded / (tau_att + 1e-6))
        envelope = torch.where(attack_mask, attack_curve, envelope)

        # DECAY: exponential decay from 1.0 to sustain
        # y = sustain + (1 - sustain) * exp(-(t-t_att)/tau)
        decay_mask = (t_expanded >= t_att) & (t_expanded < t_dec)
        t_from_att = t_expanded - t_att
        decay_curve = sustain.unsqueeze(1) + (1.0 - sustain.unsqueeze(1)) * torch.exp(-t_from_att / (tau_dec + 1e-6))
        envelope = torch.where(decay_mask, decay_curve, envelope)

        # SUSTAIN: constant at sustain level
        sustain_mask = (t_expanded >= t_dec) & (t_expanded < t_sus_end)
        envelope = torch.where(sustain_mask, sustain.unsqueeze(1), envelope)

        # RELEASE: exponential decay from sustain to 0
        # y = sustain * exp(-(t-t_sus_end)/tau)
        release_mask = t_expanded >= t_sus_end
        t_from_sus = t_expanded - t_sus_end
        release_curve = sustain.unsqueeze(1) * torch.exp(-t_from_sus / (tau_rel + 1e-6))
        envelope = torch.where(release_mask, release_curve, envelope)

        return envelope.clamp(0.0, 1.0)

    def _apply_amplitude_envelope(self, audio, params, batch_size, num_samples, velocity):
        # Generate amplitude envelope with note_hold_time if available
        note_hold_time = params.get('note_hold_time', None)
        amp_env = self._generate_envelope_exponential(
            num_samples, batch_size,
            params['amp_attack'], params['amp_decay'],
            params['amp_sustain'], params['amp_release'],
            note_hold_time=note_hold_time
        )

        # Scale by velocity
        amp_env = amp_env * velocity

        # Apply to audio
        audio = audio * amp_env

        return audio

    def _apply_effects(self, audio, params):
        # Distortion (soft clipping using tanh)
        dist_amount = params['distortion']
        if dist_amount.max() > 0.01:  # Only apply if amount > 0
            # Gain before distortion
            gain = 1.0 + dist_amount.unsqueeze(1) * 9.0  # [1, 10] (reduced for stability)
            distorted = torch.tanh(audio * gain)  # Let distortion affect amplitude
            # Slight output compensation to prevent excessive clipping
            distorted = distorted * 0.7

            # Mix dry/wet
            audio = audio * (1.0 - dist_amount.unsqueeze(1)) + distorted * dist_amount.unsqueeze(1)

        # Delay effect (feedback delay with multiple repeats)
        delay_amount = params['delay_mix']
        if delay_amount.max() > 0.01:  # Only apply if amount > 0
            # Use parameterized delay settings
            delay_time_sec = params['delay_time'].mean().item()
            feedback = params['delay_feedback'].mean().item()

            delay_samples = int(delay_time_sec * self.sr)

            if delay_samples < audio.shape[1] and delay_samples > 0:
                # Create delay buffer with multiple feedback repeats
                delayed = torch.zeros_like(audio)
                delayed_signal = audio.clone()

                # Apply up to 4 feedback repeats for audible echo effect
                for i in range(4):
                    offset = delay_samples * (i + 1)
                    if offset >= audio.shape[1]:
                        break
                    decay = feedback ** (i + 1)
                    delayed[:, offset:] = delayed[:, offset:] + delayed_signal[:, :audio.shape[1] - offset] * decay

                # Add delayed signal on top of dry (maintain volume)
                audio = audio + delayed * delay_amount.unsqueeze(1)

        return audio

    def _normalize(self, audio):
        # Find peak per batch element
        peak = torch.abs(audio).max(dim=1, keepdim=True)[0]
        peak = torch.clamp(peak, min=1e-6)

        # Only normalize if needed
        needs_norm = peak > 0.95
        scale = torch.where(needs_norm, 0.95 / peak, torch.ones_like(peak))

        audio = audio * scale

        # Apply fade-out to prevent clicks
        fade_samples = int(0.02 * self.sr)  # 20ms fade
        if fade_samples > 0 and fade_samples < audio.shape[1]:
            fade_curve = torch.linspace(1.0, 0.0, fade_samples, device=self.device)
            audio[:, -fade_samples:] *= fade_curve.unsqueeze(0)

        return audio

    def generate_audio_from_params(self, param_array, duration=1.0, note=60, velocity=0.8):
        # Convert to tensor if numpy
        if isinstance(param_array, np.ndarray):
            param_array = torch.from_numpy(param_array).float()

        # Handle single sample
        if param_array.dim() == 1:
            param_array = param_array.unsqueeze(0)
            single = True
        else:
            single = False

        # Synthesize
        audio = self.synthesize_batch(param_array, duration, note, velocity)

        # Convert back to numpy
        audio_np = audio.cpu().numpy()

        if single:
            return audio_np[0]
        return audio_np


# ==============================================================================
# REAL-TIME STREAMING SYNTHESIS
# ==============================================================================

class SynthVoice:

    def __init__(self, parent_synth, voice_id=0, realtime_mode=False):
        self.synth = parent_synth
        self.voice_id = voice_id
        self.sr = parent_synth.sr
        self.device = parent_synth.device
        self.realtime_mode = realtime_mode
        self.gate_on = False  # Track if note is currently held

        # Current note state
        self.midi_note = 60
        self.velocity = 0.8
        self.base_freq = 261.63  # C4

        # Oscillator phases (persistent across process() calls)
        self.osc1_phase = 0.0
        self.osc2_phase = 0.0
        self.sub_phase = 0.0

        # LFO phase
        self.lfo_phase = 0.0

        # Filter state (biquad Direct Form II)
        self.filter_z1 = 0.0
        self.filter_z2 = 0.0

        # Envelope state machines
        self.amp_env_stage = 'idle'  # idle, attack, decay, sustain, release
        self.amp_env_value = 0.0
        self.amp_env_time = 0.0

        self.filt_env_stage = 'idle'
        self.filt_env_value = 0.0
        self.filt_env_time = 0.0

        # Current parameters (will be set via set_parameters)
        self.params = None
        self.param_array = None  # Store original param array

        # Pre-generated audio buffer
        self.audio_buffer = None
        self.buffer_position = 0

        # Start with silence buffer to avoid blocking on first note
        self.audio_buffer = np.zeros(int(2.0 * self.sr), dtype=np.float32)

    def set_parameters(self, param_array):
        # Convert to torch tensor if needed
        if not isinstance(param_array, torch.Tensor):
            param_array = torch.from_numpy(param_array).float()

        # Ensure on correct device
        param_array = param_array.to(self.device)

        # Add batch dimension if needed
        if param_array.dim() == 1:
            param_array = param_array.unsqueeze(0)

        # Store the param array
        self.param_array = param_array

        # Parse parameters using parent's method
        self.params = self.synth._parse_parameters(param_array, self.midi_note)

    def note_on(self, midi_note, velocity=0.8):
        self.midi_note = midi_note
        self.velocity = velocity
        self.gate_on = True

        if self.param_array is None:
            return

        # Update base frequency for this note
        self.base_freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

        # Re-parse parameters with new note (updates frequencies)
        self.params = self.synth._parse_parameters(self.param_array, self.midi_note)

        if self.realtime_mode:
            # Real-time mode: Generate long buffer with sustain that can be held indefinitely
            # When note_off is called, apply release fade

            # Generate audio with very long sustain (10 seconds max hold)
            attack = self.params['amp_attack'].item() if 'amp_attack' in self.params else 0.01
            decay = self.params['amp_decay'].item() if 'amp_decay' in self.params else 0.1
            sustain_level = self.params['amp_sustain'].item() if 'amp_sustain' in self.params else 0.5
            release = self.params['amp_release'].item() if 'amp_release' in self.params else 0.3

            # Max buffer = attack + decay + 10s sustain + release
            max_duration = attack + decay + 10.0 + release

            # Build 34-parameter array: 32 synth params + note + note_hold_time
            import torch
            note_param = (midi_note - 21) / (108 - 21)  # Normalize MIDI note to [0, 1]
            note_hold_param = 0.5  # Not used since we specify duration explicitly

            param_array_34 = torch.cat([
                self.param_array,
                torch.tensor([[note_param, note_hold_param]], dtype=torch.float32, device=self.param_array.device)
            ], dim=1)

            audio = self.synth.synthesize_batch(
                param_array_34,
                duration=max_duration,
                note=midi_note,
                velocity=velocity
            )

            self.audio_buffer = audio[0].cpu().numpy()
            self.buffer_position = 0

            # Track when note_off was called (for release detection)
            self.note_off_time = None
            self.note_off_buffer_position = None
            self.release_fade_start_pos = None

            # Initialize envelope tracking
            self.amp_env_stage = 'attack'
            self.amp_env_value = 0.0
            self.amp_env_time = 0.0

            self.filt_env_stage = 'attack'
            self.filt_env_value = 0.0
            self.filt_env_time = 0.0
        else:
            # Training mode: Pre-generate shorter clip (use config duration)
            try:
                from ..config import config
                duration = config.DURATION
            except ImportError:
                import sys
                from pathlib import Path
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from config import config
                duration = config.DURATION

            audio = self.synth.synthesize_batch(
                self.param_array,
                duration=duration,
                note=midi_note,
                velocity=velocity
            )

            self.audio_buffer = audio[0].cpu().numpy()
            self.buffer_position = 0

            # Also initialize envelopes for consistency
            self.amp_env_stage = 'attack'
            self.amp_env_value = 0.0
            self.amp_env_time = 0.0

            self.filt_env_stage = 'attack'
            self.filt_env_value = 0.0
            self.filt_env_time = 0.0

    def note_off(self):
        self.gate_on = False

        # In realtime mode, record when note_off was called
        if self.realtime_mode:
            self.note_off_buffer_position = self.buffer_position
            self.note_off_time = self.buffer_position / self.sr
            # Mark where release fade should start
            self.release_fade_start_pos = self.buffer_position

        # Trigger release phase
        if self.amp_env_stage != 'idle':
            self.amp_env_stage = 'release'
            self.amp_env_time = 0.0

        if self.filt_env_stage != 'idle':
            self.filt_env_stage = 'release'
            self.filt_env_time = 0.0

    def is_finished(self):
        if self.audio_buffer is None:
            return True

        if self.realtime_mode:
            # In realtime mode, voice is finished when release phase completes
            # Check if note_off was called and we've played enough release time
            if self.note_off_buffer_position is not None:
                # Get release time parameter
                release_time = self.params['amp_release'].item()

                # Calculate how much time has elapsed since note_off
                samples_since_release = self.buffer_position - self.note_off_buffer_position
                time_since_release = samples_since_release / self.sr

                # Voice is finished if we've played the full release time
                # Add small margin (10ms) to ensure release tail completes
                return time_since_release >= (release_time + 0.01)

            # If note still held (no note_off), not finished
            return False
        else:
            # Training mode: finished when buffer exhausted
            return self.buffer_position >= len(self.audio_buffer)

    def _update_envelope(self, dt):
        import math

        # Get envelope parameters (already scaled from params dict)
        attack = self.params['amp_attack'].item()
        decay = self.params['amp_decay'].item()
        sustain = self.params['amp_sustain'].item()
        release = self.params['amp_release'].item()

        stage = self.amp_env_stage
        value = self.amp_env_value
        time = self.amp_env_time

        if stage == 'idle':
            value = 0.0

        elif stage == 'attack':
            # Exponential attack curve
            tau = attack / 4.0
            if tau > 0.0:
                value = 1.0 - math.exp(-time / tau)
            else:
                value = 1.0

            time += dt
            if time >= attack:
                stage = 'decay'
                time = 0.0
                value = 1.0

        elif stage == 'decay':
            # Exponential decay to sustain level
            tau = decay / 4.0
            if tau > 0.0:
                value = sustain + (1.0 - sustain) * math.exp(-time / tau)
            else:
                value = sustain

            time += dt
            if time >= decay:
                stage = 'sustain'
                time = 0.0
                value = sustain

        elif stage == 'sustain':
            # Hold at sustain level until note_off
            value = sustain
            time += dt

        elif stage == 'release':
            # Exponential decay to zero
            tau = release / 4.0
            if tau > 0.0:
                # Release from current value
                start_value = self.amp_env_value if self.amp_env_value > 0 else sustain
                value = start_value * math.exp(-time / tau)
            else:
                value = 0.0

            time += dt
            if value < 0.001 or time >= release:
                stage = 'idle'
                value = 0.0
                time = 0.0

        # Update state
        self.amp_env_stage = stage
        self.amp_env_value = value
        self.amp_env_time = time

        return value

    def process(self, num_samples):
        if self.audio_buffer is None:
            return np.zeros(num_samples, dtype=np.float32)

        # Return chunk from buffer
        end_pos = min(self.buffer_position + num_samples, len(self.audio_buffer))
        chunk = self.audio_buffer[self.buffer_position:end_pos].copy()

        # In realtime mode with note_off called, apply release fade
        if self.realtime_mode and self.release_fade_start_pos is not None:
            # Get release time from parameters
            release_time = self.params['amp_release'].item() if 'amp_release' in self.params else 0.3
            release_samples = int(release_time * self.sr)

            # Apply exponential fade starting from note_off position
            import math
            for i in range(len(chunk)):
                sample_pos = self.buffer_position + i
                if sample_pos >= self.release_fade_start_pos:
                    # Calculate fade progress (0 to 1)
                    fade_progress = (sample_pos - self.release_fade_start_pos) / release_samples
                    if fade_progress >= 1.0:
                        chunk[i] = 0.0
                    else:
                        # Exponential fade (sounds more natural)
                        fade_multiplier = math.exp(-fade_progress * 5.0)  # -5.0 gives nice exponential curve
                        chunk[i] *= fade_multiplier

        # Pad with zeros if needed
        if len(chunk) < num_samples:
            chunk = np.pad(chunk, (0, num_samples - len(chunk)), 'constant')

        self.buffer_position = end_pos

        return chunk


if __name__ == "__main__":
    print("=" * 80)
    print("PROFESSIONAL VST-QUALITY SYNTHESIZER - V2")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Create synthesizer
    synth = ProSynthesizer(batch_size=4, device=device)

    # Test with random parameters
    params = torch.rand(4, 25, device=device)

    print("Synthesizing 1 second of audio (batch=4)...")
    audio = synth.synthesize_batch(params, duration=1.0, note=60, velocity=0.8)

    print(f"Output shape: {audio.shape}")
    print(f"Output range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"Output mean: {audio.mean():.3f}")
    print(f"No NaNs: {not torch.isnan(audio).any()}")

    print("\n[OK] Professional synthesizer initialized!")
