import sys
import argparse
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from core.synthesizer import ProSynthesizer
from config import config

from sweep_config import (
    QUICK_CONFIG, FULL_CONFIG, PHASE_PARAMS,
    PHASE_THRESHOLD_WORKS, PHASE_THRESHOLD_STRONG,
    IDENTICAL_PAIRS, NOISE_THRESHOLD
)


# Pin values and their colors
PIN_VALUES = [0.2, 0.4, 0.6, 0.8]
PIN_COLORS = {
    0.2: '#3498db',   # Blue
    0.4: '#2ecc71',   # Green
    0.6: '#e67e22',   # Orange
    0.8: '#e74c3c'    # Red
}


def compute_statistical_noise(results):
    differences = []

    for param1, param2 in IDENTICAL_PAIRS:
        # Skip note_hold_time (not in phase params)
        if param1 == 'note_hold_time' or param2 == 'note_hold_time':
            continue
        if param1 in results and param2 in results:
            max1 = results[param1]['max_residual']
            max2 = results[param2]['max_residual']
            diff = abs(max1 - max2)
            differences.append((param1, param2, max1, max2, diff))

    if not differences:
        return None, []

    avg_noise = np.mean([d[4] for d in differences])
    return avg_noise, differences


def run_phase_sweep(cfg, output_dir, resume=False):
    num_contexts = cfg['num_contexts']
    sweep_values = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.1, ..., 1.0
    duration = cfg['duration']
    suffix = cfg['output_suffix']

    # Checkpoint file
    checkpoint_file = output_dir / f"phase_checkpoint_{suffix}.pkl"

    # Load checkpoint if resuming
    if resume and checkpoint_file.exists():
        print("="*80)
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        results = checkpoint['results']
        checkpoint_contexts = checkpoint.get('num_contexts', 0)
        print(f"Loaded checkpoint: {len(results)} params, {checkpoint_contexts} contexts completed")

        # Check if we need MORE contexts than checkpoint has
        if num_contexts > checkpoint_contexts:
            print(f"[INFO] Increasing contexts from {checkpoint_contexts} to {num_contexts}")
            print(f"[INFO] Will add {num_contexts - checkpoint_contexts} more contexts per parameter")
        print("="*80 + "\n")
    else:
        results = {}
        print("="*80)
        print("PHASE CANCELLATION SWEEP - MULTIPLE REFERENCE POINTS")
        print("="*80)
        print(f"Contexts per pin: {num_contexts}")
        print(f"Pin values: {PIN_VALUES}")
        print(f"Sweep points: {len(sweep_values)}")
        print(f"Total: {len(PHASE_PARAMS)} x {len(PIN_VALUES)} pins x {num_contexts} contexts x {len(sweep_values)} sweeps")
        print("="*80 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    synth = ProSynthesizer(batch_size=1, sample_rate=44100, device=device)

    synth_params_order = [p for p in config.SYNTH_PARAM_ORDER if p != 'note']

    for param_idx, (param_name, param_position) in enumerate(PHASE_PARAMS, 1):
        # Check if we have existing data for this parameter
        if param_name in results:
            existing_pin_results = results[param_name]['pin_results']
            # Get number of contexts already completed
            first_pin = PIN_VALUES[0]
            existing_contexts = len(existing_pin_results[first_pin]['context_curves'])

            if existing_contexts >= num_contexts:
                print(f"[{param_idx:2d}/{len(PHASE_PARAMS)}] {param_name:20s} [SKIPPED - {existing_contexts} contexts done]")
                continue
            else:
                print(f"[{param_idx:2d}/{len(PHASE_PARAMS)}] {param_name:20s} [RESUME - {existing_contexts}/{num_contexts}] ", end='', flush=True)
                pin_results = existing_pin_results
                start_context = existing_contexts
        else:
            print(f"[{param_idx:2d}/{len(PHASE_PARAMS)}] {param_name:20s} ", end='', flush=True)
            pin_results = {}
            start_context = 0

        for pin_value in PIN_VALUES:
            # Initialize or get existing context_curves
            if pin_value not in pin_results:
                pin_results[pin_value] = {'context_curves': []}

            context_curves = pin_results[pin_value]['context_curves']

            for context_idx in range(start_context, num_contexts):
                # Generate random context
                context_params = np.random.uniform(0, 1, len(synth_params_order))
                context_params[33] = 0.0  # Force noise_level=0 for deterministic synthesis

                # Reference: context with param=pin_value
                ref_params = context_params.copy()
                ref_params[param_position] = pin_value
                ref_tensor = torch.from_numpy(ref_params).unsqueeze(0).float().to(device)
                ref_audio = synth.synthesize_batch(ref_tensor, duration=duration, note=60, velocity=0.8)
                ref_audio_np = ref_audio[0].cpu().numpy()
                ref_rms = np.sqrt(np.mean(ref_audio_np ** 2))

                # Sweep parameter values
                residuals = []
                for sweep_val in sweep_values:
                    test_params = context_params.copy()
                    test_params[param_position] = sweep_val

                    test_tensor = torch.from_numpy(test_params).unsqueeze(0).float().to(device)
                    test_audio = synth.synthesize_batch(test_tensor, duration=duration, note=60, velocity=0.8)
                    test_audio_np = test_audio[0].cpu().numpy()

                    # Phase cancellation
                    min_len = min(len(ref_audio_np), len(test_audio_np))
                    residual_signal = ref_audio_np[:min_len] + (-test_audio_np[:min_len])
                    residual_rms = np.sqrt(np.mean(residual_signal ** 2))

                    # Normalize
                    if ref_rms > 1e-6:
                        residual_ratio = residual_rms / ref_rms
                    else:
                        residual_ratio = 0.0

                    residuals.append(residual_ratio)

                context_curves.append(residuals)

        # After all contexts for all pins, compute final statistics
        for pin_value in PIN_VALUES:
            context_curves = pin_results[pin_value]['context_curves']
            # Convert to array
            context_curves_array = np.array(context_curves)  # shape: (num_contexts, 21)

            # Calculate statistics
            mean_curve = np.mean(context_curves_array, axis=0)
            std_curve = np.std(context_curves_array, axis=0)

            pin_results[pin_value] = {
                'context_curves': context_curves,  # Keep as list for appending
                'mean_curve': mean_curve,
                'std_curve': std_curve
            }

        # Overall strength = average max across all pin values
        max_residuals = []
        for pin_value in PIN_VALUES:
            mean_curve = pin_results[pin_value]['mean_curve']
            max_res = np.max(mean_curve)
            max_residuals.append(max_res)

        overall_max = np.mean(max_residuals)

        results[param_name] = {
            'max_residual': overall_max,
            'pin_results': pin_results,
            'position': param_position
        }

        # Status
        if overall_max > PHASE_THRESHOLD_STRONG:
            status = "[STRONG]"
        elif overall_max > PHASE_THRESHOLD_WORKS:
            status = "[WORKS]"
        else:
            status = "[BROKEN]"

        print(f"Max: {overall_max:.4f}  {status}")

        # Save checkpoint after each parameter
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'results': results,
            'num_contexts': num_contexts,
            'suffix': suffix
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

    # Sort by max residual
    sorted_params = sorted(results.items(), key=lambda x: x[1]['max_residual'], reverse=True)

    # Compute statistical noise
    avg_noise, noise_details = compute_statistical_noise(results)

    print("\n" + "="*80)
    print("STATISTICAL NOISE VALIDATION")
    print("="*80)

    if avg_noise is not None:
        print(f"Average difference between identical pairs: {avg_noise:.4f}")
        print(f"Noise threshold (phase): {NOISE_THRESHOLD:.4f}")

        if avg_noise > NOISE_THRESHOLD:
            print(f"\n[WARNING] Statistical noise TOO HIGH!")
            print(f"Recommend increasing num_contexts (currently {num_contexts})")
        else:
            print(f"\n[OK] Statistical noise acceptable")

        print("\nIdentical parameter pairs:")
        for p1, p2, m1, m2, diff in noise_details:
            status = "[OK]" if diff <= NOISE_THRESHOLD else "[HIGH]"
            print(f"  {p1:20s} vs {p2:20s}  |  {m1:.4f} vs {m2:.4f}  |  diff={diff:.4f} {status}")
    else:
        print("No identical pairs found to validate")

    print("="*80)

    print("\n" + "="*80)
    print("RANKED RESULTS")
    print("="*80)

    for rank, (param_name, data) in enumerate(sorted_params, 1):
        max_res = data['max_residual']
        if max_res > PHASE_THRESHOLD_STRONG:
            status = "[STRONG]"
        elif max_res > PHASE_THRESHOLD_WORKS:
            status = "[WORKS]"
        else:
            status = "[BROKEN]"

        print(f"{rank:2d}. {param_name:<25s} {max_res:.4f}  {status}")

    print("="*80)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"phase_cancellation_results_{suffix}.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PHASE CANCELLATION SWEEP - MULTIPLE REFERENCE POINTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Contexts per pin: {num_contexts}\n")
        f.write(f"Pin values: {PIN_VALUES}\n")
        f.write(f"Sweep points: {len(sweep_values)}\n\n")

        # Statistical noise validation
        f.write("="*80 + "\n")
        f.write("STATISTICAL NOISE VALIDATION\n")
        f.write("="*80 + "\n")
        if avg_noise is not None:
            f.write(f"Average difference between identical pairs: {avg_noise:.4f}\n")
            f.write(f"Noise threshold: {NOISE_THRESHOLD:.4f}\n")
            if avg_noise > NOISE_THRESHOLD:
                f.write(f"\n[WARNING] Statistical noise TOO HIGH!\n")
                f.write(f"Recommend increasing num_contexts (currently {num_contexts})\n")
            else:
                f.write(f"\n[OK] Statistical noise acceptable\n")
            f.write("\nIdentical parameter pairs:\n")
            for p1, p2, m1, m2, diff in noise_details:
                status = "[OK]" if diff <= NOISE_THRESHOLD else "[HIGH]"
                f.write(f"  {p1:20s} vs {p2:20s}  |  {m1:.4f} vs {m2:.4f}  |  diff={diff:.4f} {status}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("RANKED RESULTS\n")
        f.write("="*80 + "\n")

        for rank, (param_name, data) in enumerate(sorted_params, 1):
            max_res = data['max_residual']
            if max_res > PHASE_THRESHOLD_STRONG:
                status = "[STRONG]"
            elif max_res > PHASE_THRESHOLD_WORKS:
                status = "[WORKS]"
            else:
                status = "[BROKEN]"

            f.write(f"{rank:2d}. {param_name:<30s} {max_res:.4f}  {status}\n")

    print(f"\n[SAVED] {results_file}")

    # === PLOTS ===

    # Grid plot 1: WITH FADED CURVES
    print("\n[PLOTTING] Creating grid with individual curves...")

    n_rows, n_cols = 6, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))
    axes = axes.flatten()

    for idx, (param_name, data) in enumerate(sorted_params):
        ax = axes[idx]

        # Plot curves for each pin value
        for pin_value in PIN_VALUES:
            color = PIN_COLORS[pin_value]
            pin_data = data['pin_results'][pin_value]

            # Individual context curves (faded)
            for context_curve in pin_data['context_curves']:
                ax.plot(sweep_values, context_curve, color=color, alpha=0.1, linewidth=0.8)

            # Vertical line at pin value
            ax.axvline(x=pin_value, color=color, linewidth=2, alpha=0.7, linestyle='-')

        # Threshold
        ax.axhline(y=PHASE_THRESHOLD_WORKS, color='orange', alpha=0.3, linewidth=1, linestyle='--')

        ax.set_title(f"{idx+1}. {param_name}\nMax: {data['max_residual']:.4f}",
                     fontsize=7, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Param Value', fontsize=7)
        ax.set_ylabel('Residual', fontsize=7)

    # Hide unused
    for idx in range(len(sorted_params), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    grid_with_curves_path = output_dir / f"phase_cancellation_grid_with_curves_{suffix}.png"
    plt.savefig(grid_with_curves_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {grid_with_curves_path}")

    # Grid plot 2: MEANS ONLY
    print("[PLOTTING] Creating grid with MEANS ONLY...")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 24))
    axes = axes.flatten()

    for idx, (param_name, data) in enumerate(sorted_params):
        ax = axes[idx]

        # Plot MEAN curves for each pin value (bold)
        for pin_value in PIN_VALUES:
            color = PIN_COLORS[pin_value]
            pin_data = data['pin_results'][pin_value]

            # MEAN curve ONLY (bold)
            ax.plot(sweep_values, pin_data['mean_curve'], color=color, linewidth=2.5,
                    alpha=0.9, zorder=10)

            # Vertical line at pin value
            ax.axvline(x=pin_value, color=color, linewidth=2, alpha=0.7, linestyle='-')

        # Threshold
        ax.axhline(y=PHASE_THRESHOLD_WORKS, color='orange', alpha=0.3, linewidth=1, linestyle='--')

        ax.set_title(f"{idx+1}. {param_name}\nMax: {data['max_residual']:.4f}",
                     fontsize=7, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Param Value', fontsize=7)
        ax.set_ylabel('Residual', fontsize=7)

    # Hide unused
    for idx in range(len(sorted_params), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    grid_means_path = output_dir / f"phase_cancellation_grid_means_{suffix}.png"
    plt.savefig(grid_means_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {grid_means_path}")

    # Individual plots
    print("[PLOTTING] Creating individual plots...")

    individual_dir = output_dir / "individual_phase"
    individual_dir.mkdir(exist_ok=True)

    for rank, (param_name, data) in enumerate(sorted_params, 1):
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot for each pin value
        for pin_value in PIN_VALUES:
            color = PIN_COLORS[pin_value]
            pin_data = data['pin_results'][pin_value]

            # Individual curves (faded)
            for i, context_curve in enumerate(pin_data['context_curves']):
                label = f'Pin {pin_value}' if i == 0 else None
                ax.plot(sweep_values, context_curve, color=color, alpha=0.15, linewidth=1.5, label=label)

            # Mean curve (bold)
            ax.plot(sweep_values, pin_data['mean_curve'], color=color, linewidth=3,
                    label=f'Pin {pin_value} (mean)', zorder=10)

            # Vertical line at pin
            ax.axvline(x=pin_value, color=color, linewidth=2.5, alpha=0.8, linestyle='-', zorder=5)

        # Threshold
        ax.axhline(y=PHASE_THRESHOLD_WORKS, color='orange', alpha=0.5, linewidth=2,
                   linestyle='--', label=f'Works threshold ({PHASE_THRESHOLD_WORKS})')

        ax.set_title(f"{param_name}\nMax Residual: {data['max_residual']:.4f}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel('Phase Cancellation Residual', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        individual_path = individual_dir / f"phase_cancellation_{param_name}_{suffix}.png"
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {len(sorted_params)} plots to {individual_dir}/")

    # Bar chart
    print("[PLOTTING] Creating bar chart...")

    fig, ax = plt.subplots(figsize=(14, 10))

    param_names = [p[0] for p in sorted_params]
    max_residuals = [p[1]['max_residual'] for p in sorted_params]

    colors = []
    for r in max_residuals:
        if r > PHASE_THRESHOLD_STRONG:
            colors.append('green')
        elif r > PHASE_THRESHOLD_WORKS:
            colors.append('blue')
        else:
            colors.append('red')

    ax.barh(range(len(param_names)), max_residuals, color=colors)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_xlabel('Max Residual', fontsize=12)
    ax.set_title('Parameter Effect Strength (Phase Cancellation)', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    bar_path = output_dir / f"phase_cancellation_ranking_{suffix}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {bar_path}")

    print("\n" + "="*80)
    print("PHASE CANCELLATION SWEEP COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - phase_cancellation_results_{suffix}.txt")
    print(f"  - phase_cancellation_grid_with_curves_{suffix}.png  (with faded curves)")
    print(f"  - phase_cancellation_grid_means_{suffix}.png        (MEANS ONLY)")
    print(f"  - phase_cancellation_ranking_{suffix}.png")
    print(f"  - individual_phase/ ({len(sorted_params)} plots)")
    print("="*80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test (2 contexts)')
    parser.add_argument('--full', action='store_true', help='Full test (50 contexts)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    cfg = QUICK_CONFIG if args.quick else FULL_CONFIG
    output_dir = Path(__file__).parent / "results"
    run_phase_sweep(cfg, output_dir, resume=args.resume)
