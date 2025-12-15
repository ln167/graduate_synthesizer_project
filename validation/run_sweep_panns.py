import sys
import argparse
from pathlib import Path
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from core.synthesizer import ProSynthesizer
from core.similarity_panns import PANNsSimilarityMetric
from config import config

from sweep_config import (
    QUICK_CONFIG, FULL_CONFIG, OSC_LEVEL_CONFIG,
    ALL_PARAMS, OSC_LEVEL_PARAMS,
    THRESHOLD_HIGH, THRESHOLD_MEDIUM, THRESHOLD_LOW,
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
        if param1 in results and param2 in results:
            sens1 = results[param1]['sensitivity']
            sens2 = results[param2]['sensitivity']
            diff = abs(sens1 - sens2)
            differences.append((param1, param2, sens1, sens2, diff))

    if not differences:
        return None, []

    avg_noise = np.mean([d[4] for d in differences])
    return avg_noise, differences


def run_panns_sweep(cfg, output_dir, resume=False, param_list=None):
    if param_list is None:
        param_list = ALL_PARAMS

    num_contexts = cfg['num_contexts']
    sweep_values = np.linspace(0.0, 1.0, 21)  # 0.0, 0.05, 0.1, ..., 1.0
    duration = cfg['duration']
    suffix = cfg['output_suffix']

    # Checkpoint file
    checkpoint_file = output_dir / f"panns_checkpoint_{suffix}.pkl"

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
        print("PANNS SENSITIVITY SWEEP - MULTIPLE REFERENCE POINTS")
        print("="*80)
        print(f"Contexts per pin: {num_contexts}")
        print(f"Pin values: {PIN_VALUES}")
        print(f"Sweep points: {len(sweep_values)}")
        print(f"Total: {len(param_list)} x {len(PIN_VALUES)} pins x {num_contexts} contexts x {len(sweep_values)} sweeps")
        print("="*80 + "\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    synth = ProSynthesizer(batch_size=1, sample_rate=44100, device=device)
    metric = PANNsSimilarityMetric(device=device, sample_rate=44100)

    synth_params_order = [p for p in config.SYNTH_PARAM_ORDER if p != 'note']

    for param_idx, (param_name, param_position) in enumerate(param_list, 1):
        # Check if we have existing data for this parameter
        if param_name in results:
            existing_pin_results = results[param_name]['pin_results']
            # Get number of contexts already completed
            first_pin = PIN_VALUES[0]
            existing_contexts = len(existing_pin_results[first_pin]['context_curves'])

            if existing_contexts >= num_contexts:
                print(f"[{param_idx:2d}/{len(param_list)}] {param_name:20s} [SKIPPED - {existing_contexts} contexts done]")
                continue
            else:
                print(f"[{param_idx:2d}/{len(param_list)}] {param_name:20s} [RESUME - {existing_contexts}/{num_contexts}] ", end='', flush=True)
                pin_results = existing_pin_results
                start_context = existing_contexts
        else:
            print(f"[{param_idx:2d}/{len(param_list)}] {param_name:20s} ", end='', flush=True)
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

                # Reference: context with param=pin_value
                ref_params = context_params.copy()
                ref_params[param_position] = pin_value
                ref_tensor = torch.from_numpy(ref_params).unsqueeze(0).float().to(device)
                ref_audio = synth.synthesize_batch(ref_tensor, duration=duration, note=60, velocity=0.8)
                ref_audio_np = ref_audio[0].cpu().numpy()

                # Set as PANNs target
                metric.set_target(ref_audio_np)

                # Sweep parameter values
                distances = []
                for sweep_val in sweep_values:
                    test_params = context_params.copy()
                    test_params[param_position] = sweep_val

                    test_tensor = torch.from_numpy(test_params).unsqueeze(0).float().to(device)
                    test_audio = synth.synthesize_batch(test_tensor, duration=duration, note=60, velocity=0.8)
                    test_audio_np = test_audio[0].cpu().numpy()

                    distance = metric.compute_distance(test_audio_np, ref_audio_np)
                    distances.append(distance)

                context_curves.append(distances)

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

        # Overall sensitivity = average range across all pin values
        sensitivities = []
        for pin_value in PIN_VALUES:
            mean_curve = pin_results[pin_value]['mean_curve']
            sensitivity = np.max(mean_curve) - np.min(mean_curve)
            sensitivities.append(sensitivity)

        overall_sensitivity = np.mean(sensitivities)

        results[param_name] = {
            'sensitivity': overall_sensitivity,
            'pin_results': pin_results,
            'position': param_position
        }

        # Status
        if overall_sensitivity > THRESHOLD_HIGH:
            status = "[HIGH]"
        elif overall_sensitivity > THRESHOLD_MEDIUM:
            status = "[MEDIUM]"
        elif overall_sensitivity > THRESHOLD_LOW:
            status = "[LOW]"
        else:
            status = "[BLIND]"

        print(f"Sens: {overall_sensitivity:.4f}  {status}")

        # Save checkpoint after each parameter
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'results': results,
            'num_contexts': num_contexts,
            'suffix': suffix
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

    # Sort by sensitivity
    sorted_params = sorted(results.items(), key=lambda x: x[1]['sensitivity'], reverse=True)

    # Compute statistical noise
    avg_noise, noise_details = compute_statistical_noise(results)

    print("\n" + "="*80)
    print("STATISTICAL NOISE VALIDATION")
    print("="*80)

    if avg_noise is not None:
        print(f"Average difference between identical pairs: {avg_noise:.4f}")
        print(f"Noise threshold: {NOISE_THRESHOLD:.4f}")

        if avg_noise > NOISE_THRESHOLD:
            print(f"\n[WARNING] Statistical noise TOO HIGH!")
            print(f"Recommend increasing num_contexts (currently {num_contexts})")
        else:
            print(f"\n[OK] Statistical noise acceptable")

        print("\nIdentical parameter pairs:")
        for p1, p2, s1, s2, diff in noise_details:
            status = "[OK]" if diff <= NOISE_THRESHOLD else "[HIGH]"
            print(f"  {p1:20s} vs {p2:20s}  |  {s1:.4f} vs {s2:.4f}  |  diff={diff:.4f} {status}")
    else:
        print("No identical pairs found to validate")

    print("="*80)

    print("\n" + "="*80)
    print("RANKED RESULTS")
    print("="*80)

    for rank, (param_name, data) in enumerate(sorted_params, 1):
        sens = data['sensitivity']
        if sens > THRESHOLD_HIGH:
            status = "[HIGH]"
        elif sens > THRESHOLD_MEDIUM:
            status = "[MEDIUM]"
        elif sens > THRESHOLD_LOW:
            status = "[LOW]"
        else:
            status = "[BLIND]"

        print(f"{rank:2d}. {param_name:<25s} {sens:.4f}  {status}")

    print("="*80)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"panns_sensitivity_results_{suffix}.txt"
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PANNS SENSITIVITY SWEEP - MULTIPLE REFERENCE POINTS\n")
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
            for p1, p2, s1, s2, diff in noise_details:
                status = "[OK]" if diff <= NOISE_THRESHOLD else "[HIGH]"
                f.write(f"  {p1:20s} vs {p2:20s}  |  {s1:.4f} vs {s2:.4f}  |  diff={diff:.4f} {status}\n")
        f.write("\n")

        f.write("="*80 + "\n")
        f.write("RANKED RESULTS\n")
        f.write("="*80 + "\n")

        for rank, (param_name, data) in enumerate(sorted_params, 1):
            sens = data['sensitivity']
            if sens > THRESHOLD_HIGH:
                status = "[HIGH]"
            elif sens > THRESHOLD_MEDIUM:
                status = "[MEDIUM]"
            elif sens > THRESHOLD_LOW:
                status = "[LOW]"
            else:
                status = "[BLIND]"

            f.write(f"{rank:2d}. {param_name:<30s} {sens:.4f}  {status}\n")

    print(f"\n[SAVED] {results_file}")

    # === PLOTS ===

    # Grid plot 1: WITH FADED CURVES
    print("\n[PLOTTING] Creating grid with individual curves...")

    n_rows, n_cols = 7, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 26))
    axes = axes.flatten()

    for idx, (param_name, data) in enumerate(sorted_params):
        ax = axes[idx]

        # Plot V-curves for each pin value
        for pin_value in PIN_VALUES:
            color = PIN_COLORS[pin_value]
            pin_data = data['pin_results'][pin_value]

            # Individual context curves (faded)
            for context_curve in pin_data['context_curves']:
                ax.plot(sweep_values, context_curve, color=color, alpha=0.1, linewidth=0.8)

            # Vertical line at pin value
            ax.axvline(x=pin_value, color=color, linewidth=2, alpha=0.7, linestyle='-')

        ax.set_title(f"{idx+1}. {param_name}\nSens: {data['sensitivity']:.4f}",
                     fontsize=7, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Param Value', fontsize=7)
        ax.set_ylabel('PANNs Distance', fontsize=7)

    # Hide unused
    for idx in range(len(sorted_params), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    grid_with_curves_path = output_dir / f"panns_sensitivity_grid_with_curves_{suffix}.png"
    plt.savefig(grid_with_curves_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {grid_with_curves_path}")

    # Grid plot 2: MEANS ONLY
    print("[PLOTTING] Creating grid with MEANS ONLY...")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 26))
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

        ax.set_title(f"{idx+1}. {param_name}\nSens: {data['sensitivity']:.4f}",
                     fontsize=7, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Param Value', fontsize=7)
        ax.set_ylabel('PANNs Distance', fontsize=7)

    # Hide unused
    for idx in range(len(sorted_params), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    grid_means_path = output_dir / f"panns_sensitivity_grid_means_{suffix}.png"
    plt.savefig(grid_means_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {grid_means_path}")

    # Individual plots
    print("[PLOTTING] Creating individual plots...")

    individual_dir = output_dir / "individual_panns"
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

        ax.set_title(f"{param_name}\nSensitivity: {data['sensitivity']:.4f}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Parameter Value', fontsize=12)
        ax.set_ylabel('PANNs Distance', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        plt.tight_layout()
        individual_path = individual_dir / f"panns_sensitivity_{param_name}_{suffix}.png"
        plt.savefig(individual_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {len(sorted_params)} plots to {individual_dir}/")

    # Bar chart
    print("[PLOTTING] Creating bar chart...")

    fig, ax = plt.subplots(figsize=(14, 10))

    param_names = [p[0] for p in sorted_params]
    sensitivities = [p[1]['sensitivity'] for p in sorted_params]

    colors = []
    for s in sensitivities:
        if s > THRESHOLD_HIGH:
            colors.append('green')
        elif s > THRESHOLD_MEDIUM:
            colors.append('blue')
        elif s > THRESHOLD_LOW:
            colors.append('orange')
        else:
            colors.append('red')

    ax.barh(range(len(param_names)), sensitivities, color=colors)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=9)
    ax.set_xlabel('PANNs Sensitivity', fontsize=12)
    ax.set_title('Parameter Sensitivity Ranking', fontsize=14)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    bar_path = output_dir / f"panns_sensitivity_ranking_{suffix}.png"
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {bar_path}")

    print("\n" + "="*80)
    print("PANNS SWEEP COMPLETE")
    print("="*80)
    print(f"\nOutputs:")
    print(f"  - panns_sensitivity_results_{suffix}.txt")
    print(f"  - panns_sensitivity_grid_with_curves_{suffix}.png  (with faded curves)")
    print(f"  - panns_sensitivity_grid_means_{suffix}.png        (MEANS ONLY)")
    print(f"  - panns_sensitivity_ranking_{suffix}.png")
    print(f"  - individual_panns/ ({len(sorted_params)} plots)")
    print("="*80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Quick test (2 contexts)')
    parser.add_argument('--full', action='store_true', help='Full test (50 contexts)')
    parser.add_argument('--osc-level', action='store_true', help='OSC level validation ONLY (50 contexts, 2 params)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    if args.osc_level:
        cfg = OSC_LEVEL_CONFIG
        param_list = OSC_LEVEL_PARAMS
    elif args.quick:
        cfg = QUICK_CONFIG
        param_list = None
    else:
        cfg = FULL_CONFIG
        param_list = None

    output_dir = Path(__file__).parent / "results"
    run_panns_sweep(cfg, output_dir, resume=args.resume, param_list=param_list)
