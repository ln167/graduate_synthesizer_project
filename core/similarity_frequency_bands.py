import torch
import numpy as np
from scipy import signal


class FrequencyBandEnergySimilarityMetric:

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

        # Define frequency bands (same as diagnostic analysis)
        self.bands = {
            'Sub': (20, 200),
            'Low': (200, 800),
            'Mid': (800, 4000),
            'High': (4000, 20000)
        }

    def _compute_band_energies(self, audio):
        energies = []

        for band_name, (low_freq, high_freq) in self.bands.items():
            # Design Butterworth bandpass filter
            sos = signal.butter(
                4,  # 4th order filter
                [low_freq, high_freq],
                'bandpass',
                fs=self.sample_rate,
                output='sos'
            )

            # Apply filter to isolate this frequency band
            band_signal = signal.sosfilt(sos, audio)

            # Compute RMS energy in this band
            rms_energy = np.sqrt(np.mean(band_signal**2))
            energies.append(rms_energy)

        return np.array(energies)

    def compute_similarity(self, audio1, audio2):
        # Convert to numpy if needed
        if isinstance(audio1, torch.Tensor):
            audio1 = audio1.cpu().numpy()
        if isinstance(audio2, torch.Tensor):
            audio2 = audio2.cpu().numpy()

        # Flatten if needed
        audio1 = audio1.flatten()
        audio2 = audio2.flatten()

        # Compute energies for each band
        energies1 = self._compute_band_energies(audio1)
        energies2 = self._compute_band_energies(audio2)

        # Normalize to get energy distribution (shape, not absolute level)
        total1 = energies1.sum() + 1e-10
        total2 = energies2.sum() + 1e-10

        dist1 = energies1 / total1
        dist2 = energies2 / total2

        # Compute L1 distance between distributions
        distance = np.abs(dist1 - dist2).mean()

        # Convert distance to similarity [0, 1]
        # distance=0 -> similarity=1, distance=1 -> similarity=0
        similarity = 1.0 - distance

        return similarity

    def compute_distance(self, audio1, audio2):
        return 1.0 - self.compute_similarity(audio1, audio2)


# For compatibility with metric factory
def create_metric(**kwargs):
    return FrequencyBandEnergySimilarityMetric(**kwargs)


if __name__ == "__main__":
    # Quick self-test
    print("Testing Frequency Band Energy Metric...")

    # Generate test signals
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))

    # Signal 1: Low frequency (200 Hz)
    signal1 = np.sin(2 * np.pi * 200 * t)

    # Signal 2: High frequency (4000 Hz)
    signal2 = np.sin(2 * np.pi * 4000 * t)

    # Signal 3: Same as signal 1
    signal3 = signal1.copy()

    metric = FrequencyBandEnergySimilarityMetric(sample_rate=sr)

    # Test 1: Same signal should have similarity ~1.0
    sim_same = metric.compute_similarity(signal1, signal3)
    print(f"Same signal similarity: {sim_same:.4f} (expect ~1.0)")

    # Test 2: Different frequency signals should have lower similarity
    sim_diff = metric.compute_similarity(signal1, signal2)
    print(f"Different frequency similarity: {sim_diff:.4f} (expect <1.0)")

    # Test 3: Check that distance + similarity = 1.0
    dist = metric.compute_distance(signal1, signal2)
    print(f"Distance: {dist:.4f}, Similarity: {sim_diff:.4f}, Sum: {dist + sim_diff:.4f} (expect 1.0)")

    print("[OK] Frequency Band Energy Metric test complete")
