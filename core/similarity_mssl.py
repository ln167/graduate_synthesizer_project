import numpy as np
import librosa
import torch


class MultiScaleSpectralLoss:

    def __init__(self, sample_rate=44100, window_sizes=[128, 512, 2048], use_log=False):
        self.sample_rate = sample_rate
        self.window_sizes = window_sizes
        self.use_log = use_log
        self.target_specs = {}  # Cache target spectrograms for each window size

    def set_target(self, target_audio):
        # Convert torch tensor to numpy if needed
        if isinstance(target_audio, torch.Tensor):
            target_audio = target_audio.cpu().numpy()

        # Ensure 1D
        if target_audio.ndim > 1:
            target_audio = target_audio.squeeze()

        # Compute and cache STFT at each window size
        for window_size in self.window_sizes:
            hop_length = window_size // 4  # 75% overlap
            stft = librosa.stft(target_audio, n_fft=window_size, hop_length=hop_length)
            magnitude = np.abs(stft)  # Discard phase

            # Apply log-scaling if requested
            if self.use_log:
                magnitude = np.log1p(magnitude)  # log(1 + x) for numerical stability

            self.target_specs[window_size] = magnitude

    def compute_distance(self, audio1, audio2):
        # Convert torch tensors to numpy if needed
        if isinstance(audio1, torch.Tensor):
            audio1 = audio1.cpu().numpy()
        if isinstance(audio2, torch.Tensor):
            audio2 = audio2.cpu().numpy()

        # Ensure 1D
        if audio1.ndim > 1:
            audio1 = audio1.squeeze()
        if audio2.ndim > 1:
            audio2 = audio2.squeeze()

        total_loss = 0.0

        # Compute loss at each scale
        for window_size in self.window_sizes:
            hop_length = window_size // 4

            # Compute STFTs
            stft1 = librosa.stft(audio1, n_fft=window_size, hop_length=hop_length)
            stft2 = librosa.stft(audio2, n_fft=window_size, hop_length=hop_length)

            # Extract magnitudes (phase-invariant)
            mag1 = np.abs(stft1)
            mag2 = np.abs(stft2)

            # Apply log-scaling if requested
            if self.use_log:
                mag1 = np.log1p(mag1)
                mag2 = np.log1p(mag2)

            # Trim to shortest length
            min_frames = min(mag1.shape[1], mag2.shape[1])
            mag1 = mag1[:, :min_frames]
            mag2 = mag2[:, :min_frames]

            # L1 distance on magnitudes
            loss = np.mean(np.abs(mag1 - mag2))
            total_loss += loss

        # Average across scales
        return total_loss / len(self.window_sizes)

    def compute_similarity(self, audio1, audio2):
        distance = self.compute_distance(audio1, audio2)
        # Heuristic normalization (might need tuning based on empirical results)
        return max(0.0, 1.0 - distance)
