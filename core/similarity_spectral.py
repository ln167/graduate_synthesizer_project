import numpy as np
import librosa


class SpectralSimilarityMetric:

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_spec = None

    def set_target(self, target_audio):
        # Compute log-magnitude spectrogram
        stft = librosa.stft(target_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        self.target_spec = np.log1p(np.abs(stft))  # log(1 + magnitude) for numerical stability

    def compute_distance(self, audio1, audio2):
        # Compute log-magnitude spectrograms
        stft1 = librosa.stft(audio1, n_fft=self.n_fft, hop_length=self.hop_length)
        spec1 = np.log1p(np.abs(stft1))

        stft2 = librosa.stft(audio2, n_fft=self.n_fft, hop_length=self.hop_length)
        spec2 = np.log1p(np.abs(stft2))

        # Match lengths (trim to shorter)
        min_frames = min(spec1.shape[1], spec2.shape[1])
        spec1 = spec1[:, :min_frames]
        spec2 = spec2[:, :min_frames]

        # Compute L1 distance on log-spectrograms
        distance = np.mean(np.abs(spec1 - spec2))

        return distance
