import numpy as np
from .similarity_mfcc import MFCCSimilarityMetric
from .similarity_spectral import SpectralSimilarityMetric
from .similarity_temporal import TemporalSimilarityMetric


class CombinedSimilarityMetric:

    def __init__(self, sample_rate=44100,
                 weight_mfcc=0.03, weight_spectral=0.27, weight_sc=0.70):
        self.sample_rate = sample_rate

        # Initialize individual metrics
        self.mfcc_metric = MFCCSimilarityMetric(sample_rate=sample_rate)
        self.spectral_metric = SpectralSimilarityMetric(sample_rate=sample_rate)

        # Weights (SynthRL paper values)
        self.weight_mfcc = weight_mfcc
        self.weight_spectral = weight_spectral
        self.weight_sc = weight_sc

    def set_target(self, target_audio):
        self.mfcc_metric.set_target(target_audio)
        self.spectral_metric.set_target(target_audio)

        # Compute spectral centroid for target
        import librosa
        self.target_sc = librosa.feature.spectral_centroid(y=target_audio, sr=self.sample_rate)[0]

    def compute_distance(self, audio1, audio2):
        import librosa

        # Compute individual distances
        mfcc_dist = self.mfcc_metric.compute_distance(audio1, audio2)
        spectral_mae_dist = self.spectral_metric.compute_distance(audio1, audio2)

        # Compute spectral centroid distance
        sc1 = librosa.feature.spectral_centroid(y=audio1, sr=self.sample_rate)[0]
        sc2 = librosa.feature.spectral_centroid(y=audio2, sr=self.sample_rate)[0]
        min_frames = min(len(sc1), len(sc2))
        sc_dist = np.mean(np.abs(sc1[:min_frames] - sc2[:min_frames]))

        # Normalize to similar ranges (heuristic based on typical values)
        mfcc_norm = mfcc_dist / 50.0
        spectral_norm = spectral_mae_dist / 5.0
        sc_norm = sc_dist / 5000.0  # Spectral centroid in Hz, typically 0-5000

        # Weighted sum (SynthRL paper weights)
        combined = (self.weight_mfcc * mfcc_norm +
                   self.weight_spectral * spectral_norm +
                   self.weight_sc * sc_norm)

        return combined
