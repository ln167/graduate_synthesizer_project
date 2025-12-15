import numpy as np
import librosa


class MFCCSimilarityMetric:

    def __init__(self, sample_rate=44100, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.target_mfcc = None

    def set_target(self, target_audio):
        # Compute MFCC for target
        self.target_mfcc = librosa.feature.mfcc(
            y=target_audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

    def compute_distance(self, audio1, audio2):
        # Compute MFCCs
        mfcc1 = librosa.feature.mfcc(
            y=audio1,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        mfcc2 = librosa.feature.mfcc(
            y=audio2,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Match lengths (trim to shorter)
        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc1 = mfcc1[:, :min_frames]
        mfcc2 = mfcc2[:, :min_frames]

        # Compute MAE (Mean Absolute Error)
        distance = np.mean(np.abs(mfcc1 - mfcc2))

        return distance
