import numpy as np
import librosa


class TemporalSimilarityMetric:

    def __init__(self, sample_rate=44100, frame_length=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.target_envelope = None

    def set_target(self, target_audio):
        # Compute RMS energy envelope
        self.target_envelope = librosa.feature.rms(
            y=target_audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]  # Extract first row

    def compute_distance(self, audio1, audio2):
        # Compute RMS envelopes
        envelope1 = librosa.feature.rms(
            y=audio1,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        envelope2 = librosa.feature.rms(
            y=audio2,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        # Match lengths (trim to shorter)
        min_frames = min(len(envelope1), len(envelope2))
        envelope1 = envelope1[:min_frames]
        envelope2 = envelope2[:min_frames]

        # Compute MAE on envelopes
        distance = np.mean(np.abs(envelope1 - envelope2))

        return distance
