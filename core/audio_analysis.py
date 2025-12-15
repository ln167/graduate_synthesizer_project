import numpy as np
import librosa


def detect_pitch(audio, sr=44100, fmin=50, fmax=2000):
    # Use pYIN algorithm for pitch detection (robust to noise)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048
    )

    # Filter out unvoiced frames and invalid detections
    valid_mask = (~np.isnan(f0)) & (voiced_flag > 0)

    if not np.any(valid_mask):
        # No pitch detected, default to C4
        print("[PITCH] No clear pitch detected, using default C4 (MIDI 60)")
        return 60, 261.63, 0.0

    valid_f0 = f0[valid_mask]
    valid_confidence = voiced_probs[valid_mask]

    # Weight by confidence and take median (robust to outliers)
    weighted_f0 = np.average(valid_f0, weights=valid_confidence)
    median_confidence = np.median(valid_confidence)

    # Convert frequency to MIDI note
    midi_note = librosa.hz_to_midi(weighted_f0)
    midi_note_rounded = int(np.round(midi_note))

    # Clamp to valid MIDI range (21-108 is piano range)
    midi_note_rounded = np.clip(midi_note_rounded, 21, 108)

    print(f"[PITCH] Detected: {weighted_f0:.1f} Hz = MIDI {midi_note_rounded} "
          f"(confidence: {median_confidence:.2f})")

    return midi_note_rounded, weighted_f0, median_confidence


def estimate_noise_level(audio, sr=44100, percentile=10):
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(D)

    # Focus on high frequency bands (8kHz+) where noise is more prominent
    # than musical content for most instruments
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    high_freq_mask = freq_bins >= 8000

    if not np.any(high_freq_mask):
        # Sample rate too low to analyze high frequencies
        return 0.0

    high_freq_energy = magnitude[high_freq_mask, :]
    low_freq_energy = magnitude[~high_freq_mask, :]

    # RMS energy in each band
    high_rms = np.sqrt(np.mean(high_freq_energy ** 2))
    low_rms = np.sqrt(np.mean(low_freq_energy ** 2))

    # Ratio of high to low frequency energy
    # Pure tones have low ratio, noisy signals have high ratio
    if low_rms > 1e-6:
        noise_ratio = high_rms / low_rms
    else:
        noise_ratio = 0.0

    # Also check temporal RMS variation
    # Noise has consistent RMS, music has more variation
    frame_rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)[0]
    rms_std = np.std(frame_rms)
    rms_mean = np.mean(frame_rms)
    rms_variation = rms_std / (rms_mean + 1e-6)

    # Low variation = consistent = likely noise
    # High variation = dynamic = likely clean signal

    # Combine metrics (heuristic)
    # High noise_ratio + low rms_variation = noisy
    noise_score = noise_ratio * (1.0 - np.clip(rms_variation, 0, 1))

    # Map to 0-1 range (empirical thresholds)
    # 0.0-0.05: very clean (lock to 0)
    # 0.05-0.2: some noise (allow small values)
    # 0.2+: noisy (allow full range)
    noise_level = np.clip(noise_score / 0.3, 0.0, 1.0)

    print(f"[NOISE] High/Low freq ratio: {noise_ratio:.4f}, "
          f"RMS variation: {rms_variation:.4f}, "
          f"Estimated noise level: {noise_level:.4f}")

    return noise_level


def analyze_audio_for_matching(audio, sr=44100):
    midi_note, frequency, pitch_confidence = detect_pitch(audio, sr)
    noise_level = estimate_noise_level(audio, sr)

    # Lock noise to 0 if detected level is low (< 0.1)
    lock_noise = noise_level < 0.1

    result = {
        'midi_note': midi_note,
        'frequency': frequency,
        'pitch_confidence': pitch_confidence,
        'noise_level': noise_level,
        'lock_noise': lock_noise
    }

    print(f"\n[ANALYSIS SUMMARY]")
    print(f"  Pitch: {frequency:.1f} Hz (MIDI {midi_note})")
    print(f"  Noise: {'LOCKED TO 0' if lock_noise else f'{noise_level:.2f} (allowed)'}")

    return result
