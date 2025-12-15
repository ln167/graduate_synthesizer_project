import numpy as np

try:
    from .synthesizer import ProSynthesizer
except ImportError:
    # Running as script
    from synthesizer import ProSynthesizer


class RealTimeSynth:

    def __init__(self, num_voices=8, sample_rate=44100):
        self.num_voices = num_voices
        self.sample_rate = sample_rate

        # Create single core synthesizer (CPU only for real-time audio)
        self.core = ProSynthesizer(batch_size=1, sample_rate=sample_rate, device='cpu')

        # Create voice pool
        self.voices = [self.core.create_voice(voice_id=i, realtime_mode=True) for i in range(num_voices)]

        # Track active voices: list of (voice, key_id) tuples
        self.active_voices = []

        # Track available voices (voice pool)
        self.available_voices = list(self.voices)

        # Current parameters (shared by all voices)
        self.current_params = None

    def set_parameters(self, params_dict):
        # Convert dict to array in correct order
        try:
            from ..config import config
            param_order = config.SYNTH_PARAM_ORDER
        except ImportError:
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from config import config
            param_order = config.SYNTH_PARAM_ORDER

        # Initialize with defaults instead of zeros
        param_array = np.zeros(38, dtype=np.float32)
        for i, param_name in enumerate(param_order[:38]):
            if param_name in params_dict:
                param_array[i] = params_dict[param_name]
            elif param_name in config.SYNTH_PARAM_DEFAULTS:
                param_array[i] = config.SYNTH_PARAM_DEFAULTS[param_name]

        self.current_params = param_array

        # Update all active voices with new parameters
        for voice, _ in self.active_voices:
            voice.set_parameters(param_array)

    def note_on(self, midi_note, velocity=0.8, key_id=None):
        # Check if this specific key is already playing (prevent re-trigger spam)
        if key_id is not None:
            for voice, vid in self.active_voices:
                if vid == key_id:
                    # Key already playing, don't re-trigger
                    return voice

        # Allocate voice from pool (or steal oldest if pool exhausted)
        if len(self.available_voices) > 0:
            # Take voice from pool
            voice = self.available_voices.pop(0)
        else:
            # Pool exhausted - steal oldest active voice
            voice, old_key_id = self.active_voices.pop(0)
            # Voice is forcibly stopped (no release phase)

        # Set current parameters
        if self.current_params is not None:
            voice.set_parameters(self.current_params)

        # Trigger note on
        voice.note_on(midi_note, velocity)

        # Add to active voices
        self.active_voices.append((voice, key_id))

        return voice

    def note_off(self, key_id):
        for i, (voice, vid) in enumerate(self.active_voices):
            if vid == key_id:
                # Trigger release phase - voice stays active until release completes
                voice.note_off()
                # Voice will be returned to pool by process_audio() when is_finished() is True
                return

    def process_audio(self, num_samples):
        # Start with silence
        buffer = np.zeros(num_samples, dtype=np.float32)

        # Mix all active voices
        voices_to_remove = []
        for voice, key_id in self.active_voices:
            # Generate audio from this voice
            chunk = voice.process(num_samples)
            buffer += chunk

            # Check if voice is finished (release phase complete)
            if voice.is_finished():
                voices_to_remove.append((voice, key_id))

        # Clean up finished voices and return to pool
        for voice_tuple in voices_to_remove:
            # Check if still in active_voices (may have been removed by stop_all())
            if voice_tuple in self.active_voices:
                self.active_voices.remove(voice_tuple)
                voice, _ = voice_tuple
                self.available_voices.append(voice)

        return buffer

    def stop_all(self):
        # Return all active voices to the pool
        for voice, _ in self.active_voices:
            # Reset voice buffer to silence
            voice.audio_buffer = None
            voice.buffer_position = 0
            voice.gate_on = False
            voice.amp_env_stage = 'idle'

        # Move all active voices back to available pool
        self.available_voices.extend([v for v, _ in self.active_voices])
        self.active_voices.clear()

    def get_num_active_voices(self):
        return len(self.active_voices)

    def get_num_available_voices(self):
        return len(self.available_voices)


if __name__ == "__main__":
    print("="*80)
    print("REAL-TIME SYNTHESIZER TEST")
    print("="*80)

    # Create real-time synth
    synth = RealTimeSynth(num_voices=4, sample_rate=44100)
    print(f"Created synthesizer with {synth.num_voices} voices")

    # Set some parameters
    params = {
        'osc1_waveform': 0.0,  # Sine
        'osc1_level': 0.8,
        'filter_cutoff': 0.7,
        'amp_attack': 0.1,
        'amp_release': 0.2,
    }
    synth.set_parameters(params)
    print(f"Set parameters: {params}")

    # Simulate note on/off
    print("\nSimulating note on...")
    synth.note_on(60, velocity=0.8, key_id='C4')
    print(f"Active voices: {synth.get_num_active_voices()}")

    # Generate some audio
    print("\nGenerating 0.5s of audio...")
    chunk_size = 512  # Typical buffer size
    total_samples = int(0.5 * synth.sample_rate)
    chunks = []

    for i in range(0, total_samples, chunk_size):
        samples_to_gen = min(chunk_size, total_samples - i)
        chunk = synth.process_audio(samples_to_gen)
        chunks.append(chunk)

    audio = np.concatenate(chunks)
    print(f"Generated {len(audio)} samples")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"Audio RMS: {np.sqrt(np.mean(audio**2)):.3f}")

    # Trigger note off
    print("\nTriggering note off...")
    synth.note_off('C4')

    # Generate release tail
    print("Generating release tail...")
    for i in range(10):
        chunk = synth.process_audio(chunk_size)
        chunks.append(chunk)

    print(f"\nActive voices after release: {synth.get_num_active_voices()}")
    print(f"Available voices: {synth.get_num_available_voices()}")

    print("\n[OK] Real-time synthesizer test complete!")
