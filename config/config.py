import numpy as np

# ==============================================================================
# SYNTHESIZER PARAMETERS
# ==============================================================================

# All 38 synthesizer parameters (36 synth params + 1 note param + 1 timing param)
SYNTH_PARAM_ORDER = [
    'osc1_waveform', 'osc1_octave', 'osc1_detune', 'osc1_level',
    'osc2_waveform', 'osc2_octave', 'osc2_detune', 'osc2_level', 'sub_level',
    'pulse_width', 'osc_sync', 'ring_mod',  # NEW: Core timbre shaping
    'filter_cutoff', 'filter_resonance', 'filter_type', 'filter_env_amount', 'filter_drive',  # NEW: Filter saturation
    'amp_attack', 'amp_decay', 'amp_sustain', 'amp_release',
    'filter_attack', 'filter_decay', 'filter_sustain', 'filter_release',
    'lfo1_rate', 'lfo1_to_filter',
    'distortion', 'delay_mix',
    'delay_time', 'delay_feedback',  # NEW: Parameterized delay
    'lfo1_to_pitch', 'filter_keytrack', 'noise_level',  # NEW: Core additions
    'osc_fm_amount', 'unison_detune',  # NEW: Advanced features
    'note',  # MIDI note parameter
    'note_hold_time'  # NEW: How long note is held before release (0-1 -> 0.1s-2.0s)
]

# Parameter display names for GUI
PARAM_DISPLAY_NAMES = {
    'osc1_waveform': 'OSC1 Wave',
    'osc1_octave': 'OSC1 Octave',
    'osc1_detune': 'OSC1 Detune',
    'osc1_level': 'OSC1 Level',
    'osc2_waveform': 'OSC2 Wave',
    'osc2_octave': 'OSC2 Octave',
    'osc2_detune': 'OSC2 Detune',
    'osc2_level': 'OSC2 Level',
    'sub_level': 'Sub Level',
    'pulse_width': 'PW',
    'osc_sync': 'Sync',
    'ring_mod': 'Ring Mod',
    'filter_cutoff': 'Filter Cutoff',
    'filter_resonance': 'Filter Res',
    'filter_type': 'Filter Type',
    'filter_env_amount': 'Filter Env',
    'filter_drive': 'Drive',
    'amp_attack': 'Amp Attack',
    'amp_decay': 'Amp Decay',
    'amp_sustain': 'Amp Sustain',
    'amp_release': 'Amp Release',
    'filter_attack': 'Flt Attack',
    'filter_decay': 'Flt Decay',
    'filter_sustain': 'Flt Sustain',
    'filter_release': 'Flt Release',
    'lfo1_rate': 'LFO Rate',
    'lfo1_to_filter': 'LFO Depth',
    'distortion': 'Distortion',
    'delay_mix': 'Delay Mix',
    'delay_time': 'Delay Time',
    'delay_feedback': 'Delay FB',
    'lfo1_to_pitch': 'LFO Pitch',
    'filter_keytrack': 'Key Track',
    'noise_level': 'Noise',
    'osc_fm_amount': 'FM Amount',
    'unison_detune': 'Unison',
    'note': 'Note',
    'note_hold_time': 'Hold Time'
}

# Default values for all 38 parameters (normalized 0-1)
SYNTH_PARAM_DEFAULTS = {
    # Oscillators
    'osc1_waveform': 0.5,
    'osc1_octave': 0.5,
    'osc1_detune': 0.5,
    'osc1_level': 0.8,
    'osc2_waveform': 0.5,
    'osc2_octave': 0.625,
    'osc2_detune': 0.52,
    'osc2_level': 0.6,
    'sub_level': 0.3,
    # Timbre shaping
    'pulse_width': 0.5,  # 50% duty cycle (square wave)
    'osc_sync': 0.0,  # Off by default
    'ring_mod': 0.0,  # Off by default
    # Filter
    'filter_cutoff': 0.7,
    'filter_resonance': 0.3,
    'filter_type': 0.0,
    'filter_env_amount': 0.5,
    'filter_drive': 0.0,  # No drive by default
    # Amp envelope
    'amp_attack': 0.1,
    'amp_decay': 0.15,
    'amp_sustain': 0.7,
    'amp_release': 0.25,
    # Filter envelope
    'filter_attack': 0.05,
    'filter_decay': 0.4,
    'filter_sustain': 0.3,
    'filter_release': 0.5,
    # Modulation
    'lfo1_rate': 0.15,
    'lfo1_to_filter': 0.3,
    # Effects
    'distortion': 0.0,
    'delay_mix': 0.0,
    'delay_time': 0.3,  # NEW: Default ~100ms delay
    'delay_feedback': 0.3,  # NEW: 30% feedback
    # Advanced modulation
    'lfo1_to_pitch': 0.0,  # NEW: No pitch modulation by default
    'filter_keytrack': 0.5,  # NEW: Moderate key tracking
    'noise_level': 0.0,  # NEW: No noise by default
    # Advanced synthesis
    'osc_fm_amount': 0.0,  # NEW: No FM by default
    'unison_detune': 0.0,  # NEW: No unison by default
    # Note
    'note': 0.5,  # Default to middle C (MIDI 60)
    # Timing
    'note_hold_time': 0.25,  # NEW: Default 0.5s hold time (0-1 maps to 0.1s-2.0s)
}

# ==============================================================================
# AUDIO SETTINGS
# ==============================================================================

SAMPLE_RATE = 44100              # Audio sample rate (Hz)
DURATION = 0.5                   # Audio duration for optimization (seconds)
DURATION_GUI = 3.0               # Audio duration for GUI playback (seconds)
DEVICE = 'cuda'                  # 'cuda' or 'cpu' - GPU recommended

# ==============================================================================
# SIMILARITY METRIC SELECTION
# ==============================================================================

# Choose which similarity metric to use for analysis
# Options: 'panns', 'clap'
SIMILARITY_METRIC = 'panns'  # Default to PANNs (validated, works well)

# ==============================================================================
# NOTE PARAMETER CONVERSION
# ==============================================================================

# MIDI note range
NOTE_MIN = 36  # C2 (65.4 Hz)
NOTE_MAX = 84  # C6 (1046.5 Hz)

def note_param_to_midi(note_param):
    midi_note = int(NOTE_MIN + note_param * (NOTE_MAX - NOTE_MIN))
    return np.clip(midi_note, NOTE_MIN, NOTE_MAX)

def midi_to_note_param(midi_note):
    note_param = (midi_note - NOTE_MIN) / (NOTE_MAX - NOTE_MIN)
    return np.clip(note_param, 0.0, 1.0)

def midi_to_note_name(midi_note):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]
    return f"{note}{octave}"

def note_hold_time_to_seconds(hold_param):
    MIN_HOLD = 0.1  # 100ms minimum
    MAX_HOLD = 2.0  # 2 seconds maximum
    return MIN_HOLD + hold_param * (MAX_HOLD - MIN_HOLD)

# ==============================================================================
# GUI SETTINGS
# ==============================================================================

# Window dimensions
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900

# Colors (RGB) - Professional VST-style
COLOR_BG = (25, 25, 30)
COLOR_PANEL = (35, 35, 40)
COLOR_SECTION = (40, 42, 48)
COLOR_KNOB = (60, 65, 75)
COLOR_KNOB_ACTIVE = (80, 140, 200)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (130, 130, 140)
COLOR_ACCENT = (80, 160, 240)
COLOR_SECTION_BORDER = (60, 65, 75)
COLOR_KEY_WHITE = (240, 240, 240)
COLOR_KEY_BLACK = (20, 20, 20)
COLOR_KEY_PRESSED = (80, 160, 240)

# Fonts
FONT_SIZE_TITLE = 28
FONT_SIZE_SECTION = 18
FONT_SIZE_LARGE = 16
FONT_SIZE_MEDIUM = 14
FONT_SIZE_SMALL = 11

# Layout - VST-style sections
SECTION_MARGIN = 20
SECTION_PADDING = 15
KNOB_RADIUS = 45
KNOB_SPACING_X = 90
KNOB_SPACING_Y = 110

# Piano keyboard
KEY_WHITE_WIDTH = 40
KEY_WHITE_HEIGHT = 150
KEY_BLACK_WIDTH = 24
KEY_BLACK_HEIGHT = 95

# Keyboard mapping for piano keys (extended range: -5 to +17)
# White keys: Z X C | A S D F G H J K | L M N = 14 white keys
# Black keys: Q 2 | W E T Y U | I O P = 10 black keys
PIANO_KEY_MAPPING = {
    # Lower 5 notes (below original A)
    'z': -5,  # G
    'q': -4,  # G#
    'x': -3,  # A
    '2': -2,  # A#
    'c': -1,  # B
    # Original keys (A-K = notes 0-12)
    'a': 0,   # C
    'w': 1,   # C#
    's': 2,   # D
    'e': 3,   # D#
    'd': 4,   # E
    'f': 5,   # F
    't': 6,   # F#
    'g': 7,   # G
    'y': 8,   # G#
    'h': 9,   # A
    'u': 10,  # A#
    'j': 11,  # B
    'k': 12,  # C (next octave)
    # Upper 6 notes (above original K)
    'i': 13,  # C#
    'l': 14,  # D
    'o': 15,  # D#
    'm': 16,  # E
    'p': 17,  # F
    'n': 18,  # F#
}

# Starting octave for keyboard (MIDI note for 'a' key)
KEYBOARD_BASE_NOTE = 60  # Middle C (C4)

# ==============================================================================
# PRESET DIRECTORIES
# ==============================================================================

PRESETS_DIR = 'presets'  # Directory for preset JSON files
