# Quick test config (~15 seconds)
QUICK_CONFIG = {
    'num_contexts': 2,
    'sweep_points': 5,
    'duration': 0.5,
    'output_suffix': 'quick',
}

# OSC level validation config (ONLY tests osc1_level and osc2_level)
OSC_LEVEL_CONFIG = {
    'num_contexts': 100,  # 5x longer for better statistics
    'sweep_points': 11,
    'duration': 0.5,
    'output_suffix': 'osc_level',
}

# Full test config (~45-60 minutes, statistically valid)
FULL_CONFIG = {
    'num_contexts': 20,  # 30 contexts for better statistics, still under 1 hour
    'sweep_points': 11,
    'duration': 0.5,
    'output_suffix': 'full',
}

# OSC level parameters only (for quick validation)
OSC_LEVEL_PARAMS = [
    ('osc1_level', 3),
    ('osc2_level', 7),
]

# All parameters to test (name, index in 37-param array excluding 'note')
# This is SYNTH_PARAM_ORDER minus 'note' (37 params for optimization)
ALL_PARAMS = [
    ('osc1_waveform', 0),
    ('osc1_octave', 1),
    ('osc1_detune', 2),
    ('osc1_level', 3),
    ('osc2_waveform', 4),
    ('osc2_octave', 5),
    ('osc2_detune', 6),
    ('osc2_level', 7),
    ('sub_level', 8),
    ('pulse_width', 9),
    ('osc_sync', 10),
    ('ring_mod', 11),
    ('filter_cutoff', 12),
    ('filter_resonance', 13),
    ('filter_type', 14),
    ('filter_env_amount', 15),
    ('filter_drive', 16),
    ('amp_attack', 17),
    ('amp_decay', 18),
    ('amp_sustain', 19),
    ('amp_release', 20),
    ('filter_attack', 21),
    ('filter_decay', 22),
    ('filter_sustain', 23),
    ('filter_release', 24),
    ('lfo1_rate', 25),
    ('lfo1_to_filter', 26),
    ('distortion', 27),
    ('delay_mix', 28),
    ('delay_time', 29),
    ('delay_feedback', 30),
    ('lfo1_to_pitch', 31),
    ('filter_keytrack', 32),
    ('noise_level', 33),
    ('osc_fm_amount', 34),
    ('unison_detune', 35),
    ('note_hold_time', 36),
]

# Sensitivity thresholds (for PANNs)
THRESHOLD_HIGH = 0.3
THRESHOLD_MEDIUM = 0.1
THRESHOLD_LOW = 0.05  # Below this = BLIND

# Phase cancellation parameters (exclude note_hold_time - it changes duration, not timbre)
PHASE_PARAMS = [p for p in ALL_PARAMS if p[0] != 'note_hold_time']

# Phase cancellation thresholds (RMS of residual signal)
# These are relative to reference RMS
PHASE_THRESHOLD_WORKS = 0.01  # > 1% residual = knob works
PHASE_THRESHOLD_STRONG = 0.1  # > 10% residual = strong effect

# Identical parameter pairs for statistical noise validation
# These parameters should have identical or very similar sensitivity
IDENTICAL_PAIRS = [
    ('osc1_waveform', 'osc2_waveform'),
    ('osc1_octave', 'osc2_octave'),
    ('osc1_detune', 'osc2_detune'),
    ('osc1_level', 'osc2_level'),
    ('amp_attack', 'filter_attack'),
    ('amp_decay', 'filter_decay'),
    ('amp_sustain', 'filter_sustain'),
    ('amp_release', 'filter_release'),
]

# Statistical noise threshold
# If average difference between identical pairs exceeds this, need more contexts
NOISE_THRESHOLD = 0.02  # 2% difference maximum
