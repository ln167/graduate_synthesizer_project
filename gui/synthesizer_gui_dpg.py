import dearpygui.dearpygui as dpg
import numpy as np
import sounddevice as sd
import sys
import os
from pathlib import Path
from tkinter import simpledialog
import tkinter as tk
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.realtime_synth import RealTimeSynth
from core.preset_manager import PresetManager, get_base_path
from config import config


class SynthesizerGUI:

    def __init__(self):
        print("[DPG] Initializing DearPyGui Synthesizer...")

        # Initialize preset manager with PyInstaller-aware path
        presets_path = get_base_path() / config.PRESETS_DIR
        self.preset_manager = PresetManager(presets_path)
        self.available_presets = self.preset_manager.list_presets()
        self.current_preset = self.available_presets[0] if self.available_presets else "sine_wave"

        # Real-time synthesizer with voice pooling
        # 6 voices (9ms/voice, under 11.6ms budget)
        self.NUM_VOICES = 6
        self.synth = RealTimeSynth(num_voices=self.NUM_VOICES, sample_rate=config.SAMPLE_RATE)
        self.voice_lock = None

        # Parameter storage
        self.parameters = {param: config.SYNTH_PARAM_DEFAULTS[param]
                          for param in config.SYNTH_PARAM_ORDER}

        # Audio stream
        self.audio_stream = None
        self.buffer_size = 1024

        # Widget tags storage for easy access
        self.knob_tags = {}
        self.piano_button_tags = {}

        # Matching configuration defaults
        self.match_max_evals = 60000
        self.match_target_similarity = 95.0
        self.match_sample_duration = 0.5
        self.match_num_restarts = 3
        self.match_stagnation_gens = 100
        self.user_requested_stop = False
        self.force_cpu_matching = False
        self.optimization_method = "PANNS"  # Default: PANNS only (best quality)
        self.last_matched_audio_path = None  # Track last target file for continue optimization
        self.last_matched_metric = None  # Track which method was used

        print(f"[DPG] Initialized with {self.NUM_VOICES} voice polyphony")
        print("[DPG] GUI initialized successfully")

    def create_gui(self):
        # Create DPG context
        dpg.create_context()

        # Apply VST-style theme
        self._create_theme()

        # Create main window
        with dpg.window(label="VST Synthesizer", tag="main_window"):
            # Title
            dpg.add_text("GPU SYNTHESIZER", color=(80, 160, 240))
            dpg.add_text(f"Preset: {self.current_preset}", tag="preset_display", color=(180, 180, 180))
            dpg.add_spacer(height=10)

            # Main layout - two columns
            with dpg.group(horizontal=True):
                # Left column - Parameter sections
                with dpg.child_window(width=1220, height=900, border=False):
                    self._create_oscillators_section()
                    dpg.add_spacer(height=10)

                    with dpg.group(horizontal=True):
                        self._create_filter_section()
                        dpg.add_spacer(width=10)
                        self._create_envelopes_section()

                    dpg.add_spacer(height=10)

                    with dpg.group(horizontal=True):
                        self._create_modulation_section()
                        dpg.add_spacer(width=10)
                        self._create_generators_section()
                        dpg.add_spacer(width=10)
                        self._create_delay_section()

                    dpg.add_spacer(height=10)

                    with dpg.group(horizontal=True):
                        self._create_effects_section()
                        dpg.add_spacer(width=10)
                        self._create_note_section()

                # Right column - Preset controls
                with dpg.child_window(width=380, height=700, border=False):
                    self._create_preset_controls()

            dpg.add_spacer(height=30)

            # Piano keyboard at bottom
            self._create_piano_keyboard()

            dpg.add_spacer(height=5)

            # Instructions
            dpg.add_text("Controls: A-K keys = Piano | PANIC button = Stop All | ← → = Change Preset | ESC = Quit",
                        color=(130, 130, 140))

        # Set main window as primary
        dpg.set_primary_window("main_window", True)

        # Register keyboard handlers
        self._register_keyboard_handlers()

        # Initialize button highlights for discrete parameters
        self._set_filter_type(0)  # LP is default
        self._set_waveform(0, 0.5)  # OSC1 defaults to 0.5 (will highlight closest match)
        self._set_waveform(1, 0.5)  # OSC2 defaults to 0.5
        self._set_octave(0, 0.5)  # OSC1 octave 0
        self._set_octave(1, 0.625)  # OSC2 octave +1 (0.625 from defaults)

        # Create viewport
        dpg.create_viewport(
            title='GPU Synthesizer - DearPyGui',
            width=1600,
            height=900,
            min_width=1200,
            min_height=700,
            resizable=True
        )

        print("[DPG] GUI created successfully")

    def _create_theme(self):
        with dpg.theme() as vst_theme:
            with dpg.theme_component(dpg.mvAll):
                # Convert RGB tuples to 0-255 values
                # Backgrounds
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, config.COLOR_BG)
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, config.COLOR_PANEL)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, config.COLOR_KNOB)
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (70, 75, 85))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, config.COLOR_KNOB_ACTIVE)

                # Buttons
                dpg.add_theme_color(dpg.mvThemeCol_Button, config.COLOR_SECTION)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 52, 60))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, config.COLOR_ACCENT)

                # Text
                dpg.add_theme_color(dpg.mvThemeCol_Text, config.COLOR_TEXT)

                # Knobs/Sliders
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, config.COLOR_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (100, 180, 255))

                # Borders
                dpg.add_theme_color(dpg.mvThemeCol_Border, config.COLOR_SECTION_BORDER)

        dpg.bind_theme(vst_theme)

        # Create selected button theme for filter type buttons
        with dpg.theme(tag="selected_button_theme") as selected_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, config.COLOR_ACCENT)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (100, 180, 255))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (60, 120, 200))

    def _create_oscillators_section(self):
        with dpg.child_window(height=270, border=True):
            with dpg.group(horizontal=True):
                dpg.add_text("OSCILLATORS", color=config.COLOR_ACCENT)
                dpg.add_spacer(width=20)
                dpg.add_text("Wave: 0.0=Sine | 0.33=Saw | 0.67=Square | 1.0=Triangle", color=config.COLOR_TEXT_DIM)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                # OSC1 - Blue tint
                with dpg.group():
                    dpg.add_text("OSC 1", color=config.COLOR_TEXT_DIM)
                    with dpg.group(horizontal=True):
                        # Waveform selector (4 buttons in 2x2 grid)
                        with dpg.group():
                            dpg.add_text("Wave", color=config.COLOR_TEXT)
                            dpg.add_spacer(height=3)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Sin", width=35, height=25, tag="osc1_wave_sine", callback=lambda: self._set_waveform(0, 0.0))
                                dpg.add_button(label="Saw", width=35, height=25, tag="osc1_wave_saw", callback=lambda: self._set_waveform(0, 0.33))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Sqr", width=35, height=25, tag="osc1_wave_square", callback=lambda: self._set_waveform(0, 0.67))
                                dpg.add_button(label="Tri", width=35, height=25, tag="osc1_wave_tri", callback=lambda: self._set_waveform(0, 1.0))
                        dpg.add_spacer(width=15)
                        # Octave selector (5 buttons horizontal)
                        with dpg.group():
                            dpg.add_text("Octave", color=config.COLOR_TEXT)
                            dpg.add_spacer(height=3)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="-2", width=28, height=25, tag="osc1_oct_m2", callback=lambda: self._set_octave(0, 0.0))
                                dpg.add_button(label="-1", width=28, height=25, tag="osc1_oct_m1", callback=lambda: self._set_octave(0, 0.25))
                                dpg.add_button(label="0", width=28, height=25, tag="osc1_oct_0", callback=lambda: self._set_octave(0, 0.5))
                                dpg.add_button(label="+1", width=28, height=25, tag="osc1_oct_p1", callback=lambda: self._set_octave(0, 0.75))
                                dpg.add_button(label="+2", width=28, height=25, tag="osc1_oct_p2", callback=lambda: self._set_octave(0, 1.0))
                        dpg.add_spacer(width=15)
                        self._add_knob("osc1_detune", "Detune", (70, 85, 100))
                        dpg.add_spacer(width=15)
                        self._add_knob("osc1_level", "Level", (70, 85, 100))

                dpg.add_spacer(width=40)

                # OSC2 - Purple tint
                with dpg.group():
                    dpg.add_text("OSC 2", color=config.COLOR_TEXT_DIM)
                    with dpg.group(horizontal=True):
                        # Waveform selector (4 buttons in 2x2 grid)
                        with dpg.group():
                            dpg.add_text("Wave", color=config.COLOR_TEXT)
                            dpg.add_spacer(height=3)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Sin", width=35, height=25, tag="osc2_wave_sine", callback=lambda: self._set_waveform(1, 0.0))
                                dpg.add_button(label="Saw", width=35, height=25, tag="osc2_wave_saw", callback=lambda: self._set_waveform(1, 0.33))
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="Sqr", width=35, height=25, tag="osc2_wave_square", callback=lambda: self._set_waveform(1, 0.67))
                                dpg.add_button(label="Tri", width=35, height=25, tag="osc2_wave_tri", callback=lambda: self._set_waveform(1, 1.0))
                        dpg.add_spacer(width=15)
                        # Octave selector (5 buttons horizontal)
                        with dpg.group():
                            dpg.add_text("Octave", color=config.COLOR_TEXT)
                            dpg.add_spacer(height=3)
                            with dpg.group(horizontal=True):
                                dpg.add_button(label="-2", width=28, height=25, tag="osc2_oct_m2", callback=lambda: self._set_octave(1, 0.0))
                                dpg.add_button(label="-1", width=28, height=25, tag="osc2_oct_m1", callback=lambda: self._set_octave(1, 0.25))
                                dpg.add_button(label="0", width=28, height=25, tag="osc2_oct_0", callback=lambda: self._set_octave(1, 0.5))
                                dpg.add_button(label="+1", width=28, height=25, tag="osc2_oct_p1", callback=lambda: self._set_octave(1, 0.75))
                                dpg.add_button(label="+2", width=28, height=25, tag="osc2_oct_p2", callback=lambda: self._set_octave(1, 1.0))
                        dpg.add_spacer(width=15)
                        self._add_knob("osc2_detune", "Detune", (90, 70, 100))
                        dpg.add_spacer(width=15)
                        self._add_knob("osc2_level", "Level", (90, 70, 100))

            dpg.add_spacer(height=10)
            dpg.add_text("TIMBRE", color=config.COLOR_TEXT_DIM)
            with dpg.group(horizontal=True):
                self._add_knob("pulse_width", "PW", (100, 80, 70))
                dpg.add_spacer(width=15)
                self._add_knob("osc_sync", "Sync", (100, 80, 70))
                dpg.add_spacer(width=15)
                self._add_knob("ring_mod", "Ring Mod", (100, 80, 70))

    def _create_filter_section(self):
        with dpg.child_window(width=520, height=180, border=True):
            dpg.add_text("FILTER", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                self._add_knob("filter_cutoff", "Cutoff")
                dpg.add_spacer(width=15)
                self._add_knob("filter_resonance", "Resonance")
                dpg.add_spacer(width=15)

                # Filter type buttons (NOT a knob) - 2x2 grid
                with dpg.group():
                    dpg.add_text("Type", color=config.COLOR_TEXT)
                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="LP", width=45, height=30, tag="filter_type_lp", callback=lambda: self._set_filter_type(0))
                        dpg.add_button(label="HP", width=45, height=30, tag="filter_type_hp", callback=lambda: self._set_filter_type(1))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="BP", width=45, height=30, tag="filter_type_bp", callback=lambda: self._set_filter_type(2))
                        dpg.add_button(label="Notch", width=45, height=30, tag="filter_type_notch", callback=lambda: self._set_filter_type(3))

                dpg.add_spacer(width=15)
                self._add_knob("filter_env_amount", "Env Amt")
                dpg.add_spacer(width=15)
                self._add_knob("filter_drive", "Drive")

    def _create_envelopes_section(self):
        with dpg.child_window(width=730, height=180, border=True):
            dpg.add_text("ENVELOPES", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                # Amp Envelope - Green tint
                with dpg.group():
                    dpg.add_text("AMP ENV", color=config.COLOR_TEXT_DIM)
                    with dpg.group(horizontal=True):
                        self._add_knob("amp_attack", "Attack", (70, 100, 70))
                        dpg.add_spacer(width=15)
                        self._add_knob("amp_decay", "Decay", (70, 100, 70))
                        dpg.add_spacer(width=15)
                        self._add_knob("amp_sustain", "Sustain", (70, 100, 70))
                        dpg.add_spacer(width=15)
                        self._add_knob("amp_release", "Release", (70, 100, 70))

                dpg.add_spacer(width=40)

                # Filter Envelope - Orange tint
                with dpg.group():
                    dpg.add_text("FILTER ENV", color=config.COLOR_TEXT_DIM)
                    with dpg.group(horizontal=True):
                        self._add_knob("filter_attack", "Attack", (100, 85, 60))
                        dpg.add_spacer(width=15)
                        self._add_knob("filter_decay", "Decay", (100, 85, 60))
                        dpg.add_spacer(width=15)
                        self._add_knob("filter_sustain", "Sustain", (100, 85, 60))
                        dpg.add_spacer(width=15)
                        self._add_knob("filter_release", "Release", (100, 85, 60))

    def _create_modulation_section(self):
        with dpg.child_window(width=440, height=180, border=True):
            dpg.add_text("MODULATION", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                self._add_knob("lfo1_rate", "LFO Rate")
                dpg.add_spacer(width=15)
                with dpg.group():
                    self._add_knob("lfo1_to_filter", "LFO>Filter")
                    dpg.add_text("(Filter Cutoff)", color=(100, 100, 120), pos=(5, 95))
                dpg.add_spacer(width=15)
                with dpg.group():
                    self._add_knob("lfo1_to_pitch", "LFO>Pitch")
                    dpg.add_text("(Vibrato)", color=(100, 100, 120), pos=(5, 95))
                dpg.add_spacer(width=15)
                self._add_knob("filter_keytrack", "Key Track")

    def _create_generators_section(self):
        with dpg.child_window(width=240, height=180, border=True):
            dpg.add_text("GENERATORS", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                self._add_knob("sub_level", "Sub Osc")
                dpg.add_spacer(width=15)
                self._add_knob("noise_level", "Noise")

    def _create_effects_section(self):
        with dpg.child_window(width=340, height=180, border=True):
            dpg.add_text("EFFECTS", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                self._add_knob("distortion", "Distortion")
                dpg.add_spacer(width=15)
                self._add_knob("osc_fm_amount", "FM Amount")
                dpg.add_spacer(width=15)
                self._add_knob("unison_detune", "Unison")

    def _create_delay_section(self):
        with dpg.child_window(width=340, height=180, border=True):
            dpg.add_text("DELAY", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                self._add_knob("delay_mix", "Mix")
                dpg.add_spacer(width=15)
                self._add_knob("delay_time", "Time")
                dpg.add_spacer(width=15)
                self._add_knob("delay_feedback", "Feedback")

    def _create_note_section(self):
        with dpg.child_window(width=230, height=180, border=True):
            dpg.add_text("NOTE", color=config.COLOR_ACCENT)
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.group(horizontal=True):
                self._add_knob("note", "Base Note")
                dpg.add_spacer(width=15)
                self._add_knob("note_hold_time", "Hold Time")

    def _create_preset_controls(self):
        dpg.add_text("PRESETS", color=config.COLOR_ACCENT)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Preset dropdown
        dpg.add_combo(
            items=self.available_presets,
            default_value=self.current_preset,
            label="Select Preset",
            width=300,
            callback=self._on_preset_selected,
            tag="preset_combo"
        )

        dpg.add_spacer(height=20)

        # Randomize button
        dpg.add_button(
            label="RANDOMIZE ALL",
            width=300,
            height=40,
            callback=self._randomize_parameters
        )

        dpg.add_spacer(height=20)

        # Save to Preset button
        dpg.add_button(
            label="SAVE CURRENT AS PRESET",
            width=300,
            height=40,
            callback=self._save_preset_dialog
        )

        dpg.add_spacer(height=20)

        # Matching configuration (collapsible)
        with dpg.collapsing_header(label="Matching Settings", default_open=False):
            dpg.add_spacer(height=5)

            # Info text
            dpg.add_text("Stops at whichever comes first:", color=config.COLOR_TEXT_DIM)
            dpg.add_text("  Max Evaluations OR Target Similarity", color=config.COLOR_TEXT_DIM)
            dpg.add_spacer(height=10)

            # Optimization method selection
            dpg.add_combo(
                items=[
                    "PANNS",
                    "Temporal+PANNS_Static",
                    "Temporal+PANNS_GradNorm",
                    "TSP_Static",
                    "MSSL",
                    "MSSL_Log"
                ],
                default_value=self.optimization_method,
                label="Optimization Method",
                width=250,
                callback=lambda s, v: setattr(self, 'optimization_method', v),
                tag="optimization_method_combo"
            )

            dpg.add_text("  PANNS", color=config.COLOR_TEXT_DIM)
            dpg.add_text("  Temporal+PANNS", color=config.COLOR_TEXT_DIM)
            dpg.add_text("  TSP (all 3 metrics)", color=config.COLOR_TEXT_DIM)
            dpg.add_text("  MSSL (linear)", color=config.COLOR_TEXT_DIM)
            dpg.add_text("  MSSL_Log (log)", color=config.COLOR_TEXT_DIM)
            dpg.add_text("  Combined: Weighted (mfcc=0.03, spec=0.27, sc=0.70)", color=config.COLOR_TEXT_DIM)

            dpg.add_spacer(height=10)

            # Sample duration
            dpg.add_slider_float(
                label="Sample Duration (s)",
                default_value=self.match_sample_duration,
                min_value=0.1,
                max_value=2.0,
                format="%.2f s",
                width=200,
                tag="match_sample_duration",
                callback=lambda s, v: setattr(self, 'match_sample_duration', v)
            )

            dpg.add_spacer(height=5)

            # Max evaluations
            dpg.add_input_int(
                label="Max Evaluations",
                default_value=self.match_max_evals,
                min_value=1000,
                max_value=200000,
                step=1000,
                width=200,
                tag="match_max_evals",
                callback=lambda s, v: setattr(self, 'match_max_evals', v)
            )

            dpg.add_spacer(height=5)

            # Target similarity
            dpg.add_input_float(
                label="Target Similarity (%)",
                default_value=self.match_target_similarity,
                min_value=50.0,
                max_value=99.9,
                step=0.1,
                format="%.1f",
                width=200,
                tag="match_target_similarity",
                callback=lambda s, v: setattr(self, 'match_target_similarity', v)
            )

            dpg.add_spacer(height=5)

            # Number of restarts
            dpg.add_input_int(
                label="Number of Restarts",
                default_value=self.match_num_restarts,
                min_value=1,
                max_value=10,
                step=1,
                width=200,
                tag="match_num_restarts",
                callback=lambda s, v: setattr(self, 'match_num_restarts', v)
            )

            dpg.add_spacer(height=5)

            # Stagnation limit
            dpg.add_input_int(
                label="Stagnation Limit (gens)",
                default_value=self.match_stagnation_gens,
                min_value=10,
                max_value=200,
                step=10,
                width=200,
                tag="match_stagnation_gens",
                callback=lambda s, v: setattr(self, 'match_stagnation_gens', v)
            )

            dpg.add_spacer(height=10)

            # Force CPU mode checkbox
            dpg.add_checkbox(
                label="Force CPU Mode",
                default_value=self.force_cpu_matching,
                callback=lambda s, v: setattr(self, 'force_cpu_matching', v),
                tag="force_cpu_checkbox"
            )

            dpg.add_spacer(height=5)

        dpg.add_spacer(height=10)

        # Match audio button
        dpg.add_button(
            label="MATCH AUDIO FROM FILE",
            width=300,
            height=40,
            callback=self._on_match_audio_clicked
        )

        dpg.add_spacer(height=10)

        # Continue optimization button
        dpg.add_button(
            label="CONTINUE FROM CURRENT PRESET",
            width=300,
            height=40,
            callback=self._on_continue_optimization_clicked,
            tag="continue_optimization_button"
        )
        dpg.add_text("  Start from current knob values instead of random",
                    color=config.COLOR_TEXT_DIM, tag="continue_hint")

        dpg.add_spacer(height=10)

        # Progress bar (hidden by default)
        dpg.add_progress_bar(tag="match_progress", width=300, show=False)
        dpg.add_text("", tag="match_status", color=config.COLOR_TEXT_DIM, show=False)

        dpg.add_spacer(height=5)

        # Stop early button (hidden by default, shown during optimization)
        dpg.add_button(
            label="STOP EARLY - Save Best Solution",
            width=300,
            height=30,
            callback=self._on_stop_early_clicked,
            tag="stop_early_button",
            show=False
        )

        dpg.add_spacer(height=20)
        dpg.add_separator()
        dpg.add_spacer(height=10)

        # Current preset info
        dpg.add_text("Current Preset:", color=config.COLOR_TEXT_DIM)
        dpg.add_text(self.current_preset, tag="preset_name_display", color=config.COLOR_TEXT)

    def _add_knob(self, param_name, label, color=None, discrete_values=None):
        default_value = self.parameters[param_name]
        tag = f"knob_{param_name}"
        self.knob_tags[param_name] = tag

        with dpg.theme() as knob_theme:
            with dpg.theme_component(dpg.mvKnobFloat):
                if color:
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBg, color)
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, tuple(min(c+20, 255) for c in color))
                    dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, tuple(min(c+40, 255) for c in color))
                    dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, tuple(min(c+60, 255) for c in color))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, config.KNOB_RADIUS, config.KNOB_RADIUS)

        if discrete_values:
            callback = lambda s, v, u: self._on_discrete_knob_changed(s, v, u, discrete_values)
        else:
            callback = self._on_knob_changed

        with dpg.group():
            knob = dpg.add_knob_float(
                label=label,
                default_value=default_value,
                min_value=0.0,
                max_value=1.0,
                tag=tag,
                callback=callback,
                user_data=param_name
            )

            if color:
                dpg.bind_item_theme(knob, knob_theme)

            # Add value label BELOW knob for discrete parameters
            if discrete_values:
                idx = int(default_value * (len(discrete_values) - 1) + 0.5)
                label_tag = f"label_{param_name}"

                # Create text with better styling
                dpg.add_text(
                    discrete_values[idx],
                    tag=label_tag,
                    color=(230, 230, 250)
                )

    def _on_discrete_knob_changed(self, sender, value, param_name, discrete_values):
        # Snap to nearest discrete value
        num_values = len(discrete_values)
        idx = int(value * (num_values - 1) + 0.5)
        idx = min(num_values - 1, max(0, idx))
        snapped_value = idx / (num_values - 1)

        # Update knob to snapped value
        dpg.set_value(sender, snapped_value)

        # Update parameter
        self.parameters[param_name] = snapped_value

        # Update label
        label_tag = f"label_{param_name}"
        if dpg.does_item_exist(label_tag):
            dpg.set_value(label_tag, discrete_values[idx])

        # Update synth parameters
        params_dict = self.parameters
        self.synth.set_parameters(params_dict)

    def _create_piano_keyboard(self):
        # Reduced key sizes
        white_key_width = 28
        white_key_height = 100
        black_key_width = 18
        black_key_height = 65

        with dpg.group(horizontal=True):
            # Create a table for precise layout
            with dpg.table(header_row=False, policy=dpg.mvTable_SizingFixedFit):
                # Create 14 columns for white keys
                for _ in range(14):
                    dpg.add_table_column(init_width_or_weight=white_key_width)

                # Row 1: Black keys (positioned between white keys)
                with dpg.table_row():
                    # Black keys map: Q 2 (lower) | W E T Y U (middle) | I O P (upper)
                    # Positions correspond to white key columns (14 total columns)
                    # Black keys appear after: Z, X, (skip C) | A, S, (skip D), F, G, (skip H,J), K | L, (skip M), (skip N)
                    black_keys_by_column = {
                        0: (-4, 'Q'),   # After Z
                        1: (-2, '2'),   # After X
                        2: None,        # After C - no black key
                        3: (1, 'W'),    # After A
                        4: (3, 'E'),    # After S
                        5: None,        # After D - no black key
                        6: (6, 'T'),    # After F
                        7: (8, 'Y'),    # After G
                        8: (10, 'U'),   # After H
                        9: None,        # After J - no black key
                        10: (13, 'I'),  # After K
                        11: (15, 'O'),  # After L
                        12: None,       # After M - no black key
                        13: (17, 'P'),  # After N
                    }

                    for col_idx in range(14):
                        with dpg.table_cell():
                            black_key_data = black_keys_by_column.get(col_idx)
                            if black_key_data:
                                note_offset, kbd_label = black_key_data
                                key_id = f"black_{note_offset}"
                                tag = f"piano_key_{key_id}"
                                self.piano_button_tags[key_id] = tag

                                dpg.add_button(
                                    label=kbd_label,
                                    width=black_key_width,
                                    height=black_key_height,
                                    tag=tag,
                                    user_data=(note_offset, key_id),
                                    enabled=False  # Disable clicking
                                )

                                # Set black key style
                                with dpg.theme() as black_key_theme:
                                    with dpg.theme_component(dpg.mvButton):
                                        dpg.add_theme_color(dpg.mvThemeCol_Button, config.COLOR_KEY_BLACK)
                                        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (40, 40, 40))
                                        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255))  # White text
                                dpg.bind_item_theme(tag, black_key_theme)

                # Row 2: White keys (extended range)
                with dpg.table_row():
                    # Extended white keys: Z X C | A S D F G H J K | L M N (14 total)
                    # Note offsets relative to middle C: -5, -3, -1, 0, 2, 4, 5, 7, 9, 11, 12, 14, 16, 18
                    white_keys_data = [
                        (-5, 'Z'), (-3, 'X'), (-1, 'C'),  # Lower 3
                        (0, 'A'), (2, 'S'), (4, 'D'), (5, 'F'), (7, 'G'), (9, 'H'), (11, 'J'), (12, 'K'),  # Original
                        (14, 'L'), (16, 'M'), (18, 'N')  # Upper 3
                    ]

                    for note_offset, kbd_label in white_keys_data:
                        key_id = f"white_{note_offset}"
                        tag = f"piano_key_{key_id}"
                        self.piano_button_tags[key_id] = tag

                        with dpg.table_cell():
                            dpg.add_button(
                                label=kbd_label,
                                width=white_key_width,
                                height=white_key_height,
                                tag=tag,
                                user_data=(note_offset, key_id),
                                enabled=False  # Disable clicking
                            )

                            # Set white key style
                            with dpg.theme() as white_key_theme:
                                with dpg.theme_component(dpg.mvButton):
                                    dpg.add_theme_color(dpg.mvThemeCol_Button, config.COLOR_KEY_WHITE)
                                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 255, 255))
                                    dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 0, 0))  # Black text
                            dpg.bind_item_theme(tag, white_key_theme)

            dpg.add_spacer(width=20)

            # STOP ALL button on the side
            with dpg.theme() as panic_theme:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (180, 20, 20))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (220, 30, 30))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (150, 15, 15))
            dpg.add_button(label="STOP\nALL", width=80, height=white_key_height, callback=self._on_panic_button, tag="panic_button")
            dpg.bind_item_theme("panic_button", panic_theme)

    def _register_keyboard_handlers(self):
        # Piano key mapping: lowercase char -> DPG key constant
        PIANO_KEYS = {
            'z': dpg.mvKey_Z, 'x': dpg.mvKey_X, 'c': dpg.mvKey_C,
            'q': dpg.mvKey_Q, '2': dpg.mvKey_2,
            'a': dpg.mvKey_A, 's': dpg.mvKey_S, 'd': dpg.mvKey_D, 'f': dpg.mvKey_F,
            'g': dpg.mvKey_G, 'h': dpg.mvKey_H, 'j': dpg.mvKey_J, 'k': dpg.mvKey_K,
            'w': dpg.mvKey_W, 'e': dpg.mvKey_E, 't': dpg.mvKey_T,
            'y': dpg.mvKey_Y, 'u': dpg.mvKey_U,
            'i': dpg.mvKey_I, 'l': dpg.mvKey_L, 'o': dpg.mvKey_O,
            'm': dpg.mvKey_M, 'p': dpg.mvKey_P, 'n': dpg.mvKey_N
        }

        # Helper to create closures properly
        def make_press_handler(char):
            return lambda: self._handle_key_press(char)

        def make_release_handler(char):
            return lambda: self._handle_key_release(char)

        with dpg.handler_registry():
            # Register all piano key press/release handlers
            for key_char, dpg_key in PIANO_KEYS.items():
                dpg.add_key_press_handler(
                    key=dpg_key,
                    callback=make_press_handler(key_char)
                )
                dpg.add_key_release_handler(
                    key=dpg_key,
                    callback=make_release_handler(key_char)
                )

            # Preset navigation
            dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=self._previous_preset)
            dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=self._next_preset)

            # Quit
            dpg.add_key_press_handler(key=dpg.mvKey_Escape, callback=self._quit_app)

    # ========== CALLBACK HANDLERS ==========

    def _on_knob_changed(self, sender, value, user_data):
        param_name = user_data
        self.parameters[param_name] = value

    def _set_button_parameter(self, param_name, value, button_tags, button_values):
        self.parameters[param_name] = value

        # Update button themes to highlight selected
        for tag, btn_val in zip(button_tags, button_values):
            if abs(btn_val - value) < 0.01:
                dpg.bind_item_theme(tag, "selected_button_theme")
            else:
                dpg.bind_item_theme(tag, 0)

    def _set_filter_type(self, filter_type_value):
        button_tags = ["filter_type_lp", "filter_type_hp", "filter_type_bp", "filter_type_notch"]
        button_values = [0.0, 1/3.0, 2/3.0, 1.0]
        self._set_button_parameter('filter_type', filter_type_value / 3.0, button_tags, button_values)

    def _set_waveform(self, osc_num, waveform_value):
        param_name = f"osc{osc_num+1}_waveform"
        prefix = f"osc{osc_num+1}_wave_"
        button_tags = [f"{prefix}sine", f"{prefix}saw", f"{prefix}square", f"{prefix}tri"]
        button_values = [0.0, 0.33, 0.67, 1.0]
        self._set_button_parameter(param_name, waveform_value, button_tags, button_values)

    def _set_octave(self, osc_num, octave_value):
        param_name = f"osc{osc_num+1}_octave"
        prefix = f"osc{osc_num+1}_oct_"
        button_tags = [f"{prefix}m2", f"{prefix}m1", f"{prefix}0", f"{prefix}p1", f"{prefix}p2"]
        button_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._set_button_parameter(param_name, octave_value, button_tags, button_values)

    def _on_preset_selected(self, sender, app_data):
        preset_name = app_data
        self._load_preset(preset_name)

    def _on_piano_mouse_down(self, sender, app_data, user_data):
        note_offset, key_id = user_data
        self._synthesize_and_play(note_offset, key_id)

    def _on_piano_mouse_up(self, sender, app_data, user_data):
        note_offset, key_id = user_data
        self._stop_note(key_id)

    def _handle_key_press(self, key_char):
        if key_char in config.PIANO_KEY_MAPPING:
            note_offset = config.PIANO_KEY_MAPPING[key_char]
            key_id = f"kbd_{key_char}"
            # RealTimeSynth handles duplicate key_ids
            self._synthesize_and_play(note_offset, key_id)

    def _handle_key_release(self, key_char):
        key_id = f"kbd_{key_char}"
        self._stop_note(key_id)

    def _on_panic_button(self, sender, app_data):
        print("[GUI] PANIC - Stopping all voices")
        self.synth.stop_all()

    def _previous_preset(self):
        current_idx = self.available_presets.index(self.current_preset)
        new_idx = (current_idx - 1) % len(self.available_presets)
        new_preset = self.available_presets[new_idx]
        self._load_preset(new_preset)
        dpg.set_value("preset_combo", new_preset)

    def _next_preset(self):
        current_idx = self.available_presets.index(self.current_preset)
        new_idx = (current_idx + 1) % len(self.available_presets)
        new_preset = self.available_presets[new_idx]
        self._load_preset(new_preset)
        dpg.set_value("preset_combo", new_preset)

    def _randomize_parameters(self):
        for param_name in self.parameters.keys():
            if param_name != 'note':  # Keep note parameter unchanged
                random_value = np.random.random()
                self.parameters[param_name] = random_value

                # Update knob display
                if param_name in self.knob_tags:
                    dpg.set_value(self.knob_tags[param_name], random_value)

        print("[DPG] Randomized all parameters")

    def _on_match_audio_clicked(self):
        import threading
        from tkinter import Tk, filedialog

        # Use tkinter file dialog (DPG doesn't have native file picker)
        root = Tk()
        root.withdraw()  # Hide main window
        root.attributes('-topmost', True)  # Bring to front

        file_path = filedialog.askopenfilename(
            title="Select Target Audio File",
            filetypes=[("Audio files", "*.wav;*.mp3;*.flac;*.ogg;*.m4a"), ("All files", "*.*")]
        )

        root.destroy()

        if file_path:
            print(f"[DPG] Starting parameter matching for: {file_path}")

            # Store path and metric for continue optimization feature
            self.last_matched_audio_path = file_path
            self.last_matched_metric = self.optimization_method

            # Reset stop flag
            self.user_requested_stop = False

            # Show progress UI and stop button
            dpg.configure_item("match_progress", show=True)
            dpg.configure_item("match_status", show=True)
            dpg.configure_item("stop_early_button", show=True)
            dpg.set_value("match_status", "Initializing...")

            # Run matching in background thread
            thread = threading.Thread(
                target=self._run_audio_matching,
                args=(file_path, None),  # None = no initial params
                daemon=True
            )
            thread.start()

    def _on_continue_optimization_clicked(self):
        import threading
        from tkinter import Tk, filedialog

        # Open file picker to select target audio
        root = Tk()
        root.withdraw()  # Hide main window
        root.attributes('-topmost', True)  # Bring to front

        file_path = filedialog.askopenfilename(
            title="Select Target Audio File (Continue Optimization)",
            filetypes=[("Audio files", "*.wav;*.mp3;*.flac;*.ogg;*.m4a"), ("All files", "*.*")]
        )

        root.destroy()

        if not file_path:
            return  # User cancelled

        print(f"[DPG] Continuing optimization from current preset")
        print(f"  Target: {file_path}")
        print(f"  Previous method: {self.last_matched_metric}")
        print(f"  New method: {self.optimization_method}")

        # Store new path and method
        self.last_matched_audio_path = file_path
        previous_metric = self.last_matched_metric  # Keep track of what we started with
        self.last_matched_metric = self.optimization_method  # Update for next continue

        # Get current parameters as starting point
        initial_params = self._get_current_parameters()

        # Reset stop flag
        self.user_requested_stop = False

        # Show progress UI and stop button
        dpg.configure_item("match_progress", show=True)
        dpg.configure_item("match_status", show=True)
        dpg.configure_item("stop_early_button", show=True)
        dpg.set_value("match_status", "Initializing...")

        # Run matching in background thread with initial params
        thread = threading.Thread(
            target=self._run_audio_matching,
            args=(file_path, initial_params),
            daemon=True
        )
        thread.start()

    def _run_audio_matching(self, target_path, initial_params=None):
        try:
            from match_audio import match_audio_file

            def progress_callback(current_evals, total_evals, best_fitness):
                progress = current_evals / total_evals
                similarity = 1.0 - best_fitness
                dpg.set_value("match_progress", progress)
                dpg.set_value("match_status",
                             f"Running... {progress:.0%} ({current_evals}/{total_evals}) - "
                             f"Best: {similarity:.1%}")

            # User stop checker function
            def user_stop_check():
                return self.user_requested_stop

            # Device selection
            device = 'cpu' if self.force_cpu_matching else None  # None = auto-detect

            # Determine previous metric for naming (if continuing optimization)
            previous_metric = self.last_matched_metric if initial_params is not None else None

            # Read current widget values (don't rely on cached self.* values)
            max_evals = dpg.get_value("match_max_evals")
            target_similarity = dpg.get_value("match_target_similarity") / 100.0
            sample_duration = dpg.get_value("match_sample_duration")
            num_restarts = dpg.get_value("match_num_restarts")
            stagnation_gens = dpg.get_value("match_stagnation_gens")

            # Run matching with configuration (stops at whichever comes first)
            result = match_audio_file(
                target_path,
                progress_callback=progress_callback,
                max_evals=max_evals,
                target_similarity=target_similarity,
                sample_duration=sample_duration,
                num_restarts=num_restarts,
                user_stop_callback=user_stop_check,
                device=device,
                stagnation_gens=stagnation_gens,
                optimization_method=self.optimization_method,  # Changed from similarity_metric
                initial_params=initial_params,
                previous_metric=previous_metric
            )

            # Update UI with results - show all saved presets
            num_solutions = len(result.get('all_preset_names', [result['preset_name']]))
            dpg.set_value("match_status",
                         f"Complete! Saved {num_solutions} preset(s). "
                         f"Best similarity: {result['similarity']:.1%}")

            # Refresh preset list
            self.available_presets = self.preset_manager.list_presets()
            dpg.configure_item("preset_combo", items=self.available_presets)

            # Auto-load the best matched preset
            self._load_preset(result['preset_name'])
            dpg.set_value("preset_combo", result['preset_name'])

            print(f"[DPG] Matching complete! Loaded preset: {result['preset_name']}")

            # Hide progress and stop button after 3 seconds
            import time
            time.sleep(3)
            dpg.configure_item("match_progress", show=False)
            dpg.configure_item("match_status", show=False)
            dpg.configure_item("stop_early_button", show=False)

        except Exception as e:
            print(f"[DPG] Matching error: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value("match_status", f"Error: {str(e)}")
            dpg.configure_item("match_progress", show=False)
            dpg.configure_item("stop_early_button", show=False)

    def _on_stop_early_clicked(self):
        print("[DPG] User requested early stop")
        self.user_requested_stop = True
        dpg.set_value("match_status", "Stopping... will save best solution found")
        dpg.configure_item("stop_early_button", enabled=False)

    def _quit_app(self):
        dpg.stop_dearpygui()

    # ========== PRESET MANAGEMENT ==========

    def _load_preset(self, preset_name):
        try:
            preset_data = self.preset_manager.load_preset(preset_name)
            parameters = preset_data['parameters']

            # Update parameters and knobs
            for param_name, value in parameters.items():
                if param_name in self.parameters:
                    self.parameters[param_name] = value

                    # Update knob display
                    if param_name in self.knob_tags:
                        dpg.set_value(self.knob_tags[param_name], value)

            self.current_preset = preset_name

            # Update UI
            dpg.set_value("preset_display", f"Preset: {preset_name}")
            dpg.set_value("preset_name_display", preset_name)

            print(f"[DPG] Loaded preset: {preset_name}")
        except Exception as e:
            print(f"[DPG] Error loading preset {preset_name}: {e}")

    def _save_preset_dialog(self):
        # Create a hidden Tkinter root window
        root = tk.Tk()
        root.withdraw()

        # Generate default preset name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"custom_{timestamp}"

        # Ask user for preset name
        preset_name = simpledialog.askstring(
            "Save Preset",
            "Enter a name for this preset:",
            initialvalue=default_name,
            parent=root
        )

        # Clean up Tkinter
        root.destroy()

        # Check if user cancelled
        if not preset_name:
            print("[DPG] Preset save cancelled")
            return

        # Sanitize filename (remove invalid characters)
        preset_name = "".join(c for c in preset_name if c.isalnum() or c in ('-', '_'))

        if not preset_name:
            print("[DPG] Invalid preset name")
            dpg.set_value("match_status", "Invalid preset name")
            return

        try:
            # Build metadata
            metadata = {
                "name": preset_name,
                "description": "Custom user preset",
                "author": "User",
                "timestamp": timestamp
            }

            # Save using preset manager (pass parameters and metadata separately)
            self.preset_manager.save_preset(preset_name, dict(self.parameters), metadata)

            # Refresh preset list
            self.available_presets = self.preset_manager.list_presets()

            # Update combo box with new preset list
            dpg.configure_item("preset_combo", items=self.available_presets)

            # Select the newly saved preset
            dpg.set_value("preset_combo", preset_name)
            self.current_preset = preset_name

            # Update UI
            dpg.set_value("preset_display", f"Preset: {preset_name}")
            dpg.set_value("preset_name_display", preset_name)
            dpg.set_value("match_status", f"Saved preset: {preset_name}")

            print(f"[DPG] Saved preset: {preset_name}")

        except Exception as e:
            print(f"[DPG] Error saving preset: {e}")
            dpg.set_value("match_status", f"Error saving preset: {e}")

    # ========== AUDIO SYNTHESIS ==========

    def _get_current_parameters(self):
        params = np.zeros(38)  # 38 parameters now (added PWM, osc_sync, ring_mod, filter_drive)
        for i, param_name in enumerate(config.SYNTH_PARAM_ORDER):
            params[i] = self.parameters[param_name]
        return params

    def _audio_callback(self, outdata, frames, time, status):
        if status:
            print(f"[Audio] {status}")

        # Generate audio with voice mixing
        buffer = self.synth.process_audio(frames)

        outdata[:] = buffer.reshape(-1, 1)

    def _synthesize_and_play(self, note_offset, key_id=None):
        try:
            params = self._get_current_parameters()
            note_param = params[-1]
            synth_params = params[:-1]

            base_midi = config.note_param_to_midi(note_param)
            final_midi = int(np.clip(base_midi + note_offset, config.NOTE_MIN, config.NOTE_MAX))

            param_dict = {}
            for i, param_name in enumerate(config.SYNTH_PARAM_ORDER[:-1]):
                param_dict[param_name] = synth_params[i]

            self.synth.set_parameters(param_dict)

            # Trigger note on with voice allocation
            self.synth.note_on(final_midi, velocity=0.8, key_id=key_id)

            note_name = config.midi_to_note_name(final_midi)
            print(f"[DPG] Note on: {note_name} (MIDI {final_midi}), Active: {self.synth.get_num_active_voices()}/{self.NUM_VOICES}")

        except Exception as e:
            print(f"[DPG] Audio error: {e}")
            import traceback
            traceback.print_exc()

    def _stop_note(self, key_id):
        self.synth.note_off(key_id)
        print(f"[DPG] Note off: {key_id}, Active: {self.synth.get_num_active_voices()}/{self.NUM_VOICES}")


    # ========== MAIN LOOP ==========

    def run(self):
        # Setup DearPyGui
        dpg.setup_dearpygui()
        dpg.show_viewport()

        print("[DPG] Starting main loop...")

        # Load initial preset
        self._load_preset(self.current_preset)

        # Start audio stream with callback
        print("[DPG] Starting audio stream...")
        self.audio_stream = sd.OutputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=self.buffer_size,
            callback=self._audio_callback
        )
        self.audio_stream.start()
        print(f"[DPG] Audio running at {config.SAMPLE_RATE} Hz, buffer size {self.buffer_size}")

        # Main render loop
        dpg.start_dearpygui()

        # Cleanup
        print("[DPG] Shutting down...")
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        dpg.destroy_context()
        print("[DPG] Goodbye!")


def main():
    gui = SynthesizerGUI()
    gui.create_gui()
    gui.run()


if __name__ == "__main__":
    main()
