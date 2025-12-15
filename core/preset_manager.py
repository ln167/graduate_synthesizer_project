import json
import os
import numpy as np
from pathlib import Path

# Import config from parent directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.config import SYNTH_PARAM_ORDER, SYNTH_PARAM_DEFAULTS, PRESETS_DIR


def get_base_path():
    # PyInstaller sets sys._MEIPASS when running from bundle
    if getattr(sys, '_MEIPASS', None):
        return Path(sys._MEIPASS)
    return Path(__file__).parent.parent


class PresetManager:

    def __init__(self, presets_dir=None):
        if presets_dir is None:
            base_dir = get_base_path()
            self.presets_dir = base_dir / PRESETS_DIR
        else:
            self.presets_dir = Path(presets_dir)

        # Create presets directory if it doesn't exist
        self.presets_dir.mkdir(parents=True, exist_ok=True)

        print(f"[PresetManager] Initialized with presets directory: {self.presets_dir}")

    def list_presets(self):
        preset_files = list(self.presets_dir.glob("*.json"))
        preset_names = [f.stem for f in preset_files]
        return sorted(preset_names)

    def load_preset(self, preset_name):
        # Add .json extension if not present
        if not preset_name.endswith('.json'):
            preset_name = preset_name + '.json'

        preset_path = self.presets_dir / preset_name

        if not preset_path.exists():
            raise FileNotFoundError(f"Preset not found: {preset_path}")

        # Load JSON
        with open(preset_path, 'r') as f:
            data = json.load(f)

        # Validate format
        if 'parameters' not in data:
            raise ValueError(f"Invalid preset format: missing 'parameters' key")

        # Ensure all 26 parameters are present (fill with defaults if missing)
        parameters = {}
        for param_name in SYNTH_PARAM_ORDER:
            if param_name in data['parameters']:
                parameters[param_name] = float(data['parameters'][param_name])
            else:
                # Use default value if parameter is missing
                parameters[param_name] = SYNTH_PARAM_DEFAULTS[param_name]
                print(f"[PresetManager] Warning: Parameter '{param_name}' missing in preset, using default")

        # Extract metadata
        metadata = data.get('metadata', {})

        print(f"[PresetManager] Loaded preset: {preset_name}")
        return {
            'parameters': parameters,
            'metadata': metadata
        }

    def save_preset(self, preset_name, parameters, metadata=None):
        # Add .json extension if not present
        if not preset_name.endswith('.json'):
            preset_name = preset_name + '.json'

        preset_path = self.presets_dir / preset_name

        # Convert numpy array to dict if needed
        if isinstance(parameters, np.ndarray):
            parameters = self._array_to_dict(parameters)

        # Validate parameters (skip 'note' as it's runtime-only)
        for param_name in SYNTH_PARAM_ORDER:
            if param_name == 'note':
                continue  # Note is runtime parameter, not saved in presets
            if param_name not in parameters:
                raise ValueError(f"Missing parameter: {param_name}")

        # Create JSON structure
        data = {
            'parameters': parameters,
            'metadata': metadata or {}
        }

        # Save to file
        with open(preset_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[PresetManager] Saved preset: {preset_path}")
        return preset_path

    def parameters_to_array(self, parameters):
        # Filter out only 'note' from SYNTH_PARAM_ORDER (keep note_hold_time)
        synth_params_only = [p for p in SYNTH_PARAM_ORDER if p != 'note']
        array = np.zeros(len(synth_params_only))
        for i, param_name in enumerate(synth_params_only):
            if param_name in parameters:
                array[i] = parameters[param_name]
            else:
                array[i] = SYNTH_PARAM_DEFAULTS[param_name]
        return array

    def array_to_parameters(self, array):
        return self._array_to_dict(array)

    def _array_to_dict(self, array):
        # Filter out only 'note' from SYNTH_PARAM_ORDER (keep note_hold_time)
        synth_params_only = [p for p in SYNTH_PARAM_ORDER if p != 'note']

        if len(array) != len(synth_params_only):
            raise ValueError(f"Expected {len(synth_params_only)} parameters, got {len(array)}")

        parameters = {}
        for i, param_name in enumerate(synth_params_only):
            parameters[param_name] = float(array[i])

        return parameters

    def get_default_parameters(self):
        return SYNTH_PARAM_DEFAULTS.copy()

    def create_init_preset(self):
        metadata = {
            'name': 'Init',
            'description': 'Default initialization preset with neutral settings',
            'author': 'System'
        }

        self.save_preset('init', SYNTH_PARAM_DEFAULTS, metadata)
        print("[PresetManager] Created 'init' preset")


# Standalone functions for quick usage

def load_preset(preset_name, presets_dir=None):
    manager = PresetManager(presets_dir)
    return manager.load_preset(preset_name)


def save_preset(preset_name, parameters, metadata=None, presets_dir=None):
    manager = PresetManager(presets_dir)
    return manager.save_preset(preset_name, parameters, metadata)


if __name__ == "__main__":
    # Test preset manager
    print("=" * 80)
    print("PRESET MANAGER TEST")
    print("=" * 80)

    manager = PresetManager()

    # Create init preset
    manager.create_init_preset()

    # List presets
    presets = manager.list_presets()
    print(f"\nAvailable presets: {presets}")

    # Load init preset
    if 'init' in presets:
        preset_data = manager.load_preset('init')
        print(f"\nInit preset parameters: {len(preset_data['parameters'])} params")
        print(f"Metadata: {preset_data['metadata']}")

    print("\n[OK] Preset manager test complete!")
