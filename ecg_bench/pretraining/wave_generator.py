"""
Wave generation utilities for pretraining.

Generates various waveforms at 250 Hz sampling rate:
- Sine wave
- Triangle wave
- Sawtooth wave
- Square wave
- Pulse wave
- Cosine wave
- Noise
- DC (constant)
- Ramp (linear)
- Exponential decay
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class WaveType(Enum):
    SINE = "sine"
    COSINE = "cosine"
    TRIANGLE = "triangle"
    SAWTOOTH = "sawtooth"
    SQUARE = "square"
    PULSE = "pulse"
    NOISE = "noise"
    DC = "dc"
    RAMP = "ramp"
    EXPONENTIAL_DECAY = "exponential_decay"


@dataclass
class WaveParams:
    """Parameters for wave generation."""
    wave_type: WaveType
    amplitude: float  # in mV (will be clamped to [-3, 3])
    frequency: float  # in Hz
    phase: float = 0.0  # in radians
    offset: float = 0.0  # DC offset in mV
    duty_cycle: float = 0.5  # for pulse/square waves
    decay_rate: float = 1.0  # for exponential decay

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wave_type": self.wave_type.value,
            "amplitude": self.amplitude,
            "frequency": self.frequency,
            "phase": self.phase,
            "offset": self.offset,
            "duty_cycle": self.duty_cycle,
            "decay_rate": self.decay_rate,
        }


SAMPLING_RATE = 250  # Hz


def generate_time_array(duration_seconds: float) -> np.ndarray:
    """Generate time array for given duration at 250 Hz."""
    num_samples = int(duration_seconds * SAMPLING_RATE)
    return np.linspace(0, duration_seconds, num_samples, endpoint=False)


def generate_sine(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate sine wave."""
    return params.amplitude * np.sin(2 * np.pi * params.frequency * t + params.phase) + params.offset


def generate_cosine(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate cosine wave."""
    return params.amplitude * np.cos(2 * np.pi * params.frequency * t + params.phase) + params.offset


def generate_triangle(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate triangle wave."""
    period = 1.0 / params.frequency if params.frequency > 0 else 1.0
    phase_offset = params.phase / (2 * np.pi) * period
    t_shifted = (t + phase_offset) % period
    # Triangle wave: goes from -1 to 1 in first half, 1 to -1 in second half
    normalized = t_shifted / period
    triangle = 4 * np.abs(normalized - 0.5) - 1
    return params.amplitude * triangle + params.offset


def generate_sawtooth(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate sawtooth wave (ramp up, instant drop)."""
    period = 1.0 / params.frequency if params.frequency > 0 else 1.0
    phase_offset = params.phase / (2 * np.pi) * period
    t_shifted = (t + phase_offset) % period
    # Sawtooth: linear ramp from -1 to 1
    sawtooth = 2 * (t_shifted / period) - 1
    return params.amplitude * sawtooth + params.offset


def generate_square(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate square wave."""
    period = 1.0 / params.frequency if params.frequency > 0 else 1.0
    phase_offset = params.phase / (2 * np.pi) * period
    t_shifted = (t + phase_offset) % period
    # Square wave with duty cycle
    square = np.where(t_shifted < period * params.duty_cycle, 1.0, -1.0)
    return params.amplitude * square + params.offset


def generate_pulse(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate pulse wave (short pulses with configurable duty cycle)."""
    period = 1.0 / params.frequency if params.frequency > 0 else 1.0
    phase_offset = params.phase / (2 * np.pi) * period
    t_shifted = (t + phase_offset) % period
    # Pulse: high for duty_cycle fraction, zero otherwise
    pulse = np.where(t_shifted < period * params.duty_cycle, 1.0, 0.0)
    return params.amplitude * pulse + params.offset


def generate_noise(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate random noise."""
    noise = np.random.uniform(-1, 1, len(t))
    return params.amplitude * noise + params.offset


def generate_dc(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate DC (constant) signal."""
    return np.full_like(t, params.amplitude + params.offset)


def generate_ramp(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate linear ramp."""
    duration = t[-1] - t[0] if len(t) > 1 else 1.0
    ramp = (t - t[0]) / duration * 2 - 1  # -1 to 1
    return params.amplitude * ramp + params.offset


def generate_exponential_decay(t: np.ndarray, params: WaveParams) -> np.ndarray:
    """Generate exponential decay."""
    decay = np.exp(-params.decay_rate * t)
    return params.amplitude * decay + params.offset


WAVE_GENERATORS = {
    WaveType.SINE: generate_sine,
    WaveType.COSINE: generate_cosine,
    WaveType.TRIANGLE: generate_triangle,
    WaveType.SAWTOOTH: generate_sawtooth,
    WaveType.SQUARE: generate_square,
    WaveType.PULSE: generate_pulse,
    WaveType.NOISE: generate_noise,
    WaveType.DC: generate_dc,
    WaveType.RAMP: generate_ramp,
    WaveType.EXPONENTIAL_DECAY: generate_exponential_decay,
}


def generate_wave(duration_seconds: float, params: WaveParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a waveform with given parameters.

    Args:
        duration_seconds: Duration of the wave in seconds
        params: Wave parameters

    Returns:
        Tuple of (time_array, wave_values)
    """
    t = generate_time_array(duration_seconds)
    generator = WAVE_GENERATORS[params.wave_type]
    wave = generator(t, params)
    return t, wave


def clamp_wave(wave: np.ndarray, min_val: float = -3.0, max_val: float = 3.0) -> np.ndarray:
    """Clamp wave values to valid range."""
    return np.clip(wave, min_val, max_val)


def random_wave_params(
    wave_type: Optional[WaveType] = None,
    amplitude_range: Tuple[float, float] = (0.5, 2.5),
    frequency_range: Tuple[float, float] = (0.5, 10.0),
    allow_offset: bool = True,
) -> WaveParams:
    """Generate random wave parameters."""
    if wave_type is None:
        wave_type = np.random.choice(list(WaveType))

    amplitude = np.random.uniform(*amplitude_range)
    frequency = np.random.uniform(*frequency_range)
    phase = np.random.uniform(0, 2 * np.pi)
    offset = np.random.uniform(-0.5, 0.5) if allow_offset else 0.0
    duty_cycle = np.random.uniform(0.1, 0.9)
    decay_rate = np.random.uniform(0.5, 3.0)

    return WaveParams(
        wave_type=wave_type,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase,
        offset=offset,
        duty_cycle=duty_cycle,
        decay_rate=decay_rate,
    )


def wave_type_description(wave_type: WaveType) -> str:
    """Get human-readable description of wave type."""
    descriptions = {
        WaveType.SINE: "sine wave",
        WaveType.COSINE: "cosine wave",
        WaveType.TRIANGLE: "triangle wave",
        WaveType.SAWTOOTH: "sawtooth wave",
        WaveType.SQUARE: "square wave",
        WaveType.PULSE: "pulse wave",
        WaveType.NOISE: "random noise",
        WaveType.DC: "constant (DC) signal",
        WaveType.RAMP: "linear ramp",
        WaveType.EXPONENTIAL_DECAY: "exponential decay",
    }
    return descriptions[wave_type]


def get_wave_properties_for_type(wave_type: WaveType) -> list[str]:
    """Get list of relevant properties for a wave type."""
    common = ["amplitude"]

    if wave_type in [WaveType.SINE, WaveType.COSINE, WaveType.TRIANGLE,
                     WaveType.SAWTOOTH, WaveType.SQUARE, WaveType.PULSE]:
        return common + ["frequency", "phase"]
    elif wave_type == WaveType.DC:
        return ["amplitude"]  # Just the constant value
    elif wave_type == WaveType.RAMP:
        return common
    elif wave_type == WaveType.EXPONENTIAL_DECAY:
        return common + ["decay_rate"]
    elif wave_type == WaveType.NOISE:
        return common
    else:
        return common
