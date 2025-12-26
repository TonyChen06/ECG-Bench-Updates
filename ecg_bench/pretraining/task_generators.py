"""
Task generators for pretraining.

Generates various tasks to help the model understand ECG tokens:
- Task 0: Token comparison (magnitude, sequence, halfway)
- Task 1: Values to tokens (given values, output tokens)
- Task 2: Tokens to values (given tokens, output values)
- Task 3: Wave classification and property extraction (multi-turn)
- Task 4: Wave transformation (frequency/amplitude changes)
- Task 5: Wave reconstruction from specification (chunked)
- Task 6: ECG reconstruction from real MIMIC data
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import random

from ecg_bench.pretraining.wave_generator import (
    WaveType, WaveParams, generate_wave, clamp_wave,
    random_wave_params, wave_type_description, get_wave_properties_for_type,
    SAMPLING_RATE,
)
from ecg_bench.pretraining.tokenization import (
    value_to_token, token_to_value, values_to_tokens, tokens_to_values,
    format_value, format_values_list, tokens_to_string, string_to_tokens,
    compare_magnitudes, get_halfway_token, get_next_token, value_to_bin,
)
from ecg_bench.configs.constants import ECG_RAW_LEAD_INDICES, ECG_RAW_LEADS


# Global ECG loader (initialized lazily)
_ecg_loader = None


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "human" or "assistant"
    content: str


@dataclass
class Task:
    """A pretraining task consisting of conversation turns."""
    task_type: str
    turns: List[ConversationTurn]
    metadata: Dict[str, Any]


# =============================================================================
# Task 0: Token Comparison Questions
# =============================================================================

def generate_magnitude_comparison_task() -> Task:
    """
    Generate a task asking which token represents a greater magnitude.
    """
    # Generate two random values with different magnitudes
    val1 = np.random.uniform(-3, 3)
    val2 = np.random.uniform(-3, 3)

    # Ensure they have different magnitudes
    while abs(abs(val1) - abs(val2)) < 0.1:
        val2 = np.random.uniform(-3, 3)

    token1 = value_to_token(val1)
    token2 = value_to_token(val2)

    # Determine answer
    if abs(val1) > abs(val2):
        answer = token1
        explanation = f"{token1} represents {format_value(val1)} mV (magnitude {format_value(abs(val1))}), while {token2} represents {format_value(val2)} mV (magnitude {format_value(abs(val2))}). {format_value(abs(val1))} > {format_value(abs(val2))}, so {token1} has greater magnitude."
    else:
        answer = token2
        explanation = f"{token1} represents {format_value(val1)} mV (magnitude {format_value(abs(val1))}), while {token2} represents {format_value(val2)} mV (magnitude {format_value(abs(val2))}). {format_value(abs(val2))} > {format_value(abs(val1))}, so {token2} has greater magnitude."

    turns = [
        ConversationTurn("human", f"Which of these two timeseries tokens represents a greater magnitude: {token1} or {token2}?"),
        ConversationTurn("assistant", f"{answer}"),
    ]

    return Task(
        task_type="magnitude_comparison",
        turns=turns,
        metadata={"val1": val1, "val2": val2, "token1": token1, "token2": token2, "answer": answer}
    )


def generate_sequence_next_token_task() -> Task:
    """
    Generate a task asking which token comes directly after another.
    """
    # Generate a sequence of values (like from a wave)
    duration = 0.1  # 25 samples
    params = random_wave_params(frequency_range=(1.0, 5.0))
    _, wave = generate_wave(duration, params)
    wave = clamp_wave(wave)

    # Pick a random position (not the last)
    pos = np.random.randint(0, len(wave) - 1)
    current_token = value_to_token(wave[pos])
    next_token = value_to_token(wave[pos + 1])

    # Show a few tokens for context
    context_start = max(0, pos - 3)
    context_tokens = values_to_tokens(wave[context_start:pos + 1])
    context_str = tokens_to_string(context_tokens)

    turns = [
        ConversationTurn("human", f"Given this sequence of timeseries tokens: {context_str}\nWhat token comes directly after {current_token} in this sequence?"),
        ConversationTurn("assistant", f"{next_token}"),
    ]

    return Task(
        task_type="sequence_next",
        turns=turns,
        metadata={"context": context_str, "current": current_token, "next": next_token}
    )


def generate_halfway_token_task() -> Task:
    """
    Generate a task asking for the token halfway between two others.
    """
    val1 = np.random.uniform(-2.5, 2.5)
    val2 = np.random.uniform(-2.5, 2.5)

    # Ensure they're different enough
    while abs(val1 - val2) < 0.5:
        val2 = np.random.uniform(-2.5, 2.5)

    token1 = value_to_token(val1)
    token2 = value_to_token(val2)

    halfway_val = (val1 + val2) / 2
    halfway_token = value_to_token(halfway_val)

    turns = [
        ConversationTurn("human", f"What timeseries token is halfway between {token1} and {token2}?"),
        ConversationTurn("assistant", f"{halfway_token}"),
    ]

    return Task(
        task_type="halfway_token",
        turns=turns,
        metadata={"token1": token1, "token2": token2, "halfway": halfway_token, "val1": val1, "val2": val2}
    )


def generate_token_value_comparison_task() -> Task:
    """
    Generate a task asking which token represents a higher/lower value.
    """
    val1 = np.random.uniform(-3, 3)
    val2 = np.random.uniform(-3, 3)

    while abs(val1 - val2) < 0.2:
        val2 = np.random.uniform(-3, 3)

    token1 = value_to_token(val1)
    token2 = value_to_token(val2)

    # Randomly ask for higher or lower
    ask_higher = random.choice([True, False])

    if ask_higher:
        question = f"Which timeseries token represents a higher value: {token1} or {token2}?"
        answer = token1 if val1 > val2 else token2
    else:
        question = f"Which timeseries token represents a lower value: {token1} or {token2}?"
        answer = token1 if val1 < val2 else token2

    turns = [
        ConversationTurn("human", question),
        ConversationTurn("assistant", f"{answer}"),
    ]

    return Task(
        task_type="value_comparison",
        turns=turns,
        metadata={"token1": token1, "token2": token2, "val1": val1, "val2": val2, "ask_higher": ask_higher}
    )


def generate_task_0() -> Task:
    """Generate a random Task 0 (token comparison)."""
    generators = [
        generate_magnitude_comparison_task,
        generate_sequence_next_token_task,
        generate_halfway_token_task,
        generate_token_value_comparison_task,
    ]
    return random.choice(generators)()


# =============================================================================
# Task 1: Values to Tokens
# =============================================================================

def generate_values_to_tokens_task(num_values: int = 50) -> Task:
    """
    Generate a task where given values, the model outputs tokens.
    """
    # Generate random values
    values = np.random.uniform(-2.5, 2.5, num_values)
    values = np.round(values, 2)  # Round for cleaner display

    tokens = values_to_tokens(values)

    values_str = format_values_list(values)
    tokens_str = tokens_to_string(tokens)

    turns = [
        ConversationTurn("human", f"Convert these values (in mV) to their corresponding timeseries tokens:\n{values_str}"),
        ConversationTurn("assistant", f"{tokens_str}"),
    ]

    return Task(
        task_type="values_to_tokens",
        turns=turns,
        metadata={"values": values.tolist(), "tokens": tokens, "num_values": num_values}
    )


def generate_task_1(num_values: int = 50) -> Task:
    """Generate Task 1 (values to tokens)."""
    return generate_values_to_tokens_task(num_values)


# =============================================================================
# Task 2: Tokens to Values
# =============================================================================

def generate_tokens_to_values_task(num_tokens: int = 20) -> Task:
    """
    Generate a task where given tokens, the model outputs values.
    Fewer tokens than Task 1 since this is harder.
    """
    # Generate random bin indices
    bin_indices = np.random.randint(0, 600, num_tokens)
    tokens = [f"ecg_{idx}" for idx in bin_indices]
    values = tokens_to_values(tokens)

    tokens_str = tokens_to_string(tokens)
    values_str = format_values_list(values)

    turns = [
        ConversationTurn("human", f"What values (in mV) do these timeseries tokens represent?\n{tokens_str}"),
        ConversationTurn("assistant", f"{values_str}"),
    ]

    return Task(
        task_type="tokens_to_values",
        turns=turns,
        metadata={"tokens": tokens, "values": values.tolist(), "num_tokens": num_tokens}
    )


def generate_task_2(num_tokens: int = 20) -> Task:
    """Generate Task 2 (tokens to values)."""
    return generate_tokens_to_values_task(num_tokens)


# =============================================================================
# Task 3: Wave Classification + Properties (Multi-turn)
# =============================================================================

def generate_wave_classification_task(duration: float = 2.0) -> Task:
    """
    Generate a multi-turn conversation about wave classification and properties.
    """
    # Generate a random wave
    wave_type = random.choice([
        WaveType.SINE, WaveType.COSINE, WaveType.TRIANGLE,
        WaveType.SAWTOOTH, WaveType.SQUARE, WaveType.PULSE,
    ])

    # Cap amplitude at 3.0 to stay within [-3, 3] mV range without clamping
    params = random_wave_params(
        wave_type=wave_type,
        amplitude_range=(0.8, 3.0),
        frequency_range=(1.0, 5.0),
        allow_offset=False,  # Keep it simpler for classification
    )

    _, wave = generate_wave(duration, params)
    tokens = values_to_tokens(wave)
    tokens_str = tokens_to_string(tokens)

    wave_name = wave_type_description(wave_type)
    properties = get_wave_properties_for_type(wave_type)

    turns = []

    # Turn 1: Classification
    turns.append(ConversationTurn("human", f"Analyze this timeseries signal and identify what type of wave it is:\n{tokens_str}"))
    turns.append(ConversationTurn("assistant", f"This is a {wave_name}."))

    # Turn 2: Amplitude
    if "amplitude" in properties:
        turns.append(ConversationTurn("human", "What is the amplitude of this wave?"))
        turns.append(ConversationTurn("assistant", f"The amplitude is approximately {format_value(params.amplitude)} mV."))

    # Turn 3: Frequency
    if "frequency" in properties:
        turns.append(ConversationTurn("human", "What is the frequency of this wave?"))
        turns.append(ConversationTurn("assistant", f"The frequency is approximately {format_value(params.frequency)} Hz."))

    # Turn 4: Additional property based on wave type
    if wave_type in [WaveType.SQUARE, WaveType.PULSE] and "duty_cycle" in dir(params):
        turns.append(ConversationTurn("human", "What is the duty cycle of this wave?"))
        turns.append(ConversationTurn("assistant", f"The duty cycle is approximately {format_value(params.duty_cycle * 100)}%."))

    return Task(
        task_type="wave_classification",
        turns=turns,
        metadata={
            "wave_type": wave_type.value,
            "params": params.to_dict(),
            "num_samples": len(wave),
        }
    )


def generate_task_3(duration: float = 2.0) -> Task:
    """Generate Task 3 (wave classification + properties)."""
    return generate_wave_classification_task(duration)


# =============================================================================
# Task 4: Wave Transformation
# =============================================================================

def generate_frequency_transformation_task(duration: float = 1.0) -> Task:
    """
    Generate a task: given a wave, show what it looks like with different frequency.
    """
    # Generate original wave with amplitude capped at 3.0 to avoid clamping
    wave_type = random.choice([WaveType.SINE, WaveType.TRIANGLE, WaveType.SAWTOOTH])
    original_freq = np.random.uniform(1.0, 3.0)
    amplitude = np.random.uniform(1.0, 3.0)

    params = WaveParams(
        wave_type=wave_type,
        amplitude=amplitude,
        frequency=original_freq,
        phase=0,
        offset=0,
    )

    _, original_wave = generate_wave(duration, params)
    original_tokens = values_to_tokens(original_wave)

    # Transform frequency (amplitude stays the same, so no clamping needed)
    freq_multiplier = random.choice([2.0, 0.5, 3.0])
    new_freq = original_freq * freq_multiplier

    new_params = WaveParams(
        wave_type=wave_type,
        amplitude=amplitude,
        frequency=new_freq,
        phase=0,
        offset=0,
    )

    _, new_wave = generate_wave(duration, new_params)
    new_tokens = values_to_tokens(new_wave)

    original_str = tokens_to_string(original_tokens)
    new_str = tokens_to_string(new_tokens)

    turns = [
        ConversationTurn("human", f"Here is a {wave_type_description(wave_type)} signal:\n{original_str}\n\nShow me what this wave would look like with {freq_multiplier}x the frequency."),
        ConversationTurn("assistant", f"{new_str}"),
    ]

    return Task(
        task_type="frequency_transformation",
        turns=turns,
        metadata={
            "wave_type": wave_type.value,
            "original_freq": original_freq,
            "new_freq": new_freq,
            "multiplier": freq_multiplier,
            "amplitude": amplitude,
        }
    )


def generate_amplitude_transformation_task(duration: float = 1.0) -> Task:
    """
    Generate a task: given a wave, show what it looks like with different amplitude.
    Ensures both original and transformed waves stay within [-3, 3] mV without clamping.
    """
    wave_type = random.choice([WaveType.SINE, WaveType.TRIANGLE, WaveType.SQUARE])
    frequency = np.random.uniform(1.0, 4.0)

    # Pick multiplier first, then constrain original amplitude so result stays <= 3.0
    amp_multiplier = random.choice([2.0, 0.5, 1.5])

    if amp_multiplier > 1.0:
        # If multiplying up, original must be small enough that result <= 3.0
        max_original = 3.0 / amp_multiplier
        original_amp = np.random.uniform(0.5, max_original)
    else:
        # If multiplying down, original can be up to 3.0
        original_amp = np.random.uniform(1.0, 3.0)

    new_amp = original_amp * amp_multiplier

    params = WaveParams(
        wave_type=wave_type,
        amplitude=original_amp,
        frequency=frequency,
        phase=0,
        offset=0,
    )

    _, original_wave = generate_wave(duration, params)
    original_tokens = values_to_tokens(original_wave)

    new_params = WaveParams(
        wave_type=wave_type,
        amplitude=new_amp,
        frequency=frequency,
        phase=0,
        offset=0,
    )

    _, new_wave = generate_wave(duration, new_params)
    new_tokens = values_to_tokens(new_wave)

    original_str = tokens_to_string(original_tokens)
    new_str = tokens_to_string(new_tokens)

    turns = [
        ConversationTurn("human", f"Here is a signal:\n{original_str}\n\nShow me what this wave would look like with {amp_multiplier}x the amplitude."),
        ConversationTurn("assistant", f"{new_str}"),
    ]

    return Task(
        task_type="amplitude_transformation",
        turns=turns,
        metadata={
            "wave_type": wave_type.value,
            "original_amp": original_amp,
            "new_amp": new_amp,
            "multiplier": amp_multiplier,
            "frequency": frequency,
        }
    )


def generate_task_4(duration: float = 1.0) -> Task:
    """Generate Task 4 (wave transformation)."""
    return random.choice([
        generate_frequency_transformation_task,
        generate_amplitude_transformation_task,
    ])(duration)


# =============================================================================
# Task 5: Wave Reconstruction (Chunked)
# =============================================================================

def generate_wave_reconstruction_task(
    duration: float = 2.0,
    chunk_size: int = 100,
) -> Task:
    """
    Generate a task where the model reconstructs a wave from specification.
    The wave is generated in chunks to prevent cheating.
    """
    wave_type = random.choice([
        WaveType.SINE, WaveType.COSINE, WaveType.TRIANGLE,
        WaveType.SAWTOOTH, WaveType.SQUARE,
    ])

    # Cap amplitude at 3.0 to stay within [-3, 3] mV range without clamping
    amplitude = round(np.random.uniform(0.8, 3.0), 2)
    frequency = round(np.random.uniform(1.0, 4.0), 2)

    params = WaveParams(
        wave_type=wave_type,
        amplitude=amplitude,
        frequency=frequency,
        phase=0,
        offset=0,
    )

    _, wave = generate_wave(duration, params)
    total_samples = len(wave)

    turns = []

    # Initial specification
    spec = f"{wave_type_description(wave_type)} with amplitude {amplitude} mV and frequency {frequency} Hz"
    turns.append(ConversationTurn("human", f"Generate a {spec} sampled at 250 Hz for {duration} seconds. Output the timeseries tokens in chunks. Start with tokens 1-{chunk_size}."))

    # Generate chunks
    num_chunks = (total_samples + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk_tokens = values_to_tokens(wave[start_idx:end_idx])
        chunk_str = tokens_to_string(chunk_tokens)

        turns.append(ConversationTurn("assistant", f"{chunk_str}"))

        if end_idx < total_samples:
            next_start = end_idx + 1
            next_end = min(end_idx + chunk_size, total_samples)
            turns.append(ConversationTurn("human", f"Continue with tokens {next_start}-{next_end}."))

    return Task(
        task_type="wave_reconstruction",
        turns=turns,
        metadata={
            "wave_type": wave_type.value,
            "amplitude": amplitude,
            "frequency": frequency,
            "duration": duration,
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "num_chunks": num_chunks,
        }
    )


def generate_task_5(duration: float = 2.0, chunk_size: int = 100) -> Task:
    """Generate Task 5 (wave reconstruction from specification)."""
    return generate_wave_reconstruction_task(duration, chunk_size)


# =============================================================================
# Task 6: ECG Reconstruction from Real MIMIC Data
# =============================================================================

def init_ecg_loader(dataset_name: str = "ecg-qa-mimic-iv-ecg-250-1250", fold: str = "1"):
    """
    Initialize the global ECG loader for Task 6.

    Must be called before generating Task 6 examples.
    """
    global _ecg_loader
    from ecg_bench.pretraining.ecg_loader import PretrainECGLoader
    _ecg_loader = PretrainECGLoader(dataset_name=dataset_name, fold=fold)
    return _ecg_loader


def get_ecg_loader():
    """Get the global ECG loader, raising an error if not initialized."""
    global _ecg_loader
    if _ecg_loader is None:
        raise RuntimeError(
            "ECG loader not initialized. Call init_ecg_loader() before generating Task 6 examples."
        )
    return _ecg_loader


def extract_lead_signal(ecg_signal: np.ndarray, lead_idx: int) -> np.ndarray:
    """
    Extract a single lead from the ECG signal.

    Args:
        ecg_signal: ECG signal array of shape (num_leads, num_samples) or (num_samples, num_leads)
        lead_idx: Index of the lead to extract (0-11 for 12-lead ECG)

    Returns:
        1D array of the lead signal
    """
    # Handle different shapes
    if ecg_signal.shape[0] == 12:  # leads first: (12, num_samples)
        return ecg_signal[lead_idx, :]
    elif ecg_signal.shape[1] == 12:  # samples first: (num_samples, 12)
        return ecg_signal[:, lead_idx]
    else:
        # Try to infer the shape
        if ecg_signal.shape[0] < ecg_signal.shape[1]:
            return ecg_signal[lead_idx, :]
        else:
            return ecg_signal[:, lead_idx]


def generate_real_ecg_reconstruction_task(
    chunk_size: int = 100,
    max_samples: int = 500,  # Limit samples to keep task reasonable (500 samples = 2 seconds at 250 Hz)
) -> Task:
    """
    Generate a task where the model outputs ECG tokens from real MIMIC data.

    Uses one of the 4 leads (II, aVR, V1, V4) from a real ECG recording.
    Includes the diagnostic report in the prompt.
    """
    loader = get_ecg_loader()

    # Try to get a valid ECG with a non-empty report
    max_attempts = 20
    for _ in range(max_attempts):
        ecg_signal, metadata = loader.get_random_ecg()
        if ecg_signal is not None and metadata.get("report", "").strip():
            break
    else:
        raise RuntimeError("Failed to load ECG with diagnosis after multiple attempts")

    # Get the diagnosis report
    report = metadata.get("report", "").strip()

    # Select a random lead from our 4 leads
    lead_idx_in_list = random.randint(0, len(ECG_RAW_LEAD_INDICES) - 1)
    lead_idx = ECG_RAW_LEAD_INDICES[lead_idx_in_list]
    lead_name = ECG_RAW_LEADS[lead_idx_in_list]

    # Extract the lead signal
    lead_signal = extract_lead_signal(ecg_signal, lead_idx)

    # Limit the number of samples
    if len(lead_signal) > max_samples:
        # Take a random segment
        start = random.randint(0, len(lead_signal) - max_samples)
        lead_signal = lead_signal[start:start + max_samples]

    # Clamp to [-3, 3] mV range
    lead_signal = clamp_wave(lead_signal)
    total_samples = len(lead_signal)
    duration = total_samples / SAMPLING_RATE

    turns = []

    # Initial prompt - includes diagnosis and asks for ECG tokens
    prompt = f"Generate the {lead_name} lead ECG signal for a patient with the following diagnosis: {report}\n\nOutput the timeseries tokens sampled at 250 Hz for {format_value(duration)} seconds. Output in chunks. Start with tokens 1-{min(chunk_size, total_samples)}."

    turns.append(ConversationTurn("human", prompt))

    # Generate chunks
    num_chunks = (total_samples + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_samples)
        chunk_tokens = values_to_tokens(lead_signal[start_idx:end_idx])
        chunk_str = tokens_to_string(chunk_tokens)

        turns.append(ConversationTurn("assistant", f"{chunk_str}"))

        if end_idx < total_samples:
            next_start = end_idx + 1
            next_end = min(end_idx + chunk_size, total_samples)
            turns.append(ConversationTurn("human", f"Continue with tokens {next_start}-{next_end}."))

    return Task(
        task_type="ecg_reconstruction",
        turns=turns,
        metadata={
            "source": "mimic-iv-ecg",
            "ecg_path": metadata.get("ecg_path", ""),
            "report": report,
            "lead": lead_name,
            "lead_idx": lead_idx,
            "duration": duration,
            "chunk_size": chunk_size,
            "total_samples": total_samples,
            "num_chunks": num_chunks,
        }
    )


def generate_task_6(chunk_size: int = 100) -> Task:
    """Generate Task 6 (ECG reconstruction from real MIMIC data)."""
    return generate_real_ecg_reconstruction_task(chunk_size=chunk_size)


# =============================================================================
# Master Task Generator
# =============================================================================

def generate_random_task(
    task_weights: Optional[Dict[int, float]] = None,
) -> Task:
    """
    Generate a random pretraining task.

    Args:
        task_weights: Optional weights for each task type (0-6).
                     Default is uniform distribution.
    """
    if task_weights is None:
        task_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}

    task_types = list(task_weights.keys())
    weights = [task_weights[t] for t in task_types]
    weights = [w / sum(weights) for w in weights]  # Normalize

    task_type = np.random.choice(task_types, p=weights)

    generators = {
        0: generate_task_0,
        1: lambda: generate_task_1(num_values=50),
        2: lambda: generate_task_2(num_tokens=20),
        3: lambda: generate_task_3(duration=2.0),
        4: lambda: generate_task_4(duration=1.0),
        5: lambda: generate_task_5(duration=2.0, chunk_size=100),
        6: lambda: generate_task_6(chunk_size=100),
    }

    return generators[task_type]()


def task_to_conversation(task: Task) -> List[Dict[str, str]]:
    """Convert a Task to the conversation format used by the dataloader."""
    return [{"from": turn.role, "value": turn.content} for turn in task.turns]
