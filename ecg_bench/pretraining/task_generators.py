"""
Task generators for pretraining.

Generates various tasks to help the model understand ECG tokens:
- Task 0: Values to tokens (given values, output tokens) - explicit mapping
- Task 1: Tokens to values (given tokens, output values) - reverse mapping
- Task 2: Token comparison (magnitude, sequence, halfway) - implicit understanding
- Task 3: Wave classification and property extraction (multi-turn)
- Task 4: Wave transformation (frequency/amplitude changes)
- Task 5: Wave reconstruction from specification (chunked)
- Task 6: ECG reconstruction from real MIMIC data
- Task 7: Long-range wave prediction
- Task 8: Long-range ECG prediction
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
# Task 0: Values to Tokens (explicit mapping - easy)
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


def generate_task_0(num_values: int = 50) -> Task:
    """Generate Task 0 (values to tokens)."""
    return generate_values_to_tokens_task(num_values)


# =============================================================================
# Task 1: Tokens to Values (reverse mapping - medium)
# =============================================================================

def generate_tokens_to_values_task(num_tokens: int = 20) -> Task:
    """
    Generate a task where given tokens, the model outputs values.
    Fewer tokens than Task 0 since this is harder.
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


def generate_task_1(num_tokens: int = 20) -> Task:
    """Generate Task 1 (tokens to values)."""
    return generate_tokens_to_values_task(num_tokens)


# =============================================================================
# Task 2: Token Comparison Questions (implicit understanding - harder)
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
    Generate a task asking which token comes directly after another in an arithmetic sequence.

    Uses deterministic arithmetic sequences (linear progression) so the answer
    is uniquely determined by the pattern. E.g., ecg_100 ecg_120 ecg_140 ecg_160 → ecg_180

    Step sizes are limited to "round" numbers (multiples of 5 or 10) to make
    the arithmetic more learnable for the model.
    """
    # Use round step sizes that are easier to learn
    # Instead of arbitrary [-30, 30], use multiples of 5 or 10
    round_steps = [-20, -15, -10, -5, 5, 10, 15, 20]
    step = random.choice(round_steps)

    # Determine valid start range based on step direction
    # Need 5 tokens in sequence, so start + 4*step must be in [0, 599]
    if step > 0:
        min_start = 0
        max_start = 599 - 4 * step
    else:
        min_start = -4 * step
        max_start = 599

    start = random.randint(min_start, max_start)

    # Generate 5 token indices (4 for context, 1 for answer)
    token_indices = [start + i * step for i in range(5)]

    # Convert to token strings
    context_tokens = [f"ecg_{idx}" for idx in token_indices[:4]]
    next_token = f"ecg_{token_indices[4]}"
    current_token = context_tokens[-1]
    context_str = " ".join(context_tokens)

    turns = [
        ConversationTurn("human", f"Given this sequence of timeseries tokens: {context_str}\nWhat token comes directly after {current_token} in this sequence?"),
        ConversationTurn("assistant", f"{next_token}"),
    ]

    return Task(
        task_type="sequence_next",
        turns=turns,
        metadata={"context": context_str, "current": current_token, "next": next_token, "step": step}
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


def generate_task_2() -> Task:
    """Generate Task 2 (all token comparison sub-tasks - for backward compatibility)."""
    generators = [
        generate_magnitude_comparison_task,
        generate_sequence_next_token_task,
        generate_halfway_token_task,
        generate_token_value_comparison_task,
    ]
    return random.choice(generators)()


def generate_task_1_comparison() -> Task:
    """
    Generate Stage 1 task: magnitude_comparison or value_comparison.

    These are simpler comparison tasks that only require understanding
    which token index is higher/lower (no arithmetic required).
    """
    generators = [
        generate_magnitude_comparison_task,
        generate_token_value_comparison_task,
    ]
    return random.choice(generators)()


def generate_task_2_arithmetic() -> Task:
    """
    Generate Stage 2 task: sequence_next or halfway_token.

    These are harder tasks that require arithmetic operations
    on token indices.
    """
    generators = [
        generate_sequence_next_token_task,
        generate_halfway_token_task,
    ]
    return random.choice(generators)()


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

    # Generate random params with values rounded to 1 decimal place for learnability
    amplitude = round(np.random.uniform(0.8, 3.0), 1)
    frequency = round(np.random.uniform(1.0, 5.0), 1)
    duty_cycle = round(np.random.uniform(0.1, 0.9), 1)

    params = WaveParams(
        wave_type=wave_type,
        amplitude=amplitude,
        frequency=frequency,
        phase=np.random.uniform(0, 2 * np.pi),
        offset=0.0,  # Keep it simpler for classification
        duty_cycle=duty_cycle,
        decay_rate=np.random.uniform(0.5, 3.0),
    )

    _, wave = generate_wave(duration, params)
    tokens = values_to_tokens(wave)
    tokens_str = tokens_to_string(tokens)

    wave_name = wave_type_description(wave_type)
    properties = get_wave_properties_for_type(wave_type)

    # Map wave type to single-word answer (no "wave" suffix)
    wave_type_names = {
        WaveType.SINE: "sine",
        WaveType.COSINE: "cosine",
        WaveType.TRIANGLE: "triangle",
        WaveType.SAWTOOTH: "sawtooth",
        WaveType.SQUARE: "square",
        WaveType.PULSE: "pulse",
    }
    wave_name_short = wave_type_names[wave_type]

    turns = []

    # Turn 1: Classification - answer is just the type (e.g., "sine")
    turns.append(ConversationTurn("human", f"Analyze this timeseries signal and identify what type of wave it is:\n{tokens_str}"))
    turns.append(ConversationTurn("assistant", wave_name_short))

    # Turn 2: Amplitude - answer is just the number (e.g., "2.5")
    if "amplitude" in properties:
        turns.append(ConversationTurn("human", "What is the amplitude of this wave in mV?"))
        turns.append(ConversationTurn("assistant", f"{format_value(params.amplitude, precision=1)}"))

    # Turn 3: Frequency - answer is just the number (e.g., "3.0")
    if "frequency" in properties:
        turns.append(ConversationTurn("human", "What is the frequency of this wave in Hz?"))
        turns.append(ConversationTurn("assistant", f"{format_value(params.frequency, precision=1)}"))

    # Turn 4: Additional property based on wave type - answer is just the number (e.g., "50")
    if wave_type in [WaveType.SQUARE, WaveType.PULSE] and "duty_cycle" in dir(params):
        turns.append(ConversationTurn("human", "What is the duty cycle of this wave as a percentage?"))
        turns.append(ConversationTurn("assistant", f"{int(params.duty_cycle * 100)}"))

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
# Task 5: Wave Generation (Scheduled Sampling)
# =============================================================================

def generate_wave_generation_task(
    duration: float = 2.0,
) -> Task:
    """
    Generate a task where the model generates a complete wave from specification.

    This task uses scheduled sampling during training - the model must generate
    the full sequence and may use its own predictions as input during training.
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

    # Convert wave to tokens
    wave_tokens = values_to_tokens(wave)
    wave_str = tokens_to_string(wave_tokens)

    # Specification prompt
    spec = f"{wave_type_description(wave_type)} with amplitude {amplitude} mV and frequency {frequency} Hz"

    turns = [
        ConversationTurn("human", f"Generate a {spec} sampled at 250 Hz for {duration} seconds."),
        ConversationTurn("assistant", wave_str),
    ]

    return Task(
        task_type="wave_generation",
        turns=turns,
        metadata={
            "wave_type": wave_type.value,
            "amplitude": amplitude,
            "frequency": frequency,
            "duration": duration,
            "total_samples": total_samples,
            "scheduled_sampling": True,  # Flag for training loop
        }
    )


def generate_task_5(duration: float = 2.0, chunk_size: int = 100) -> Task:
    """Generate Task 5 (wave generation with scheduled sampling)."""
    # chunk_size is kept for API compatibility but not used
    return generate_wave_generation_task(duration)


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


def generate_real_ecg_generation_task(
    max_samples: int = 500,  # Limit samples to keep task reasonable (500 samples = 2 seconds at 250 Hz)
    partition: str = "task6",  # Use task6 partition to avoid overlap with task8
) -> Task:
    """
    Generate a task where the model outputs ECG tokens from real MIMIC data.

    Uses one of the 4 leads (II, aVR, V1, V4) from a real ECG recording.
    Includes the diagnostic report in the prompt.

    This task uses scheduled sampling during training.
    """
    loader = get_ecg_loader()

    # Try to get a valid ECG with a non-empty report
    max_attempts = 20
    for _ in range(max_attempts):
        ecg_signal, metadata = loader.get_random_ecg(partition=partition)
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

    # Convert to tokens
    ecg_tokens = values_to_tokens(lead_signal)
    ecg_str = tokens_to_string(ecg_tokens)

    # Prompt with diagnosis
    prompt = f"Generate the {lead_name} lead ECG signal for a patient with the following diagnosis: {report}\n\nOutput the timeseries tokens sampled at 250 Hz for {format_value(duration)} seconds."

    turns = [
        ConversationTurn("human", prompt),
        ConversationTurn("assistant", ecg_str),
    ]

    return Task(
        task_type="ecg_generation",
        turns=turns,
        metadata={
            "source": "mimic-iv-ecg",
            "ecg_path": metadata.get("ecg_path", ""),
            "report": report,
            "lead": lead_name,
            "lead_idx": lead_idx,
            "duration": duration,
            "total_samples": total_samples,
            "scheduled_sampling": True,  # Flag for training loop
        }
    )


def generate_task_6(chunk_size: int = 100) -> Task:
    """Generate Task 6 (ECG generation from real MIMIC data with scheduled sampling)."""
    # chunk_size is kept for API compatibility but not used
    return generate_real_ecg_generation_task(partition="task6")


# =============================================================================
# Task 7: Long-Range Wave Prediction
# =============================================================================

# Prediction offsets for long-range tasks (relative to context end)
LONG_RANGE_OFFSETS = [10, 30, 60, 100, 150, 210, 280]


def generate_long_range_wave_prediction_task(
    context_tokens: int = 200,
    duration: float = 3.0,  # Need longer signal to have tokens at far offsets
) -> Task:
    """
    Generate a task testing long-range pattern understanding for synthetic waves.

    Given ~200 tokens of a wave pattern, predict specific tokens far into the future:
    tokens at offsets +10, +30, +60, +100, +150, +210, +280 from context end.

    This tests whether the model understands the periodic structure well enough
    to extrapolate far beyond the given context.
    """
    wave_type = random.choice([
        WaveType.SINE, WaveType.COSINE, WaveType.TRIANGLE,
        WaveType.SAWTOOTH, WaveType.SQUARE,
    ])

    # Cap amplitude at 3.0 to stay within [-3, 3] mV range
    amplitude = round(np.random.uniform(0.8, 3.0), 2)
    frequency = round(np.random.uniform(0.5, 2.0), 2)  # Lower freq for clearer patterns

    params = WaveParams(
        wave_type=wave_type,
        amplitude=amplitude,
        frequency=frequency,
        phase=0,
        offset=0,
    )

    _, wave = generate_wave(duration, params)
    all_tokens = values_to_tokens(wave)

    # We need enough tokens for context + max offset
    min_required = context_tokens + max(LONG_RANGE_OFFSETS) + 1
    if len(all_tokens) < min_required:
        # Extend duration if needed
        _, wave = generate_wave(duration * 2, params)
        all_tokens = values_to_tokens(wave)

    if len(all_tokens) < min_required:
        raise RuntimeError(f"Not enough tokens generated: {len(all_tokens)} < {min_required}")

    # Context tokens
    context = all_tokens[:context_tokens]
    context_str = tokens_to_string(context)

    # Build the target tokens at specified offsets
    target_tokens = []
    target_positions = []
    for offset in LONG_RANGE_OFFSETS:
        pos = context_tokens + offset
        if pos < len(all_tokens):
            target_tokens.append(all_tokens[pos])
            target_positions.append(pos + 1)  # 1-indexed for user

    # Format answer as concatenated tokens (no spaces, matching other tasks)
    answer = tokens_to_string(target_tokens)

    # Format the question with positions
    positions_str = ", ".join([str(p) for p in target_positions])

    wave_name = wave_type_description(wave_type)
    prompt = (
        f"Here is a {wave_name} signal with amplitude {amplitude} mV and frequency {frequency} Hz:\n"
        f"{context_str}\n\n"
        f"Predict the tokens at positions {positions_str}."
    )

    turns = [
        ConversationTurn("human", prompt),
        ConversationTurn("assistant", answer),
    ]

    return Task(
        task_type="long_range_wave_prediction",
        turns=turns,
        metadata={
            "wave_type": wave_type.value,
            "amplitude": amplitude,
            "frequency": frequency,
            "context_tokens": context_tokens,
            "prediction_offsets": LONG_RANGE_OFFSETS,
            "target_positions": target_positions,
        }
    )


def generate_task_7(context_tokens: int = 200) -> Task:
    """Generate Task 7 (long-range wave prediction)."""
    return generate_long_range_wave_prediction_task(context_tokens=context_tokens)


# =============================================================================
# Task 8: Long-Range ECG Prediction
# =============================================================================

def generate_long_range_ecg_prediction_task(
    context_tokens: int = 200,
    min_signal_length: int = 500,  # Need at least this many samples
) -> Task:
    """
    Generate a task testing long-range pattern understanding for real ECGs.

    Given ~200 tokens of a real ECG pattern, predict specific tokens far into the future.
    Uses ECGs from the task8 partition (separate from Task 6) to avoid data leakage.
    """
    loader = get_ecg_loader()

    # Calculate minimum required samples
    min_required = context_tokens + max(LONG_RANGE_OFFSETS) + 1

    # Try to get a valid ECG with enough samples
    max_attempts = 50
    for _ in range(max_attempts):
        ecg_signal, metadata = loader.get_random_ecg(partition="task8")
        if ecg_signal is None:
            continue

        # Select a random lead
        lead_idx_in_list = random.randint(0, len(ECG_RAW_LEAD_INDICES) - 1)
        lead_idx = ECG_RAW_LEAD_INDICES[lead_idx_in_list]
        lead_name = ECG_RAW_LEADS[lead_idx_in_list]

        lead_signal = extract_lead_signal(ecg_signal, lead_idx)

        if len(lead_signal) >= min_required:
            break
    else:
        raise RuntimeError("Failed to load ECG with sufficient length after multiple attempts")

    # Clamp signal to valid range
    lead_signal = clamp_wave(lead_signal)

    # Convert to tokens
    all_tokens = values_to_tokens(lead_signal)

    # Randomly select a starting point that allows for full context + predictions
    max_start = len(all_tokens) - min_required
    if max_start <= 0:
        start_idx = 0
    else:
        start_idx = random.randint(0, max_start)

    # Get context tokens
    context = all_tokens[start_idx:start_idx + context_tokens]
    context_str = tokens_to_string(context)

    # Build the target tokens at specified offsets
    target_tokens = []
    target_positions = []
    for offset in LONG_RANGE_OFFSETS:
        pos = start_idx + context_tokens + offset
        if pos < len(all_tokens):
            target_tokens.append(all_tokens[pos])
            target_positions.append(context_tokens + offset + 1)  # Relative 1-indexed

    # Format answer as concatenated tokens (no spaces, matching other tasks)
    answer = tokens_to_string(target_tokens)

    # Format the question with positions
    positions_str = ", ".join([str(p) for p in target_positions])

    # Get diagnosis if available
    report = metadata.get("report", "").strip()
    if report:
        prompt = (
            f"Here is a {lead_name} lead ECG signal from a patient with: {report}\n\n"
            f"{context_str}\n\n"
            f"Predict the tokens at positions {positions_str}."
        )
    else:
        prompt = (
            f"Here is a {lead_name} lead ECG signal:\n"
            f"{context_str}\n\n"
            f"Predict the tokens at positions {positions_str}."
        )

    turns = [
        ConversationTurn("human", prompt),
        ConversationTurn("assistant", answer),
    ]

    return Task(
        task_type="long_range_ecg_prediction",
        turns=turns,
        metadata={
            "source": "mimic-iv-ecg",
            "ecg_path": metadata.get("ecg_path", ""),
            "report": report,
            "lead": lead_name,
            "lead_idx": lead_idx,
            "context_tokens": context_tokens,
            "prediction_offsets": LONG_RANGE_OFFSETS,
            "target_positions": target_positions,
        }
    )


def generate_task_8(context_tokens: int = 200) -> Task:
    """Generate Task 8 (long-range ECG prediction from real MIMIC data)."""
    return generate_long_range_ecg_prediction_task(context_tokens=context_tokens)


# =============================================================================
# Master Task Generator
# =============================================================================

def generate_random_task(
    task_weights: Optional[Dict[int, float]] = None,
) -> Task:
    """
    Generate a random pretraining task.

    Args:
        task_weights: Optional weights for each task type (0-8).
                     Default is uniform distribution.
    """
    if task_weights is None:
        task_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}

    task_types = list(task_weights.keys())
    weights = [task_weights[t] for t in task_types]
    weights = [w / sum(weights) for w in weights]  # Normalize

    task_type = np.random.choice(task_types, p=weights)

    generators = {
        0: generate_task_0,
        1: lambda: generate_task_1(num_tokens=50),
        2: generate_task_2,
        3: lambda: generate_task_3(duration=2.0),
        4: lambda: generate_task_4(duration=1.0),
        5: lambda: generate_task_5(duration=2.0, chunk_size=100),
        6: lambda: generate_task_6(chunk_size=100),
        7: lambda: generate_task_7(context_tokens=200),
        8: lambda: generate_task_8(context_tokens=200),
    }

    return generators[task_type]()


def task_to_conversation(task: Task) -> List[Dict[str, str]]:
    """Convert a Task to the conversation format used by the dataloader."""
    return [{"from": turn.role, "value": turn.content} for turn in task.turns]
