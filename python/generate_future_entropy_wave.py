"""
Alpha-wave generation loop for the future-entropy sampler ("rhythmic
decoding" in Will Kurt's article, see /docs).

Same future-entropy decoding process as generate_future_entropy.py, but
instead of a fixed alpha, alpha is swept along a sine wave (alpha_wave.py)
across generation steps. This lets the sampler alternate between
predictable and more entropy-driven text as it writes, rather than
committing to one behavior for the whole generation.
"""

import mlx.core as mx

from alpha_wave import alpha_wave
from normalized_entropy import sample_future_entropy_token


def generate_future_entropy_wave(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 300,
    candidate_top_k: int = 40,
    future_top_n: int = 20,
    temperature: float = 1.0,
    wave_period: int = 80,
    min_alpha: float = -0.7,
    max_alpha: float = 0.8,
    stop_sequence: str = None,
):
    """
    Generate text from `prompt` using future-entropy sampling with alpha
    oscillating sinusoidally over the generation steps.

    Args:
        model: an mlx_lm causal LM.
        tokenizer: the matching mlx_lm tokenizer.
        prompt: user-facing instruction/prompt text.
        max_new_tokens: hard cap on generated tokens.
        candidate_top_k: size of the initial top-k candidate pool considered
            at each step (before future-entropy probing).
        future_top_n: size of the top-n pool used to compute each
            candidate's future entropy.
        temperature: softmax temperature applied to both the current-step
            logits and each probed future-step logits.
        wave_period: number of generation steps for one full sine cycle of
            alpha (see alpha_wave.py).
        min_alpha: trough value of the alpha sine wave, in [-1, 1].
        max_alpha: peak value of the alpha sine wave, in [-1, 1].
        stop_sequence: optional substring (e.g. "\\n\\n") that ends
            generation early once it appears in the newly generated text.
            Only checked against generated tokens, never the prompt.

    Returns:
        The generated continuation as a string (prompt/chat-template
        tokens are excluded).
    """
    # See generate_future_entropy.py for why the chat template and
    # enable_thinking=False are needed here.
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = mx.array(prompt_tokens)[None, :]
    prompt_len = input_ids.shape[1]

    for step in range(max_new_tokens):
        alpha = alpha_wave(
            step=step,
            period=wave_period,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
        )

        next_id, debug = sample_future_entropy_token(
            model=model,
            input_ids=input_ids,
            candidate_top_k=candidate_top_k,
            future_top_n=future_top_n,
            alpha=alpha,
            temperature=temperature,
        )

        input_ids = mx.concatenate([input_ids, next_id], axis=1)

        if next_id.item() in tokenizer.eos_token_ids:
            break

        if stop_sequence and stop_sequence in tokenizer.decode(input_ids[0, prompt_len:].tolist()):
            break

    return tokenizer.decode(input_ids[0, prompt_len:].tolist())
