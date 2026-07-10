"""
Fixed-alpha generation loop for the future-entropy sampler.

Runs the future-entropy token-by-token decoding process described in
Will Kurt's "Making LLMs Better at Creative Writing using Entropy"
(see /docs), with a single alpha value held constant for the whole
generation. See generate_future_entropy_wave.py for the variant that
oscillates alpha over time.
"""

import mlx.core as mx

from normalized_entropy import sample_future_entropy_token


def generate_future_entropy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    candidate_top_k: int = 40,
    future_top_n: int = 20,
    alpha: float = 0.0,
    temperature: float = 1.0,
    stop_sequence: str = None,
):
    """
    Generate text from `prompt` using future-entropy sampling.

    Args:
        model: an mlx_lm causal LM.
        tokenizer: the matching mlx_lm tokenizer.
        prompt: user-facing instruction/prompt text.
        max_new_tokens: hard cap on generated tokens.
        candidate_top_k: size of the initial top-k candidate pool considered
            at each step (before future-entropy probing).
        future_top_n: size of the top-n pool used to compute each
            candidate's future entropy.
        alpha: blends original probability vs. future entropy in the score,
            in [-1, 1]. -1 behaves like plain top-k sampling, 1 is driven
            almost entirely by future entropy. See future_entropy_scores in
            normalized_entropy.py for the exact formula.
        temperature: softmax temperature applied to both the current-step
            logits and each probed future-step logits.
        stop_sequence: optional substring (e.g. "\\n\\n") that ends
            generation early once it appears in the newly generated text.
            Only checked against generated tokens, never the prompt.

    Returns:
        The generated continuation as a string (prompt/chat-template
        tokens are excluded).
    """
    # The model is an instruct model: raw tokenizer.encode(prompt) would be
    # treated as a document to extend rather than an instruction to follow.
    # enable_thinking=False keeps Qwen3's <think> block empty so it doesn't
    # interfere with stop_sequence matching (it's part of the prompt, not
    # generated output).
    prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = mx.array(prompt_tokens)[None, :]
    prompt_len = input_ids.shape[1]

    for _ in range(max_new_tokens):
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
