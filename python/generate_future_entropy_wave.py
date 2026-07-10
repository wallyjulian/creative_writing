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
