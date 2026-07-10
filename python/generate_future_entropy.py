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
