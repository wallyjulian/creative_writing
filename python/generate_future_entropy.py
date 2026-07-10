@torch.no_grad()
def generate_future_entropy(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    candidate_top_k: int = 40,
    future_top_n: int = 20,
    alpha: float = 0.0,
    temperature: float = 1.0,
):
    model.eval()

    device = next(model.parameters()).device

    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids.to(device)

    for _ in range(max_new_tokens):
        next_id, debug = sample_future_entropy_token(
            model=model,
            input_ids=input_ids,
            candidate_top_k=candidate_top_k,
            future_top_n=future_top_n,
            alpha=alpha,
            temperature=temperature,
        )

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if tokenizer.eos_token_id is not None:
            if next_id.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
