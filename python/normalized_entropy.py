import math
import torch
import torch.nn.functional as F


def normalized_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    probs is a 1D probability vector that sums to 1.
    Returns entropy normalized to the range 0 to 1.
    """
    n = probs.numel()
    if n <= 1:
        return torch.tensor(0.0, device=probs.device)

    entropy = -(probs * torch.log(probs + eps)).sum()
    return entropy / math.log(n)


@torch.no_grad()
def future_entropy_scores(
    model,
    input_ids: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_probs: torch.Tensor,
    future_top_n: int = 20,
    alpha: float = 0.0,
    temperature: float = 1.0,
):
    """
    Computes Future-entropy scores for candidate next tokens.

    input_ids shape is [1, seq_len]
    candidate_ids shape is [k]
    candidate_probs shape is [k]
    """

    entropies = []

    for token_id in candidate_ids:
        next_input_ids = torch.cat(
            [input_ids, token_id.view(1, 1)],
            dim=1
        )

        out = model(next_input_ids)
        future_logits = out.logits[:, -1, :].squeeze(0)

        if temperature != 1.0:
            future_logits = future_logits / temperature

        future_probs = F.softmax(future_logits, dim=-1)

        top_probs, _ = torch.topk(future_probs, k=future_top_n)
        top_probs = top_probs / top_probs.sum()

        h = normalized_entropy(top_probs)
        entropies.append(h)

    entropies = torch.stack(entropies)

    alpha = max(-1.0, min(1.0, alpha))

    a = 1.0 - max(0.0, alpha)
    b = 1.0 - max(0.0, -alpha)

    eps = 1e-12

    scores = (candidate_probs + eps).pow(a) * (entropies + eps).pow(b)
    scores = scores / scores.sum()

    return scores, entropies


@torch.no_grad()
def sample_future_entropy_token(
    model,
    input_ids: torch.Tensor,
    candidate_top_k: int = 40,
    future_top_n: int = 20,
    alpha: float = 0.0,
    temperature: float = 1.0,
):
    """
    Samples one token using Future-entropy sampling.
    """

    out = model(input_ids)
    logits = out.logits[:, -1, :].squeeze(0)

    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    candidate_probs, candidate_ids = torch.topk(probs, k=candidate_top_k)
    candidate_probs = candidate_probs / candidate_probs.sum()

    scores, entropies = future_entropy_scores(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_probs=candidate_probs,
        future_top_n=future_top_n,
        alpha=alpha,
        temperature=temperature,
    )

    sampled_idx = torch.multinomial(scores, num_samples=1).item()
    chosen_id = candidate_ids[sampled_idx].view(1, 1)

    debug = {
        "candidate_ids": candidate_ids,
        "candidate_probs": candidate_probs,
        "future_entropies": entropies,
        "final_scores": scores,
        "sampled_index": sampled_idx,
    }

    return chosen_id, debug
