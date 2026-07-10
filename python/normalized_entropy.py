import math
import mlx.core as mx

from choose_probe_candidates import choose_probe_candidates


def normalized_entropy(probs: mx.array, eps: float = 1e-12) -> mx.array:
    """
    probs is a 1D probability vector that sums to 1.
    Returns entropy normalized to the range 0 to 1.
    """
    n = probs.shape[0]
    if n <= 1:
        return mx.array(0.0)

    entropy = -(probs * mx.log(probs + eps)).sum()
    return entropy / math.log(n)


def top_k(probs: mx.array, k: int):
    """
    Returns the top-k probabilities and their indices, sorted descending.
    """
    k = min(k, probs.shape[0])
    order = mx.argsort(probs)[::-1][:k]
    return probs[order], order


def future_entropy_scores(
    model,
    input_ids: mx.array,
    candidate_ids: mx.array,
    candidate_probs: mx.array,
    future_top_n: int = 20,
    alpha: float = 0.0,
    temperature: float = 1.0,
    probe_n_head: int = 8,
    probe_n_tail: int = 4,
):
    """
    Computes Future-entropy scores for a probed subset of candidate next tokens.

    input_ids shape is [1, seq_len]
    candidate_ids shape is [k]
    candidate_probs shape is [k]
    """
    probe_ids, probe_probs = choose_probe_candidates(
        candidate_ids, candidate_probs, n_head=probe_n_head, n_tail=probe_n_tail
    )

    entropies = []

    for token_id in probe_ids:
        next_input_ids = mx.concatenate(
            [input_ids, token_id.reshape(1, 1).astype(input_ids.dtype)],
            axis=1,
        )

        future_logits = model(next_input_ids)[:, -1, :].squeeze(0)

        if temperature != 1.0:
            future_logits = future_logits / temperature

        future_probs = mx.softmax(future_logits, axis=-1)

        top_probs, _ = top_k(future_probs, future_top_n)
        top_probs = top_probs / top_probs.sum()

        h = normalized_entropy(top_probs)
        entropies.append(h)

    entropies = mx.stack(entropies)

    alpha = max(-1.0, min(1.0, alpha))

    a = 1.0 - max(0.0, alpha)
    b = 1.0 - max(0.0, -alpha)

    eps = 1e-12

    scores = (probe_probs + eps) ** a * (entropies + eps) ** b
    scores = scores / scores.sum()

    return probe_ids, scores, entropies


def sample_future_entropy_token(
    model,
    input_ids: mx.array,
    candidate_top_k: int = 40,
    future_top_n: int = 20,
    alpha: float = 0.0,
    temperature: float = 1.0,
    probe_n_head: int = 8,
    probe_n_tail: int = 4,
):
    """
    Samples one token using Future-entropy sampling.
    """
    logits = model(input_ids)[:, -1, :].squeeze(0)

    if temperature != 1.0:
        logits = logits / temperature

    probs = mx.softmax(logits, axis=-1)

    candidate_probs, candidate_ids = top_k(probs, candidate_top_k)
    candidate_probs = candidate_probs / candidate_probs.sum()

    probe_ids, scores, entropies = future_entropy_scores(
        model=model,
        input_ids=input_ids,
        candidate_ids=candidate_ids,
        candidate_probs=candidate_probs,
        future_top_n=future_top_n,
        alpha=alpha,
        temperature=temperature,
        probe_n_head=probe_n_head,
        probe_n_tail=probe_n_tail,
    )

    sampled_idx = mx.random.categorical(mx.log(scores + 1e-12)).item()
    chosen_id = probe_ids[sampled_idx].reshape(1, 1).astype(input_ids.dtype)

    debug = {
        "candidate_ids": candidate_ids,
        "candidate_probs": candidate_probs,
        "probe_ids": probe_ids,
        "future_entropies": entropies,
        "final_scores": scores,
        "sampled_index": sampled_idx,
    }

    return chosen_id, debug
