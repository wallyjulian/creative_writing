"""
Selects which candidate next-tokens get probed with an extra forward pass
in normalized_entropy.py's future_entropy_scores.

Probing every top-k candidate would mean one forward pass per candidate on
top of the main one, at every generation step, so only a subset is probed:
the most likely few tokens are always probed, plus a small random sample
of the remaining tail (weighted by their probability) so lower-ranked but
still plausible tokens occasionally get a chance to influence the score.
"""

import mlx.core as mx


def _sample_without_replacement(probs: mx.array, num_samples: int) -> mx.array:
    """
    Weighted sampling without replacement via the Gumbel-top-k trick.
    mx.random.categorical only samples with replacement, so this fills the
    same role as torch.multinomial(..., replacement=False).
    """
    log_probs = mx.log(probs + 1e-12)
    gumbel_noise = -mx.log(-mx.log(mx.random.uniform(shape=probs.shape) + 1e-12) + 1e-12)
    order = mx.argsort(log_probs + gumbel_noise)[::-1]
    return order[:num_samples]


def choose_probe_candidates(candidate_ids, candidate_probs, n_head=8, n_tail=4):
    """
    Always probe the most likely candidates.
    Randomly probe a few lower-ranked candidates.
    """
    head_ids = candidate_ids[:n_head]
    head_probs = candidate_probs[:n_head]

    tail_ids = candidate_ids[n_head:]
    tail_probs = candidate_probs[n_head:]

    if tail_ids.shape[0] == 0 or n_tail == 0:
        return head_ids, head_probs

    tail_probs_norm = tail_probs / tail_probs.sum()
    sampled_tail_idx = _sample_without_replacement(
        tail_probs_norm,
        num_samples=min(n_tail, tail_ids.shape[0]),
    )

    probe_ids = mx.concatenate([head_ids, tail_ids[sampled_tail_idx]])
    probe_probs = mx.concatenate([head_probs, tail_probs[sampled_tail_idx]])

    probe_probs = probe_probs / probe_probs.sum()

    return probe_ids, probe_probs
