def choose_probe_candidates(candidate_ids, candidate_probs, n_head=8, n_tail=4):
    """
    Always probe the most likely candidates.
    Randomly probe a few lower-ranked candidates.
    """
    head_ids = candidate_ids[:n_head]
    head_probs = candidate_probs[:n_head]

    tail_ids = candidate_ids[n_head:]
    tail_probs = candidate_probs[n_head:]

    if tail_ids.numel() == 0 or n_tail == 0:
        return head_ids, head_probs

    tail_probs = tail_probs / tail_probs.sum()
    sampled_tail_idx = torch.multinomial(
        tail_probs,
        num_samples=min(n_tail, tail_ids.numel()),
        replacement=False
    )

    probe_ids = torch.cat([head_ids, tail_ids[sampled_tail_idx]])
    probe_probs = torch.cat([head_probs, candidate_probs[n_head:][sampled_tail_idx]])

    probe_probs = probe_probs / probe_probs.sum()

    return probe_ids, probe_probs
