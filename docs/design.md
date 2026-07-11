---

editor_options: 
  markdown: 
    wrap: 72
---

# Future-Entropy Sampler: Design Document

## Overview

This project implements the "future-entropy sampler" from Will Kurt's article [Making LLMs Better at Creative Writing using Entropy](https://www.countbayesie.com/blog/2026/7/1/making-llms-better-at-creative-writing-using-entropy) (saved locally as `Making LLMs Better at Creative Writing using Entropy — Count Bayesie.md` in this directory).

The core idea: instead of picking the next token purely by its own probability (greedy or temperature sampling), score each candidate token by how much creative freedom it leaves for the *token after that* — its "future entropy". Tokens that lead into an open, diverse continuation are favored over tokens that funnel into a single forced/cliché continuation.

The article's pseudocode targets PyTorch. This codebase is an MLX port (`python/`), built to run on Apple Silicon.

## Algorithm

### Entropy and normalized entropy

Shannon entropy of a probability distribution measures how uncertain / open the outcome is:

```         
H = -sum(p(x) * log p(x))
```

To compare entropy across candidate sets of different sizes, it's divided by its maximum possible value (`log n` for `n` outcomes), giving a `[0, 1]`-normalized entropy: 0 means a forced/deterministic outcome, 1 means maximally open.

### Future-entropy score

For a candidate token `w` under the current context `c`:

1.  Append `w` to the context, run the model, get the distribution over the *next* token, `q_w(V) = p(V | c, w)`.
2.  Take `q_w`'s top-`n` tokens, renormalize over just that set, compute its normalized entropy, `H(w)`.
3.  Combine `H(w)` with `w`'s own probability under `c`:

```         
score(w) = p(w | c)^a * H(w)^b
a = 1 - max(0, alpha)
b = 1 - max(0, -alpha)
```

`alpha` is in `[-1, 1]`: - `alpha = -1`: `b = 0`, score is probability-only (behaves like plain top-k sampling). - `alpha = 1`: `a = 0`, score is future-entropy-only. - `alpha = 0`: both factors weighted equally.

The final token is sampled from the renormalized `score` distribution over the probed candidates.

### Alpha-wave ("rhythmic decoding")

Rather than fixing `alpha` for an entire generation, `alpha` can be swept along a sine wave across generation steps, oscillating between a low and high value with a given period. This lets text alternate between more predictable and more entropy-driven stretches, which the article's author found produced the most engaging results.

## Module map

```         
python/
├── normalized_entropy.py         # entropy math + future-entropy scoring/sampling
├── choose_probe_candidates.py    # picks which candidates get the extra forward pass
├── alpha_wave.py                 # sine-wave schedule for alpha
├── generate_future_entropy.py    # generation loop, fixed alpha
├── generate_future_entropy_wave.py  # generation loop, alpha-wave
├── example_01.py                 # CLI driver for the fixed-alpha loop
├── example_02.py                 # driver for the alpha-wave loop
└── requirements.txt               # mlx, mlx-lm, transformers<5
```

Data flow per generated token (both generation loops):

1.  `generate_future_entropy(_wave)` calls `sample_future_entropy_token` (in `normalized_entropy.py`) with the current `input_ids`.
2.  `sample_future_entropy_token` runs one forward pass to get the current top-`candidate_top_k` next-token candidates.
3.  It calls `future_entropy_scores`, which:
    - calls `choose_probe_candidates` to pick a subset of those candidates to actually probe (always the top `probe_n_head`, plus a weighted random sample of `probe_n_tail` from the rest — probing every candidate would mean one extra forward pass per candidate, on top of the one per generation step, which is the dominant cost of this sampler);
    - runs one forward pass per probed candidate to get its future distribution and normalized entropy;
    - combines each candidate's probability and future entropy into a score via the `alpha` formula above.
4.  One token is sampled from the score distribution and appended to `input_ids`. The wave variant recomputes `alpha` at each step from `alpha_wave` before scoring.
5.  Loop continues until `max_new_tokens`, an EOS token, or (if configured) `stop_sequence` is found in the newly generated text.

## MLX port notes

The article's pseudocode is PyTorch-oriented; a few things needed MLX-specific solutions:

- **No `torch.multinomial(replacement=False)` equivalent.** `mx.random.categorical` only samples with replacement. Weighted sampling without replacement (used to pick the random "tail" probe candidates) is implemented via the Gumbel-top-k trick in `choose_probe_candidates._sample_without_replacement`: add Gumbel noise to the log-probabilities and take the top-k by that perturbed score.
- **No `torch.topk` equivalent.** `normalized_entropy.top_k` implements it via `mx.argsort`.
- **Chat template is required for instruct models.** Raw `tokenizer.encode(prompt)` is treated by an instruct model as a document to keep extending, producing meta-commentary/preamble instead of following the instruction. Both generation loops use `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=False)`.
- **`enable_thinking=False` matters for stop-sequence correctness.** Qwen3's default chat template opens a `<think>...</think>` block that (if thinking were enabled) is prompt structure, not generated content — but its presence caused a naive stop-sequence check to trigger on a blank line inside it. Disabling thinking keeps this out of the way, and the returned text is sliced to only the tokens generated after the prompt (`input_ids[0, prompt_len:]`), so chat-template control tokens never leak into the output.

## Stop sequence

Both generation loops take an optional `stop_sequence` (e.g. `"\n\n"`). After each token, only the newly generated suffix is decoded and checked for the substring — cheap relative to the forward-pass cost that dominates this sampler, since it's a string operation rather than another model call.

**Known limitation:** this only helps when the model naturally produces the stop sequence. For a single-paragraph prompt, the model may finish one good paragraph and then, with no double-newline ever appearing, run to the full `max_new_tokens` budget producing degenerate filler. The straightforward fix is a shorter `max_new_tokens`, left as a per-experiment tuning knob rather than a default change.

## Model choice

Default model: `Qwen/Qwen3-8B-MLX-6bit` (dense `Qwen3ForCausalLM`, no vision component). Chosen after comparison against `mlx-community/Qwen2.5-1.5B-Instruct-4bit`: markedly better prose (coherent, no prompt-echoing, no asterisk spam) at roughly 4.5x the wall-clock cost.

Before adopting any new model for this pipeline, check `config.json` for a `vision_config` key and the `architectures` field — a vision-language model (e.g. `Qwen3_5MoeForConditionalGeneration`) is incompatible with `mlx_lm.load`, which expects a plain causal LM. Pulling just `config.json`/`tokenizer_config.json` via `huggingface_hub.hf_hub_download` before committing to a full multi-GB download avoids wasting time/disk on the wrong architecture.

Because `future_entropy_scores` performs `probe_n_head + probe_n_tail` (default 12) extra forward passes per generation step on top of the main one, generation time scales convexly (worse than linearly) with `max_new_tokens`, and larger models cost roughly proportionally to their active parameter count. Budget wall-clock time accordingly before long or batched runs.

## Usage

``` bash
pip install -r python/requirements.txt

# fixed-alpha sampler, CLI prompt
python python/example_01.py "Write the opening paragraph of a ghost story set in a railway station."
python python/example_01.py --alpha 0.5 --max-new-tokens 100

# alpha-wave sampler (edit the prompt in the file directly)
python python/example_02.py
```
