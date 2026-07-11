"""
CLI driver for generate_future_entropy (fixed-alpha future-entropy
sampler). Loads the model once, then generates from a prompt given on the
command line.

Usage:
    python example_01.py "your prompt here"
    python example_01.py --alpha 0.5 --max-new-tokens 100
    python example_01.py  # uses the default ghost-story prompt
"""

import argparse

from mlx_lm import load

from generate_future_entropy import generate_future_entropy

model_name = "Qwen/Qwen3-8B-MLX-6bit"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Write the opening paragraph of a ghost story set in a railway station.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--candidate-top-k", type=int, default=32)
    parser.add_argument("--future-top-n", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--stop-sequence",
        default="\n\n",
        help='stop early when this substring appears in the generated text. Pass "" to disable and run to max_new_tokens.',
    )
    args = parser.parse_args()

    model, tokenizer = load(model_name)

    text = generate_future_entropy(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        candidate_top_k=args.candidate_top_k,
        future_top_n=args.future_top_n,
        alpha=args.alpha,
        temperature=args.temperature,
        stop_sequence=args.stop_sequence,
    )

    print(text)


if __name__ == "__main__":
    main()
