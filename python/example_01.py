from mlx_lm import load

from generate_future_entropy import generate_future_entropy

model_name = "Qwen/Qwen3-8B-MLX-6bit"

model, tokenizer = load(model_name)

text = generate_future_entropy(
    model=model,
    tokenizer=tokenizer,
    prompt="Write the opening paragraph of a ghost story set in a railway station.",
    max_new_tokens=250,
    candidate_top_k=32,
    future_top_n=20,
    alpha=0.0,
    temperature=1.0,
    stop_sequence="\n\n",
)

print(text)
