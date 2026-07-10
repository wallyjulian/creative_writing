"""
Driver for generate_future_entropy_wave (alpha-wave / rhythmic-decoding
future-entropy sampler). Loads the model and generates from a fixed
prompt; edit `prompt` below to try other text.

For a CLI equivalent of the fixed-alpha sampler, see example_01.py.
"""

from mlx_lm import load

from generate_future_entropy_wave import generate_future_entropy_wave

model_name = "Qwen/Qwen3-8B-MLX-6bit"

model, tokenizer = load(model_name)

text = generate_future_entropy_wave(
    model=model,
    tokenizer=tokenizer,
    prompt="Write the opening paragraph of a ghost story set in a railway station.",
    max_new_tokens=250,
    candidate_top_k=32,
    future_top_n=20,
    temperature=1.0,
    wave_period=80,
    stop_sequence="\n\n",
)

print(text)
