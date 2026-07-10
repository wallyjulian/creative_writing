from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

text = generate_future_entropy(
    model=model,
    tokenizer=tokenizer,
    prompt="Write the opening paragraph of a ghost story set in a railway station.",
    max_new_tokens=250,
    candidate_top_k=32,
    future_top_n=20,
    alpha=0.0,
    temperature=1.0,
)

print(text)
