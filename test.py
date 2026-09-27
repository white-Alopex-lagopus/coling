import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoConfig

# model_name = "../model/Qwen3-0.6B-base"
models = [
    "../model/Qwen3-0.6B-base",
    "../model/Qwen3-1.7B-base",
    "../model/Qwen3-4B-base",
]

for name in models:
    config = AutoConfig.from_pretrained(name)

    print("\n", name)
    print("hidden:", config.hidden_size)
    print("layers:", config.num_hidden_layers)
    print("heads:", config.num_attention_heads)
    print("kv heads:", config.num_key_value_heads)
    print("head dim:", config.head_dim)


# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# print(model)

# print(model.config)

# print("hidden size:", model.config.hidden_size)
# print("num layers:", model.config.num_hidden_layers)
# print("num attention heads:", model.config.num_attention_heads)
# print("num KV heads:", model.config.num_key_value_heads)
# print("head dim:", model.config.head_dim)