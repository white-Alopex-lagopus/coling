import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kvbridge import Mapper, transfer

# 1. 加载模型 (假设已安装 kvbridge: pip install git+https://github.com/Sidd03192/kvbridge)
model_small = AutoModelForCausalLM.from_pretrained("../model/Qwen3-0.6B-base", torch_dtype=torch.float16).cuda().eval()
model_large = AutoModelForCausalLM.from_pretrained("../model/Qwen3-1.7B-base", torch_dtype=torch.float16).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained("../model/Qwen3-0.6B-base")

# 2. 加载预训练的映射器 (对应 Qwen3 1.7B -> 4B，实际需适配你的模型对)
mapper = Mapper.from_pretrained("Siddharth85/kvbridge-qwen3-1.7b-to-qwen3-4b")

# 3. 小模型预填充长上下文
prompt = "Your long context document here..." * 100
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

with torch.no_grad():
    # 小模型执行 prefill，获取 KV cache
    small_outputs = model_small(prompt_ids, use_cache=True)
    small_cache = small_outputs.past_key_values

# 4. 执行跨模型 KV 传递
# transfer 函数内部会应用线性映射，将小模型的 KV 转换为大模型可用的格式
big_cache = transfer(small_cache, mapper)

# 5. 大模型直接使用传递来的 Cache 进行生成 (跳过 Prefill)
question = "\n\nQuestion: Based on the document above, ..."
question_ids = tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

with torch.no_grad():
    # 大模型只处理问题部分，从传递来的 Cache 开始生成
    generated_ids = model_large.generate(
        question_ids,
        past_key_values=big_cache, # 注入映射后的 KV Cache
        max_new_tokens=50,
        do_sample=False
    )

# 6. 解码输出
answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(answer)