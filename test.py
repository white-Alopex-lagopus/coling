import inspect
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
print(inspect.getsource(Qwen3Attention.forward))