"""Base model and LoRA wrapping: frozen bf16 Qwen3-0.6B with fp32 LoRA adapters on selected projections."""
import hashlib

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, GenerationConfig

MODEL = "Qwen/Qwen3-0.6B"
EOS_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>
DTYPE = torch.bfloat16
RANK, ALPHA = 8, 16
TARGET = r"model\.layers\.27\.self_attn\.o_proj"
# PEFT re.fullmatch-es a string target against module names, so these regexes select exact modules.
TARGETS = {
    "last-o-proj": TARGET,
    "last2-attention": r"model\.layers\.(26|27)\.self_attn\.(q|v|o)_proj",
}
EXPECTED_MODULES = {
    "last-o-proj": ["model.layers.27.self_attn.o_proj"],
    "last2-attention": [f"model.layers.{i}.self_attn.{p}_proj" for i in (26, 27) for p in "qvo"],
}


def lora_config(target=TARGET, r=RANK, alpha=ALPHA, dropout=0.0):
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules=target)



def lora_model(dtype, device, target=TARGET, r=RANK, alpha=ALPHA, dropout=0.0, name=MODEL):
    base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
    return get_peft_model(base, lora_config(target, r, alpha, dropout)).eval()



def lora_modules(model):
    """(name without PEFT prefix, module) for every module carrying a LoRA adapter."""
    return [(n.removeprefix("base_model.model."), m) for n, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)]



def expected_lora_params(modules, r=RANK):
    """sum over adapted Linear layers of r * (in + out): A is [r, in], B is [out, r]."""
    return sum(r * (m.base_layer.in_features + m.base_layer.out_features) for _, m in modules)



def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()

