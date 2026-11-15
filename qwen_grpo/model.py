"""Base model and LoRA wrapping: frozen base model (bf16 by default) with fp32 LoRA adapters on selected projections."""
import hashlib
import re

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

MODEL = "Qwen/Qwen3-0.6B"  # default of the historical runs and research/ scripts; configs may name another Qwen3 model
EOS_IDS = [151645, 151643]  # <|im_end|>, <|endoftext|>: shared by the Qwen3 tokenizer family
DTYPE = torch.bfloat16
RANK, ALPHA = 8, 16
# Named presets: (projection names inside each decoder layer, number of last decoder layers). Resolved per model by
# target_modules(), so "last2-attention" means the last two layers of whichever model is loaded.
PRESETS = {
    "last-o-proj": (("o_proj",), 1),
    "last2-attention": (("q_proj", "v_proj", "o_proj"), 2),
}
# Historical Qwen3-0.6B (28 layers) regexes, kept for research/ scripts and as the reference the presets must reproduce.
TARGET = r"model\.layers\.27\.self_attn\.o_proj"
TARGETS = {
    "last-o-proj": TARGET,
    "last2-attention": r"model\.layers\.(26|27)\.self_attn\.(q|v|o)_proj",
}
EXPECTED_MODULES = {
    "last-o-proj": ["model.layers.27.self_attn.o_proj"],
    "last2-attention": [f"model.layers.{i}.self_attn.{p}_proj" for i in (26, 27) for p in "qvo"],
}


def decoder_layers(model):
    """(name, ModuleList) of the transformer blocks: the unique ModuleList with config.num_hidden_layers entries."""
    n = model.config.num_hidden_layers
    found = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == n]
    if len(found) != 1:
        raise ValueError(f"expected one ModuleList of {n} decoder layers, found {[name for name, _ in found]}")
    return found[0]


def target_modules(model, modules, num_layers):
    """Full names of the Linear layers called `modules` (last name component) in each of the last `num_layers` blocks."""
    prefix, layers = decoder_layers(model)
    if not 1 <= num_layers <= len(layers):
        raise ValueError(f"num_layers {num_layers}: the model has {len(layers)} decoder layers")
    names = []
    for i in range(len(layers) - num_layers, len(layers)):
        hits = [(n.rsplit(".", 1)[-1], f"{prefix}.{i}.{n}") for n, m in layers[i].named_modules()
                if isinstance(m, torch.nn.Linear) and n.rsplit(".", 1)[-1] in modules]
        if missing := set(modules) - {short for short, _ in hits}:
            raise ValueError(f"layer {i} has no Linear module named {sorted(missing)}")
        if len(hits) != len(set(modules)):
            raise ValueError(f"ambiguous module names in layer {i}: {[full for _, full in hits]}")
        names += [full for _, full in hits]
    return names


def matched_modules(model, target):
    """Module names PEFT adapts for a saved `target_modules`: a regex is fullmatched, a list matches a name or a suffix."""
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    if isinstance(target, str):
        return [n for n in names if re.fullmatch(target, n)]
    return [n for n in names if any(n == t or n.endswith("." + t) for t in target)]


def skeleton(name):
    """Module structure of `name` on the meta device: needs config.json only, no weights."""
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name))


def lora_config(target=TARGET, r=RANK, alpha=ALPHA, dropout=0.0):
    return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM", target_modules=target)


def lora_model(dtype, device, target=TARGET, r=RANK, alpha=ALPHA, dropout=0.0, name=MODEL):
    """target: a PEFT target (regex or list of names) or a (modules, num_layers) spec resolved on the loaded model."""
    base = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    base.generation_config = GenerationConfig(eos_token_id=EOS_IDS, pad_token_id=EOS_IDS[1])
    if isinstance(target, tuple):
        target = target_modules(base, *target)
    return get_peft_model(base, lora_config(target, r, alpha, dropout)).eval()


def lora_modules(model):
    """(name without PEFT prefix, module) for every module carrying a LoRA adapter."""
    return [(n.removeprefix("base_model.model."), m) for n, m in model.named_modules() if hasattr(m, "lora_A") and len(m.lora_A)]


def expected_lora_params(modules, r=RANK):
    """sum over adapted Linear layers of r * (in + out): A is [r, in], B is [out, r]."""
    return sum(r * (m.base_layer.in_features + m.base_layer.out_features) for _, m in modules)


def digest(t):
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy()).hexdigest()
