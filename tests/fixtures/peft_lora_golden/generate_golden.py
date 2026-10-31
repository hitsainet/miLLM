"""Regenerate the PEFT golden fixture beside this file. NOT run by the test suite.

Provenance of the committed files: peft 0.21.2, torch 2.10.0+cpu, transformers 5.15.1, run in a
throwaway venv (peft is not a miLLM dependency). A tiny bfloat16 Llama (non-square MLP, GQA k/v)
with a LoRA on all seven projections, `init_lora_weights=False` and B scaled x4 so the adapter's
effect is far above bfloat16 rounding, `lora_alpha=12, r=4` (scaling 3), dropout 0.1 (must be
ignored at inference). PEFT keeps A and B in float32 (`autocast_adapter_dtype=True`, asserted).
`peft_logits.safetensors` holds the input ids and PEFT's own logits for them.
`base_logits.safetensors` holds the bare base model's logits for the same ids. It was added on
2026-10-16 from the committed base weights on the machine that reproduces the fixture bit for bit
(torch 2.10.0, CPU, a Ryzen 9 8945HS with avx512_bf16). The test uses it to tell "this CPU rounds bfloat16
differently" apart from "the adapter math is wrong".

    python -I generate_golden.py <out_dir>
"""
import json, sys, torch
from pathlib import Path
from transformers import LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(1234)
cfg = LlamaConfig(vocab_size=64, hidden_size=48, intermediate_size=80, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                  tie_word_embeddings=False)
base = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
base_sd = {k: v.detach().clone().contiguous() for k, v in base.state_dict().items()}
save_file(base_sd, str(out / "base_model.safetensors"))
(out / "base_config.json").write_text(json.dumps(cfg.to_dict(), indent=1, default=str))
lcfg = LoraConfig(r=4, lora_alpha=12, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                  lora_dropout=0.1, bias="none", init_lora_weights=False)
pm = get_peft_model(base, lcfg).eval()
for n, p in pm.named_parameters():
    if "lora_" in n:
        assert p.dtype == torch.float32, (n, p.dtype)
        with torch.no_grad(): p.mul_(4.0)
pm.save_pretrained(str(out / "adapter"))
for name in ("adapter_config.json", "adapter_model.safetensors"):
    (out / "adapter" / name).rename(out / name)
ids = torch.tensor([[1, 5, 9, 17, 33, 2, 60, 7, 11, 40, 3, 22]])
with torch.no_grad():
    with pm.disable_adapter():
        save_file({"input_ids": ids, "logits": pm(input_ids=ids).logits.contiguous()},
                  str(out / "base_logits.safetensors"))
    logits = pm(input_ids=ids).logits
save_file({"input_ids": ids, "logits": logits.contiguous()}, str(out / "peft_logits.safetensors"))
print(logits.dtype, logits.shape)
import peft, transformers; print(peft.__version__, torch.__version__, transformers.__version__)
