from __future__ import annotations

import json

from .base import ModelBase, MmprojModel, gguf
from .qwen import Qwen2Model


@ModelBase.register_hparams_loader(
    lambda path: (path / "llm_config.json").is_file() and (path / "vit_config.json").is_file()
)
def load_bagel_config(path):
    config = json.loads((path / "llm_config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen2ForCausalLM"]:
        raise ValueError("Unsupported BAGEL language model configuration")
    # The official inference setup enables these outside llm_config.json.
    config.update(architectures=["Bagel"], qk_norm=True, tie_word_embeddings=False)
    return config


@ModelBase.register("Bagel", "BagelForConditionalGeneration")
@ModelBase.example("ByteDance-Seed/BAGEL-7B-MoT")
class BagelModel(Qwen2Model):
    # The understanding expert uses the existing Qwen3 graph: per-head Q/K
    # RMSNorm, NeoX RoPE, SwiGLU, and optional Q/K/V projection biases.
    model_arch = gguf.MODEL_ARCH.QWEN3
    safetensors_prefix = "ema"

    @classmethod
    def filter_tensors(cls, item):
        name, data = item
        if not name.startswith("language_model.") or "_moe_gen" in name:
            return None
        return super().filter_tensors((name.removeprefix("language_model."), data))

    def set_gguf_parameters(self):
        if not self.hparams.get("qk_norm", True):
            raise ValueError("BAGEL understanding requires Q/K normalization")
        if self.hparams.get("use_sliding_window", False) or self.hparams.get("rope_scaling"):
            raise ValueError("BAGEL sliding attention and scaled RoPE are not supported")
        for suffix in ("q_norm.weight", "k_norm.weight"):
            if f"model.layers.0.self_attn.{suffix}" not in self.model_tensors:
                raise ValueError(f"BAGEL understanding checkpoint is missing {suffix}")
        super().set_gguf_parameters()
        self.gguf_writer.add_string("bagel.branch", "understanding")


@ModelBase.register("BagelForConditionalGeneration")
@ModelBase.example("ByteDance-Seed/BAGEL-7B-MoT")
class BagelVisionModel(MmprojModel):
    safetensors_prefix = "ema"

    @classmethod
    def filter_tensors(cls, item):
        return item if item[0].startswith(("vit_model.", "connector.", "vit_pos_embed.")) else None

    def get_vision_config(self):
        config = dict(self.global_config["vit_config"])
        layers = {int(name.split(".")[4]) for name in self.model_tensors
                  if name.startswith("vit_model.vision_model.encoder.layers.")}
        if not layers or layers != set(range(len(layers))):
            raise ValueError("BAGEL vision layers must be contiguous")
        config["num_hidden_layers"] = len(layers)
        return config

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.BAGEL)
        self.gguf_writer.add_vision_attention_layernorm_eps(1e-6)
        self.gguf_writer.add_vision_use_gelu(True)

    def modify_tensors(self, data_torch, name, bid):
        if name.startswith("connector."):
            name = name.replace("connector.fc1", "mm.0").replace("connector.fc2", "mm.1")
            yield name, data_torch
        elif name == "vit_pos_embed.pos_embed":
            yield "mm.position_embd.weight", data_torch
        else:
            name = name.replace("vit_model.vision_model.", "model.vision_model.")
            if name.endswith("embeddings.patch_embedding.weight"):
                data_torch = data_torch.reshape(1152, 14, 14, 3).permute(0, 3, 1, 2).contiguous()
            yield from super().modify_tensors(data_torch, name, bid)
