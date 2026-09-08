from __future__ import annotations

import torch

from .base import ModelBase, gguf
from .qwen import Qwen2Model


@ModelBase.register("NEOChatModel")
class SenseNovaU1Model(Qwen2Model):
    model_arch = gguf.MODEL_ARCH.SENSENOVA_U1

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        if not name.startswith("language_model.") or "_mot_gen" in name:
            return None
        return super().filter_tensors(item)

    def set_gguf_parameters(self):
        if self.hparams.get("num_experts", 0):
            raise ValueError("SenseNova U1 conversion currently supports the dense understanding branch")
        if self.rope_parameters.get("rope_type", "default") not in ("default", None):
            raise ValueError("SenseNova U1 scaled RoPE is not supported")
        if self.hparams.get("attention_bias", False):
            raise ValueError("SenseNova U1 attention bias is not supported")
        if self.hparams.get("use_sliding_window", False):
            raise ValueError("SenseNova U1 sliding attention is not supported")
        super().set_gguf_parameters()
        self.gguf_writer.add_float32(
            "sensenova_u1.rope.freq_base_spatial", self.hparams["rope_theta_hw"])

    def get_tensors(self):
        for name, gen in self.model_tensors.items():
            if name.endswith((".q_norm_hw.weight", ".k_norm_hw.weight")):
                continue
            data = gen()
            if name.endswith((".q_norm.weight", ".k_norm.weight")):
                spatial = self.model_tensors[name.replace("_norm.weight", "_norm_hw.weight")]()
                # Keep one weight vector; the graph normalizes its two halves independently.
                data = torch.cat((data, spatial), dim=0)
            yield name, data
