"""Selected-target interface on AutoModel's native Qwen3.5 implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from nemo_automodel._transformers.registry import register_architecture
from nemo_automodel.components.distributed.context_parallel.sharder import (
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM
from nemo_automodel.components.models.qwen3_5.state_dict_adapter import Qwen3_5DenseStateDictAdapter
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from optical_adaptor.automodel.parallel import select_context_targets


class OpticalTextCheckpointAdapter(Qwen3_5DenseStateDictAdapter):
    """Read the text tower of a Qwen VLM using native Qwen dtype/key conversion."""

    def __init__(self, tied_embeddings: bool):
        super().__init__()
        self.tied_embeddings = tied_embeddings

    def to_hf(self, state_dict: dict, **kwargs) -> dict:
        """Expose native parameter destinations under the VLM checkpoint names.

        Args:
            state_dict: Native parameter tensors of arbitrary shape; axes are
                unchanged and output values alias the original destinations.
            **kwargs: Shared checkpoint conversion options.

        Returns:
            The same tensors with text-tower prefixes and a single tied embedding.
        """
        converted = super().to_hf(state_dict, **kwargs)
        return {
            ("model.language_model." + key.removeprefix("model."))
            if key.startswith("model.")
            else key: value
            for key, value in converted.items()
            if key != "lm_head.weight" or not self.tied_embeddings
        }

    def from_hf(self, hf_state_dict: dict, **kwargs) -> dict:
        """Restore native text names; ignore the unused Qwen vision tower.

        Args:
            hf_state_dict: Checkpoint parameter tensors of arbitrary shape.
                Only names change; the existing Qwen adapter handles FP32 SSMs.
            **kwargs: Shared checkpoint conversion options.

        Returns:
            Native text parameters, with a shared alias for a tied output head.
        """
        text = {
            "model." + key.removeprefix("model.language_model."): value
            for key, value in hf_state_dict.items()
            if key.startswith("model.language_model.")
        }
        if self.tied_embeddings and "model.embed_tokens.weight" in text:
            text["lm_head.weight"] = text["model.embed_tokens.weight"]
        elif "lm_head.weight" in hf_state_dict:
            text["lm_head.weight"] = hf_state_dict["lm_head.weight"]
        return super().from_hf(text, **kwargs)


class OpticalQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Keep native weights, layers and checkpoint conversion; select before head."""

    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        supports_tp: bool = True
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = False

    def __init__(self, config, backend=None, **kwargs):
        # Checkpoints may advertise an auxiliary MTP head. Optical KD supervises
        # the ordinary next-token head, as in the original HF text-only route.
        super().__init__(config, backend=backend, num_nextn_predict_layers=0, **kwargs)
        # Reuse the HF Qwen TP plan consumed by AutoModel's Qwen strategy.
        self.model._tp_plan = Qwen3_5TextConfig.base_model_tp_plan
        self._tp_plan = {"lm_head": "colwise_rep"}
        self.state_dict_adapter = OpticalTextCheckpointAdapter(config.tie_word_embeddings)

    def forward(
        self,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        image_positions: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        """Decode and project compact targets through the native model.

        Args:
            inputs_embeds: Full embeddings [batch, input_sequence, hidden].
                Optional alternative to input_ids, primarily for parity checks.
            input_ids: Token IDs [batch, input_sequence], embedded inside the
                FSDP root so tied embedding/head weights are materialized.
            image_embeds: Optional visual embeddings [images, image_tokens, hidden].
            image_positions: Optional slots [images, image_tokens, 2], with final
                axis storing batch and input-position coordinates.
            attention_mask: Right-padding validity [batch, input_sequence].
            position_ids: Full token positions [batch, input_sequence].
            loss_positions: Input prediction positions [batch, targets], -1 padded.

        Returns:
            Logits [batch, targets, vocab], or [batch, padded_targets / CP, vocab]
            under CP. TP uses a vocabulary-sharded DTensor with that global shape.
        """
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if image_embeds is not None:
            positions = image_positions.flatten(0, 1)
            flat = positions[:, 0] * inputs_embeds.shape[1] + positions[:, 1]
            inputs_embeds = (
                inputs_embeds.flatten(0, 1)
                .index_copy(0, flat, image_embeds.to(inputs_embeds.dtype).flatten(0, 1))
                .view_as(inputs_embeds)
            )
        if self.cp_mesh is not None:
            inputs_embeds = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds)[0]
            position_ids = shard_sequence_for_cp_round_robin(self.cp_mesh, position_ids)[0]
        # OpticalProcessor right-pads independent rows. Causal valid prefixes
        # cannot attend to trailing padding; keeping rows dense also avoids
        # treating padded microbatches as concatenated packed GDN sequences.
        attention_mask = None
        hidden = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        selected = select_context_targets(hidden, loss_positions, self.cp_mesh)
        return CausalLMOutput(logits=self.lm_head(selected))


register_architecture("OpticalQwen3_5ForCausalLM", OpticalQwen3_5ForCausalLM)
