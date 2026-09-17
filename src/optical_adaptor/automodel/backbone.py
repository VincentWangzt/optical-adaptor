"""Selected-target interface on AutoModel's native Qwen3.5 implementation."""

from __future__ import annotations

from types import SimpleNamespace

from nemo_automodel._transformers.registry import register_architecture
from nemo_automodel.components.distributed.context_parallel.sharder import (
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from optical_adaptor.automodel.parallel import select_context_targets


class OpticalQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Keep native weights, layers and checkpoint conversion; select before head."""

    cp_mesh = None

    def __init__(self, config, backend=None, **kwargs):
        super().__init__(config, backend=backend, **kwargs)
        if self.mtp is not None:
            raise ValueError("Frozen optical KD does not use a trainable MTP head")
        # Reuse the HF Qwen TP plan consumed by AutoModel's Qwen strategy.
        self.model._tp_plan = Qwen3_5TextConfig.base_model_tp_plan
        self._tp_plan = {"lm_head": "colwise_rep"}

    def forward(self, *, inputs_embeds, attention_mask, position_ids, loss_positions):
        """Decode and project compact targets through the native model.

        Args:
            inputs_embeds: Full embeddings [batch, input_sequence, hidden].
            attention_mask: Right-padding validity [batch, input_sequence].
            position_ids: Full token positions [batch, input_sequence].
            loss_positions: Input prediction positions [batch, targets], -1 padded.

        Returns:
            Logits [batch, targets, vocab], or [batch, padded_targets / CP, vocab]
            under CP. TP uses a vocabulary-sharded DTensor with that global shape.
        """
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
        return SimpleNamespace(logits=self.lm_head(selected))


register_architecture("OpticalQwen3_5ForCausalLM", OpticalQwen3_5ForCausalLM)
