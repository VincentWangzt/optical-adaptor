from __future__ import annotations

import torch
import torch.nn.functional as F
from nemo_automodel import NeMoAutoModelForCausalLM
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoTokenizer

from optical_adaptor.automodel.processing import PairedTokens


class FrozenBackbone:
    """One frozen text model serves both serial passes on each data-parallel rank."""

    def __init__(self, config: dict, device: torch.device, output_dim: int):
        model_id, revision = config["model_id"], config["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        full_config = AutoConfig.from_pretrained(model_id, revision=revision)
        key = config["text_config_key"]
        text_config = getattr(full_config, key) if key else full_config
        self.model = NeMoAutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            config=text_config,
            dtype=torch.bfloat16,
            attn_implementation=config["attn_implementation"],
            force_hf=True,
            use_liger_kernel=False,
            use_sdpa_patching=False,
        ).to(device)
        self.model.requires_grad_(False).eval()
        self.model.config.use_cache = False
        self.device = device
        if config["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if self.model.get_input_embeddings().weight.shape[1] != output_dim:
            raise ValueError("Language-model embedding width does not match the adapter output")

    def mode(self, training: bool):
        # HF checks .training to activate checkpointing, even for frozen weights.
        self.model.train(training)
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    def inputs(self, pair: PairedTokens, adapted: torch.Tensor) -> torch.Tensor:
        ids = torch.tensor(pair.student_ids, device=self.device)
        embeddings = self.model.get_input_embeddings()(ids)
        positions = torch.tensor(pair.image_positions, device=self.device).flatten()
        return embeddings.index_copy(0, positions, adapted.flatten(0, 1).to(embeddings.dtype))

    def hidden(self, inputs: list[torch.Tensor], positions: list[list[int]]) -> torch.Tensor:
        lengths = torch.tensor([len(item) for item in inputs], device=self.device)
        padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        attention = torch.arange(padded.shape[1], device=self.device)[None] < lengths[:, None]
        position_ids = (attention.long().cumsum(-1) - 1).clamp_min(0)
        hidden = self.model.base_model(
            inputs_embeds=padded,
            attention_mask=attention,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        return torch.cat(
            [
                hidden[index, torch.tensor(selected, device=self.device)]
                for index, selected in enumerate(positions)
            ]
        )

    @torch.no_grad()
    def teacher_hidden(self, pairs: list[PairedTokens]) -> torch.Tensor:
        self.mode(False)
        embed = self.model.get_input_embeddings()
        inputs = [embed(torch.tensor(pair.teacher_ids, device=self.device)) for pair in pairs]
        return self.hidden(inputs, [pair.teacher_positions for pair in pairs])

    def student_hidden(self, pairs: list[PairedTokens], adapted: torch.Tensor) -> torch.Tensor:
        self.mode(torch.is_grad_enabled())
        inputs, offset = [], 0
        for pair in pairs:
            count = len(pair.image_positions)
            inputs.append(self.inputs(pair, adapted[offset : offset + count]))
            offset += count
        if offset != len(adapted):
            raise ValueError("Number of adapted images differs from the compiled visual slots")
        return self.hidden(inputs, [pair.student_positions for pair in pairs])

    @torch.no_grad()
    def generate(self, pair: PairedTokens, adapted: torch.Tensor, max_new_tokens: int) -> str:
        self.mode(False)
        inputs = self.inputs(pair, adapted)[: pair.generation_prefix_length].unsqueeze(0)
        ids = self.model.generate(
            inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
        )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


def aligned_losses(
    student: torch.Tensor,
    teacher: torch.Tensor,
    targets: torch.Tensor,
    head: nn.Module,
    kd_loss: nn.Module,
    chunk_size: int,
) -> torch.Tensor:
    """Return sums [student CE, forward KL, teacher CE, agreement, correct, tokens].

    Project only aligned hidden states. Checkpoint each vocabulary projection so
    backward never retains a full [trajectory_length, vocabulary] probability array.
    The KD primitive is AutoModel's exact full-vocabulary KDLoss.
    """
    if student.shape != teacher.shape or student.shape[0] != len(targets):
        raise ValueError("Hidden states and supervised target identities must align")

    def chunk_loss(student_chunk, teacher_chunk, labels):
        s_logits = head(student_chunk).float()
        with torch.no_grad():
            t_logits = head(teacher_chunk).float()
        ce = F.cross_entropy(s_logits, labels, reduction="sum")
        kl = kd_loss(s_logits, t_logits, labels, num_batch_labels=1)
        with torch.no_grad():
            t_ce = F.cross_entropy(t_logits, labels, reduction="sum")
            prediction = s_logits.argmax(-1)
            agreement = (prediction == t_logits.argmax(-1)).sum()
            correct = (prediction == labels).sum()
        return torch.stack((ce, kl, t_ce, agreement, correct, labels.new_tensor(len(labels))))

    losses = []
    for start in range(0, len(targets), chunk_size):
        args = (
            student[start : start + chunk_size],
            teacher[start : start + chunk_size],
            targets[start : start + chunk_size],
        )
        if torch.is_grad_enabled():
            losses.append(checkpoint(chunk_loss, *args, use_reentrant=False))
        else:
            losses.append(chunk_loss(*args))
    return torch.stack(losses).sum(0)
