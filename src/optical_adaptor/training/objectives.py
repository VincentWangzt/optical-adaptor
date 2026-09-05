from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

# Additive sufficient statistics; only convert to means after distributed reduction.
STAT_NAMES = ("objective", "student_ce", "teacher_ce", "agreement", "correct", "tokens")


@dataclass
class LossResult:
    loss: torch.Tensor
    statistics: torch.Tensor


def continuation_chunk(
    student: torch.Tensor, teacher: torch.Tensor, targets: torch.Tensor, head: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    student_logits = head(student).float()
    student_logp = F.log_softmax(student_logits, dim=-1)
    with torch.no_grad():
        teacher_logits = head(teacher).float()
        teacher_logp = F.log_softmax(teacher_logits, dim=-1)
        teacher_p = teacher_logp.exp()
    loss = (teacher_p * (teacher_logp - student_logp)).sum()
    with torch.no_grad():
        indices = targets[:, None]
        student_ce = -student_logp.gather(1, indices).sum()
        teacher_ce = -teacher_logp.gather(1, indices).sum()
        prediction = student_logits.argmax(-1)
        agreement = (prediction == teacher_logits.argmax(-1)).sum()
        correct = (prediction == targets).sum()
        stats = torch.stack(
            [
                loss.detach(),
                student_ce,
                teacher_ce,
                agreement,
                correct,
                targets.new_tensor(targets.numel()),
            ]
        ).double()
    return loss, stats


def reconstruction_chunk(
    student: torch.Tensor, targets: torch.Tensor, text_mask: torch.Tensor, head: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = head(student).float()
    ce = F.cross_entropy(logits, targets, reduction="none")
    loss = ce.sum()
    with torch.no_grad():
        text_ce = ce[text_mask].sum()
        correct = ((logits.argmax(-1) == targets) & text_mask).sum()
        zero = loss.new_zeros(())
        stats = torch.stack([loss.detach(), text_ce, zero, zero, correct, text_mask.sum()]).double()
    return loss, stats


def task_loss(
    student: torch.Tensor,
    targets: torch.Tensor,
    text_mask: torch.Tensor,
    *,
    head: nn.Module,
    teacher: torch.Tensor | None,
    chunk_size: int,
) -> LossResult:
    losses, statistics = [], []
    if teacher is not None:
        teacher = teacher.reshape_as(student)
    for start in range(0, len(targets), chunk_size):
        end = start + chunk_size
        if teacher is not None:
            fn, args = (
                continuation_chunk,
                (student[start:end], teacher[start:end], targets[start:end]),
            )
        else:
            fn, args = (
                reconstruction_chunk,
                (student[start:end], targets[start:end], text_mask[start:end]),
            )
        if torch.is_grad_enabled():
            loss, stats = checkpoint(fn, *args, head, use_reentrant=False)
        else:
            loss, stats = fn(*args, head)
        losses.append(loss)
        statistics.append(stats)
    return LossResult(torch.stack(losses).sum(), torch.stack(statistics).sum(0))


def mean_metrics(statistics: torch.Tensor, *, continuation: bool) -> dict[str, float]:
    values = dict(zip(STAT_NAMES, statistics.cpu().tolist(), strict=True))
    count = values["tokens"]
    if count <= 0:
        raise ValueError("cannot report metrics without target tokens")
    ce = values["student_ce"] / count
    metrics = {
        "ce": ce,
        "ppl": float(torch.exp(torch.tensor(ce, dtype=torch.float64))),
        "token_accuracy": values["correct"] / count,
        "tokens": count,
    }
    if continuation:
        teacher_ce = values["teacher_ce"] / count
        metrics.update(
            kl=values["objective"] / count,
            teacher_ce=teacher_ce,
            teacher_ppl=float(torch.exp(torch.tensor(teacher_ce, dtype=torch.float64))),
            top1_agreement=values["agreement"] / count,
        )
    return metrics
