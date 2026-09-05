import torch
import torch.nn.functional as F
from torch import nn

from optical_adaptor.training.objectives import continuation_chunk, task_loss
from optical_adaptor.training.sampling import epoch_windows, task_permutations


def test_exact_kl_and_metric_ce_detachment():
    torch.manual_seed(1)
    head = nn.Linear(5, 13, bias=False).requires_grad_(False)
    student = torch.randn(9, 5, requires_grad=True)
    teacher = torch.randn(9, 5)
    targets = torch.arange(9)
    loss, metrics = continuation_chunk(student, teacher, targets, head)
    expected = F.kl_div(
        F.log_softmax(head(student), -1), F.softmax(head(teacher), -1), reduction="sum"
    )
    torch.testing.assert_close(loss, expected)
    assert not metrics.requires_grad
    (actual_gradient,) = torch.autograd.grad(loss, student, retain_graph=True)
    (expected_gradient,) = torch.autograd.grad(expected, student)
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_checkpointed_chunks_match_whole_loss_and_gradient():
    torch.manual_seed(2)
    head = nn.Linear(5, 17, bias=False).requires_grad_(False)
    targets = torch.arange(11)
    mask = torch.tensor([True] * 10 + [False])
    for teacher in (None, torch.randn(11, 5)):
        student = torch.randn(11, 5, requires_grad=True)
        chunked = task_loss(student, targets, mask, head=head, teacher=teacher, chunk_size=3)
        whole = task_loss(student, targets, mask, head=head, teacher=teacher, chunk_size=11)
        torch.testing.assert_close(chunked.loss, whole.loss)
        torch.testing.assert_close(chunked.statistics, whole.statistics)
        (first,) = torch.autograd.grad(chunked.loss, student, retain_graph=True)
        (second,) = torch.autograd.grad(whole.loss, student)
        torch.testing.assert_close(first, second)


def test_mixed_tasks_cover_every_record_and_partial_window():
    rows = [
        {"id": i, "visual_ids": list(range(i % 7 + 1)), "continuation_ids": [1] * 128}
        for i in range(10000)
    ]
    assert task_permutations(100, 42, 0) == task_permutations(100, 42, 0)
    assert task_permutations(100, 42, 0)[0] != task_permutations(100, 42, 0)[1]
    ranks = [
        list(epoch_windows(rows, seed=42, epoch=0, global_pairs=32, rank=rank, world_size=2))
        for rank in range(2)
    ]
    assert all(len(windows) == 313 for windows in ranks)
    assert all(windows[-1].global_pairs == 16 for windows in ranks)
    for task in ("continuation", "reconstruction"):
        ids = [
            row["id"] for windows in ranks for window in windows for row in getattr(window, task)
        ]
        assert sorted(ids) == list(range(10000))
    for index in range(313):
        counts = sum(
            len(row["visual_ids"]) + 1 for windows in ranks for row in windows[index].reconstruction
        )
        assert counts == ranks[0][index].reconstruction_tokens
