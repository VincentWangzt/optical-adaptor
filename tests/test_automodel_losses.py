import torch
import torch.nn.functional as F
from nemo_automodel.components.loss.kd_loss import KDLoss

from optical_adaptor.automodel.model import aligned_losses


def test_chunked_ce_kl_matches_dense_values_and_gradients():
    torch.manual_seed(17)
    head = torch.nn.Linear(7, 19, bias=False).requires_grad_(False)
    student = torch.randn(11, 7, requires_grad=True)
    teacher = torch.randn(11, 7)
    labels = torch.randint(0, 19, (11,))
    temperature, ratio = 1.7, 0.4
    kd = KDLoss(temperature=temperature, fp32_upcast=True, chunk_size=3)
    sums = aligned_losses(student, teacher, labels, head, kd, chunk_size=4)
    loss = ((1 - ratio) * sums[0] + ratio * sums[1]) / len(labels)
    actual_grad = torch.autograd.grad(loss, student)[0]

    independent = student.detach().clone().requires_grad_()
    s, t = head(independent), head(teacher)
    ce = F.cross_entropy(s, labels)
    expected_kl = (
        F.kl_div(
            (s / temperature).log_softmax(-1), (t / temperature).softmax(-1), reduction="batchmean"
        )
        * temperature**2
    )
    expected_grad = torch.autograd.grad((1 - ratio) * ce + ratio * expected_kl, independent)[0]
    torch.testing.assert_close(sums[0] / len(labels), ce)
    torch.testing.assert_close(sums[1] / len(labels), expected_kl)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert head.weight.grad is None and not teacher.requires_grad


def test_unequal_microbatches_use_global_token_denominator():
    torch.manual_seed(29)
    head = torch.nn.Linear(5, 13, bias=False).requires_grad_(False)
    teacher = torch.randn(9, 5)
    labels = torch.randint(0, 13, (9,))
    kd = KDLoss(temperature=1, fp32_upcast=True, chunk_size=2)
    whole = torch.randn(9, 5, requires_grad=True)
    split = whole.detach().clone().requires_grad_()
    sums = aligned_losses(whole, teacher, labels, head, kd, 4)
    full_loss = (sums[0] + sums[1]) / 18
    full_grad = torch.autograd.grad(full_loss, whole)[0]
    first = aligned_losses(split[:2], teacher[:2], labels[:2], head, kd, 3)
    second = aligned_losses(split[2:], teacher[2:], labels[2:], head, kd, 3)
    split_loss = (first[0] + first[1] + second[0] + second[1]) / 18
    split_grad = torch.autograd.grad(split_loss, split)[0]
    torch.testing.assert_close(split_grad, full_grad)
