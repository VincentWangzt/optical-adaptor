"""Selected projection and unchunked KD must preserve values and input gradients."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from nemo_automodel.components.loss.kd_loss import KDLoss

from optical_adaptor.automodel.backbone import OpticalQwen3_5ForCausalLM


def test_selected_logits_match_dense_projection_and_gradients():
    torch.manual_seed(17)
    head = torch.nn.Linear(7, 19, bias=False).requires_grad_(False)
    embedding = torch.nn.Embedding(40, 7).requires_grad_(False)
    positions = torch.tensor([[1, 4], [0, 3]])
    ids = torch.randint(0, 40, (2, 6))
    labels = torch.randint(0, 19, (2, 2))

    class Decoder(torch.nn.Module):
        def forward(self, **kwargs):
            # Causal dependence provides a gradient path from image embeddings.
            return SimpleNamespace(last_hidden_state=kwargs["inputs_embeds"].cumsum(1))

    language = OpticalQwen3_5ForCausalLM.__new__(OpticalQwen3_5ForCausalLM)
    torch.nn.Module.__init__(language)
    language.model = Decoder()
    language.model.embed_tokens = embedding
    language.lm_head = head
    hidden = torch.randn(2, 6, 7, requires_grad=True)
    actual = language(
        inputs_embeds=hidden,
        attention_mask=torch.ones_like(ids),
        position_ids=torch.arange(6)[None],
        loss_positions=positions,
    ).logits
    dense = head(hidden.cumsum(1)).gather(1, positions[..., None].expand(-1, -1, 19))
    torch.testing.assert_close(actual, dense)
    teacher = torch.randn_like(actual)
    kd = KDLoss(temperature=1.7, fp32_upcast=True, chunk_size=0)
    loss = kd(actual, teacher, labels, num_batch_labels=4)
    expected = (
        F.kl_div((dense / 1.7).log_softmax(-1), (teacher / 1.7).softmax(-1), reduction="sum")
        * 1.7**2
        / 4
    )
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, hidden, retain_graph=True)[0],
        torch.autograd.grad(expected, hidden)[0],
    )
    assert head.weight.grad is None


def test_unequal_target_counts_use_global_denominator():
    torch.manual_seed(29)
    student = torch.randn(9, 13, requires_grad=True)
    teacher, labels = torch.randn(9, 13), torch.randint(0, 13, (9,))
    kd = KDLoss(chunk_size=0)
    whole = kd(student, teacher, labels, num_batch_labels=9)
    parts = kd(student[:2], teacher[:2], labels[:2], num_batch_labels=9) + kd(
        student[2:], teacher[2:], labels[2:], num_batch_labels=9
    )
    torch.testing.assert_close(whole, parts)
    torch.testing.assert_close(
        torch.autograd.grad(whole, student, retain_graph=True)[0],
        torch.autograd.grad(parts, student)[0],
    )


def test_temperature_one_bfloat16_kd_matches_explicit_fp32_reference():
    torch.manual_seed(31)
    student = torch.randn(7, 23, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.randn(7, 23, dtype=torch.bfloat16)
    labels = torch.tensor([3, -100, 7, 4, -100, 1, 9])
    valid = labels != -100

    actual = KDLoss(temperature=1.0, fp32_upcast=True, chunk_size=0)(
        student,
        teacher,
        labels,
        num_batch_labels=5,
    )
    teacher_logprob = F.log_softmax(teacher[valid], dim=-1, dtype=torch.float32)
    student_logprob = F.log_softmax(student[valid], dim=-1, dtype=torch.float32)
    expected = F.kl_div(
        student_logprob,
        teacher_logprob,
        reduction="sum",
        log_target=True,
    ) / 5

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, student, retain_graph=True)[0],
        torch.autograd.grad(expected, student)[0],
    )
