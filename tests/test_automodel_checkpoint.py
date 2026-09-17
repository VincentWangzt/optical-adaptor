"""Exact adapter/Adam continuation through the framework checkpoint engine on CPU."""

import torch
from nemo_automodel.components.checkpoint.config import CheckpointingConfig

from optical_adaptor.adapters import MLPAdapter


def test_adapter_and_adam_checkpoint_preserve_next_update(tmp_path):
    torch.manual_seed(42)
    model = MLPAdapter(3, 5, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    batches = [torch.randn(2, 4, 3) for _ in range(3)]

    def update(module, opt, batch):
        loss = module(batch).square().sum()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss.detach()

    update(model, optimizer, batches[0])
    update(model, optimizer, batches[1])
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path, save_consolidated=False, is_async=False
    ).build(0, 0, 0)
    try:
        checkpointer.save_model([model], str(tmp_path))
        checkpointer.save_optimizer([optimizer], [model], str(tmp_path))
        expected_loss = update(model, optimizer, batches[2])
        restored = MLPAdapter(3, 5, 7)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.002)
        checkpointer.load_model([restored], str(tmp_path / "model"))
        checkpointer.load_optimizer([restored_optimizer], [restored], str(tmp_path))
        actual_loss = update(restored, restored_optimizer, batches[2])
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
        for source, target in zip(
            optimizer.state.values(), restored_optimizer.state.values(), strict=True
        ):
            for name in source:
                torch.testing.assert_close(target[name], source[name], rtol=0, atol=0)
    finally:
        checkpointer.close()
