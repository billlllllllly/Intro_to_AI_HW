import pytest
import torch

from main import ClsMeanHead, MeanPoolHead


@pytest.mark.parametrize("head_type", [MeanPoolHead, ClsMeanHead])
def test_pooling_heads_ignore_extra_padding(head_type):
    head = head_type(hidden_size=4).eval()
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    padded_hidden = torch.cat([hidden, torch.full((2, 2, 4), 100.0)], dim=1)
    padded_mask = torch.cat([mask, torch.zeros((2, 2), dtype=mask.dtype)], dim=1)

    logits = head(hidden, mask)
    assert logits.shape == (2, 3)
    torch.testing.assert_close(logits, head(padded_hidden, padded_mask))
