import torch

from main import VotingRule, vote


def test_equal_soft_is_the_mean_of_member_probabilities():
    probabilities = [
        torch.tensor([[0.5, 0.4, 0.1], [0.3, 0.2, 0.5]]),
        torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]]),
        torch.tensor([[0.3, 0.3, 0.4], [0.1, 0.7, 0.2]]),
    ]
    assert torch.equal(vote(probabilities, VotingRule("equal soft")),
                       torch.stack(probabilities).mean(0))
