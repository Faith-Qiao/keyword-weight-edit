import torch
import pytest
from edit_math import contrastive_direction, edited_row, rate, score
from prompts import paired_prompts, TEST


def test_direction_is_normalized():
    a = torch.tensor([[2., 1.], [4., 1.]])
    b = torch.tensor([[1., 1.], [1., 1.]])
    assert torch.allclose(contrastive_direction(a, b), torch.tensor([1., 0.]))


def test_zero_edit_is_noop():
    old = torch.tensor([0.25, 0.75])
    assert torch.equal(edited_row(old, torch.tensor([1., 0.]), 0), old)


def test_one_row_edit():
    old = torch.tensor([0., 2.])
    assert torch.allclose(edited_row(old, torch.tensor([1., 0.]), 3), torch.tensor([3., 2.]))


def test_equal_means_rejected():
    with pytest.raises(ValueError):
        contrastive_direction(torch.zeros(2, 3), torch.zeros(2, 3))


def test_paired_splits_are_disjoint():
    a, b = paired_prompts(TEST, "penguin", "elephant")
    assert len(a) == len(b) == 6
    assert all("penguin" in p for p in a)
    assert all("elephant" in p for p in b)


def test_score_penalizes_false_triggers():
    assert score(1., 0., 0.) > score(1., 1., 0.)
    assert rate([True, False]) == .5
