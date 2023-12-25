import os

import pytest
import torch
import torch.nn as nn

from learning_kit.nn.losses.combine import CombineLoss


def test_combine_loss():
    loss_module_1 = nn.L1Loss()
    loss_module_2 = nn.MSELoss()

    combined_loss = CombineLoss([loss_module_1, loss_module_2])

    y_pred = (torch.tensor([1.0]), torch.tensor([2.0]))
    y_true = (torch.tensor([0.5]), torch.tensor([2.5]))
    total_loss = combined_loss(y_pred, y_true)

    # Ensure the output is a torch.Tensor
    assert isinstance(total_loss, torch.Tensor)

    # Add more specific assertions based on your requirements
    # For example, you could test the shape or specific values of the total_loss tensor
    # assert total_loss.shape == (1,)  # Adjust this based on your expected output shape

    # You can also test with different weights or loss functions to cover more scenarios
    # For instance, change the weights and check if the output changes accordingly
    # Update weights for the combined loss and check the output
    weights = torch.tensor([0.5, 0.5])
    combined_loss.weight = weights
    total_loss_weights = combined_loss(y_pred, y_true)

    # Ensure the output shape remains the same
    # assert total_loss_weights.shape == (1,)  # Adjust this based on your expected output shape

    # Assert that the total_loss and total_loss_weights are different with different weights
    assert not torch.allclose(total_loss, total_loss_weights)

    # Add more specific assertions based on your requirements


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
