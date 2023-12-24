import os

import pytest
import torch

from learning_kit.nn.fc_block import FullyConnectedBlock


def test_fully_connected_block_no_activation():
    # 활성화 함수를 사용하지 않는 FullyConnectedBlock을 테스트합니다.
    block = FullyConnectedBlock(in_features=10, out_features=20)
    input_tensor = torch.randn(32, 10)  # 배치 크기 32, 입력 특성 10
    output = block(input_tensor)
    assert output.shape == (32, 20)  # 예상된 출력 형태: (배치 크기 32, 출력 특성 20)


def test_fully_connected_block_with_activation():
    # ReLU 활성화 함수를 사용하는 FullyConnectedBlock을 테스트합니다.
    block = FullyConnectedBlock(in_features=10, out_features=20, activation="ReLU")
    input_tensor = torch.randn(32, 10)  # 배치 크기 32, 입력 특성 10
    output = block(input_tensor)
    assert output.shape == (32, 20)  # 예상된 출력 형태: (배치 크기 32, 출력 특성 20)


def test_fully_connected_block_with_batchnorm():
    # 배치 정규화를 사용하는 FullyConnectedBlock을 테스트합니다.
    block = FullyConnectedBlock(in_features=10, out_features=20, batch_norm=True)
    input_tensor = torch.randn(32, 10)  # 배치 크기 32, 입력 특성 10
    output = block(input_tensor)
    assert output.shape == (32, 20)  # 예상된 출력 형태: (배치 크기 32, 출력 특성 20)


def test_fully_connected_block_with_dropout():
    # 드롭아웃을 사용하는 FullyConnectedBlock을 테스트합니다.
    block = FullyConnectedBlock(in_features=10, out_features=20, dropout_probability=0.2)
    input_tensor = torch.randn(32, 10)  # 배치 크기 32, 입력 특성 10
    output = block(input_tensor)
    assert output.shape == (32, 20)  # 예상된 출력 형태: (배치 크기 32, 출력 특성 20)


# 추가적인 테스트 케이스를 작성할 수 있습니다.

# 예를 들어, 다른 활성화 함수와 함께 배치 정규화 및 드롭아웃을 사용하는 경우를 테스트할 수 있습니다.
def test_fully_connected_block_complex():
    block = FullyConnectedBlock(
        in_features=10,
        out_features=20,
        activation="LeakyReLU",
        activation_kwargs={"negative_slope": 0.1},
        batch_norm=True,
        dropout_probability=0.3
    )
    input_tensor = torch.randn(32, 10)  # 배치 크기 32, 입력 특성 10
    output = block(input_tensor)
    assert output.shape == (32, 20)  # 예상된 출력 형태: (배치 크기 32, 출력 특성 20)


if __name__ == "__main__":
    pytest.main(["-s", "-v", f"{os.path.abspath(__file__)}"])
