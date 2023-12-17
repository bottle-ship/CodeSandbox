import pytest
import torch

from activation import ActivationFunction  # 여기서 'your_module'은 ActivationFunction 클래스가 정의된 모듈을 나타냅니다.


def test_activation_function_valid():
    # 유효한 활성화 함수 이름으로 클래스를 인스턴스화하고 예상된 형태의 출력을 확인합니다.
    activation_module = ActivationFunction('ReLU')
    input_tensor = torch.randn(1, 10)
    output = activation_module(input_tensor)
    assert output.shape == input_tensor.shape


def test_activation_function_invalid():
    # 잘못된 활성화 함수 이름을 제공하여 ValueError가 발생하는지 확인합니다.
    with pytest.raises(ValueError):
        activation_module = ActivationFunction('InvalidFunctionName')


# 기타 추가적인 테스트 케이스를 작성할 수 있습니다.

# 예를 들어, 다른 활성화 함수를 사용하여 테스트할 수 있습니다.
def test_activation_function_sigmoid():
    activation_module = ActivationFunction('Sigmoid')
    input_tensor = torch.randn(1, 10)
    output = activation_module(input_tensor)
    assert output.shape == input_tensor.shape
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
