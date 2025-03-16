from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from functools import wraps
import math


@dataclass(slots=True)
class Layer:
    weights: list[list[float]]
    
    def inner_product(self, weights: list[float], inputs: list[float]) -> float:
        return sum(weight * input_value for weight, input_value in zip(weights, inputs))
    
    def forward(self, input_values: list[float]) -> list[float]:
        output_values = []
        for neuron_weights in self.weights:
            result = self.inner_product(neuron_weights, input_values)
            output_values.append(result)
        return output_values

@dataclass(slots=True)
class Network:
    layers: list[Layer]
    bias: float
    
    def forward(self, input_values: list[float]) -> list[float]:
        current_values = input_values + [self.bias]
        for layer in self.layers:
            current_values = layer.forward(current_values)
            if layer != self.layers[-1]:
                current_values = current_values + [self.bias]
        return current_values

@dataclass(slots=True)
class ActivationLayer(Layer):
    bias_weights: list[float]
    activation_func: Callable
    
    def forward(self, input_values: list[float]) -> list[float]:
        output_values = []
        for neuron_weights, bias_weight in zip(self.weights, self.bias_weights):
            result = self.inner_product(neuron_weights, input_values)
            result += bias_weight
            result = self.activation_func(result)
            output_values.append(result)
        return output_values

@dataclass(slots=True)
class MultiOutputLayer(ActivationLayer):
    def forward(self, input_values: list[float]) -> list[float]:
        pre_activation_values = []
        for neuron_weights, bias_weight in zip(self.weights, self.bias_weights):
            result = self.inner_product(neuron_weights, input_values)
            result += bias_weight
            pre_activation_values.append(result)
        return self.activation_func(pre_activation_values)

class Activation:
    @staticmethod
    def relu(value: float) -> float:
        return max(0, value)
    
    @staticmethod
    def linear(value: float) -> float:
        return value
    
    @staticmethod
    def sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))
    
    @staticmethod
    def softmax(values: list[float]) -> list[float]:
        max_value = max(values)
        exp_values = [math.exp(value - max_value) for value in values]
        total = sum(exp_values)
        return [value / total for value in exp_values]

class Loss:
    @staticmethod
    def mse(expected: list[float], actual: list[float]) -> float:
        errors = [(exp - act) ** 2 for exp, act in zip(expected, actual)]
        return sum(errors) / len(expected)
    
    @staticmethod
    def binary_cross_entropy(expected: list[float], actual: list[float]) -> float:
        return -sum(exp * math.log(act) + (1 - exp) * math.log(1 - act) 
                   for exp, act in zip(expected, actual))
    
    @staticmethod
    def categorical_cross_entropy(expected: list[float], actual: list[float]) -> float:
        epsilon = 1e-15
        return -sum(exp * math.log(max(min(act, 1 - epsilon), epsilon))
                   for exp, act in zip(expected, actual) if exp > 0)

@dataclass(slots=True)
class ActivationNetwork(Network):
    def forward(self, input_values: list[float]) -> list[float]:
        current_values = input_values
        for layer in self.layers:
            current_values = layer.forward(current_values)
        return current_values

# 測試數據
class TestData:
    REGRESSION = {
        "weights": [
            ([[0.5, 0.2], [0.6, -0.6]], [0.3, 0.25]),
            ([[0.8, -0.5], [0.4, 0.5]], [0.6, -0.25])
        ],
        "test_cases": [
            ([1.5, 0.5], [0.8, 1.0]),
            ([0, 1], [0.5, 0.5])
        ]
    }
    
    BINARY_CLASSIFICATION = {
        "weights": [
            ([[0.5, 0.2], [0.6, -0.6]], [0.3, 0.25]),
            ([[0.8, 0.4]], [-0.5])
        ],
        "test_cases": [
            ([0.75, 1.25], [1]),
            ([-1, 0.5], [0])
        ]
    }
    
    MULTI_LABEL = {
        "weights": [
            ([[0.5, 0.2], [0.6, -0.6]], [0.3, 0.25]),
            ([[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]], [0.6, 0.5, -0.5])
        ],
        "test_cases": [
            ([1.5, 0.5], [1, 0, 1]),
            ([0, 1], [1, 1, 0])
        ]
    }
    
    MULTI_CLASS = {
        "weights": [
            ([[0.5, 0.2], [0.6, -0.6]], [0.3, 0.25]),
            ([[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]], [0.6, 0.5, -0.5])
        ],
        "test_cases": [
            ([1.5, 0.5], [1, 0, 0]),
            ([0, 1], [0, 0, 1])
        ]
    }

def neural_network_task(task_name: str, loss_func: Callable) -> Callable:
    def decorator(task_func: Callable) -> Callable:
        @wraps(task_func)
        def wrapper():
            print(f"------ {task_name} Tasks -------")
            network, test_cases = task_func()
            for inputs, expected in test_cases:
                outputs = network.forward(inputs)
                print(f"Output: {outputs}")
                print(f"Total Loss: {loss_func(expected, outputs)}\n")
        return wrapper
    return decorator

@neural_network_task("Regression", Loss.mse)
def execute_regression_task():
    data = TestData.REGRESSION
    network = ActivationNetwork(
        layers=[
            ActivationLayer(
                weights=data["weights"][0][0],
                bias_weights=data["weights"][0][1],
                activation_func=Activation.relu
            ),
            ActivationLayer(
                weights=data["weights"][1][0],
                bias_weights=data["weights"][1][1],
                activation_func=Activation.linear
            )
        ],
        bias=1.0
    )
    return network, data["test_cases"]

@neural_network_task("Binary Classification", Loss.binary_cross_entropy)
def execute_binary_classification_task():
    data = TestData.BINARY_CLASSIFICATION
    network = ActivationNetwork(
        layers=[
            ActivationLayer(
                weights=data["weights"][0][0],
                bias_weights=data["weights"][0][1],
                activation_func=Activation.relu
            ),
            ActivationLayer(
                weights=data["weights"][1][0],
                bias_weights=data["weights"][1][1],
                activation_func=Activation.sigmoid
            )
        ],
        bias=1.0
    )
    return network, data["test_cases"]

@neural_network_task("Multi-Label Classification", Loss.binary_cross_entropy)
def execute_multi_label_classification_task():
    data = TestData.MULTI_LABEL
    network = ActivationNetwork(
        layers=[
            ActivationLayer(
                weights=data["weights"][0][0],
                bias_weights=data["weights"][0][1],
                activation_func=Activation.relu
            ),
            ActivationLayer(
                weights=data["weights"][1][0],
                bias_weights=data["weights"][1][1],
                activation_func=Activation.sigmoid
            )
        ],
        bias=1.0
    )
    return network, data["test_cases"]

@neural_network_task("Multi-Class Classification", Loss.categorical_cross_entropy)
def execute_multi_class_classification_task():
    data = TestData.MULTI_CLASS
    network = ActivationNetwork(
        layers=[
            ActivationLayer(
                weights=data["weights"][0][0],
                bias_weights=data["weights"][0][1],
                activation_func=Activation.relu
            ),
            MultiOutputLayer(
                weights=data["weights"][1][0],
                bias_weights=data["weights"][1][1],
                activation_func=Activation.softmax
            )
        ],
        bias=1.0
    )
    return network, data["test_cases"]

if __name__ == '__main__':
    execute_regression_task()
    execute_binary_classification_task()
    execute_multi_label_classification_task()
    execute_multi_class_classification_task()