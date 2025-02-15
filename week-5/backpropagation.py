from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable
from functools import wraps
from abc import ABC, abstractmethod
import math

@dataclass(slots=True)
class Layer:
    weights: list[list[float]]
    bias_weights: list[float]
    cached_inputs: list[float] = field(default=None, init=False)
    cached_pre_activation: list[float] = field(default=None, init=False)
    cached_outputs: list[float] = field(default=None, init=False)
    weight_gradients: list[list[float]] = field(default=None, init=False)
    bias_gradients: list[float] = field(default=None, init=False)
    
    def __post_init__(self):
        self.weight_gradients = [[0.0] * len(self.weights[0]) for _ in range(len(self.weights))]
        self.bias_gradients = [0.0] * len(self.bias_weights)
    
    def inner_product(self, weights: list[float], inputs: list[float]) -> float:
        return sum(weight * input_value for weight, input_value in zip(weights, inputs))
    
    def forward(self, input_values: list[float], activation_func: Callable) -> list[float]:
        self.cached_inputs = input_values.copy()
        pre_activation = []
        for neuron_weights, bias_weight in zip(self.weights, self.bias_weights):
            result = self.inner_product(neuron_weights, input_values)
            result += bias_weight
            pre_activation.append(result)
        self.cached_pre_activation = pre_activation
        outputs = [activation_func(value) for value in pre_activation]
        self.cached_outputs = outputs
        return outputs
    
    def backward(self, next_layer_gradients: list[float], activation_derivative: Callable) -> list[float]:
        local_gradients = []
        for i, pre_act in enumerate(self.cached_pre_activation):
            act_derivative = activation_derivative(pre_act)
            local_gradient = next_layer_gradients[i] * act_derivative
            local_gradients.append(local_gradient)
        
        previous_layer_gradients = [0.0] * len(self.cached_inputs)
        for i in range(len(self.cached_inputs)):
            gradient_sum = 0.0
            for j, local_grad in enumerate(local_gradients):
                gradient_sum += local_grad * self.weights[j][i]
            previous_layer_gradients[i] = gradient_sum
        
        for i, local_grad in enumerate(local_gradients):
            for j, input_value in enumerate(self.cached_inputs):
                self.weight_gradients[i][j] = local_grad * input_value
            self.bias_gradients[i] = local_grad
        
        return previous_layer_gradients

    def zero_grad(self, learning_rate: float) -> None:
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= learning_rate * self.weight_gradients[i][j]
        
        for i in range(len(self.bias_weights)):
            self.bias_weights[i] -= learning_rate * self.bias_gradients[i]

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
    def relu_derivative(value: float) -> float:
        return 1.0 if value > 0 else 0.0
    
    @staticmethod
    def linear_derivative(value: float) -> float:
        return 1.0
    
    @staticmethod
    def sigmoid_derivative(value: float) -> float:
        sig = Activation.sigmoid(value)
        return sig * (1 - sig)

@dataclass(slots=True)
class Loss(ABC):
    @abstractmethod
    def calculate_total_loss(self, outputs: list[float], expects: list[float]) -> float:
        pass
    
    @abstractmethod
    def calculate_gradients(self, outputs: list[float], expects: list[float]) -> list[float]:
        pass

@dataclass(slots=True)
class MSELoss(Loss):
    def calculate_total_loss(self, outputs: list[float], expects: list[float]) -> float:
        errors = [(output - expect) ** 2 for output, expect in zip(outputs, expects)]
        return sum(errors) / len(outputs)
    
    def calculate_gradients(self, outputs: list[float], expects: list[float]) -> list[float]:
        n = len(outputs)
        return [(2/n) * (output - expect) for output, expect in zip(outputs, expects)]

@dataclass(slots=True)
class BCELoss(Loss):
    def calculate_total_loss(self, outputs: list[float], expects: list[float]) -> float:
        epsilon = 1e-15
        return -sum(
            expect * math.log(max(min(output, 1 - epsilon), epsilon)) + 
            (1 - expect) * math.log(max(min(1 - output, 1 - epsilon), epsilon))
            for output, expect in zip(outputs, expects)
        )
    
    def calculate_gradients(self, outputs: list[float], expects: list[float]) -> list[float]:
        epsilon = 1e-15
        return [
            -expect/(max(output, epsilon)) + 
            (1-expect)/(max(1-output, epsilon))
            for output, expect in zip(outputs, expects)
        ]

@dataclass(slots=True)
class Network:
    layers: list[Layer]
    network_type: str
    bias: float = 1.0
    
    def get_activation_function(self, layer_index: int) -> Callable:
        if self.network_type == "regression":
            if layer_index == 0:
                return Activation.relu
            return Activation.linear
        else: 
            if layer_index == len(self.layers) - 1:
                return Activation.sigmoid
            return Activation.relu 
    
    def get_activation_derivative(self, layer_index: int) -> Callable:
        if self.network_type == "regression":
            if layer_index == 0:
                return Activation.relu_derivative
            return Activation.linear_derivative
        else:
            if layer_index == len(self.layers) - 1:
                return Activation.sigmoid_derivative
            return Activation.relu_derivative
    
    def forward(self, input_values: list[float]) -> list[float]:
        current_values = input_values
        for i, layer in enumerate(self.layers):
            activation_func = self.get_activation_function(i)
            current_values = layer.forward(current_values, activation_func)
        return current_values
    
    def backward(self, output_gradients: list[float]) -> None:
        current_gradients = output_gradients
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            activation_derivative = self.get_activation_derivative(i)
            current_gradients = layer.backward(current_gradients, activation_derivative)
    
    def zero_grad(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.zero_grad(learning_rate)
    
    def print_weights(self) -> None:
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}")
            print(layer.weights)
            print(layer.bias_weights)
            print()

def neural_network_task(task_name: str, repeat_times: int = 1, print_weights: bool = True) -> Callable:
    def decorator(task_func: Callable) -> Callable:
        @wraps(task_func)
        def wrapper():
            print(f"------ {task_name} ------")
            network, loss_function, learning_rate, test_cases = task_func()
            
            for inputs, expects in test_cases:
                for _ in range(repeat_times):
                    outputs = network.forward(inputs)
                    total_loss = loss_function.calculate_total_loss(outputs, expects)
                    gradients = loss_function.calculate_gradients(outputs, expects)
                    network.backward(gradients)
                    network.zero_grad(learning_rate)
                
                if print_weights:
                    network.print_weights()
                else:
                    print(f"Total Loss {total_loss}")
                    
        return wrapper
    return decorator

class TestData:
    REGRESSION = {
        "network_type": "regression",
        "layers": [
            {  
                "weights": [[0.5, 0.2], [0.6, -0.6]],
                "bias_weights": [0.3, 0.25]
            },
            {  
                "weights": [[0.8, -0.5]],
                "bias_weights": [0.6]
            },
            { 
                "weights": [[0.6], [-0.3]],
                "bias_weights": [0.4, 0.75]
            }
        ],
        "test_cases": [
            ([1.5, 0.5], [0.8, 1.0])
        ],
        "learning_rate": 0.01
    }
    
    BINARY_CLASSIFICATION = {
        "network_type": "binary",
        "layers": [
            {  
                "weights": [[0.5, 0.2], [0.6, -0.6]],
                "bias_weights": [0.3, 0.25]
            },
            { 
                "weights": [[0.8, 0.4]],
                "bias_weights": [-0.5]
            }
        ],
        "test_cases": [
            ([0.75, 1.25], [1])
        ],
        "learning_rate": 0.1
    }

@neural_network_task("Model 1\nTask 1", repeat_times=1, print_weights=True)
def execute_regression_task1():
    data = TestData.REGRESSION
    network = Network(
        layers=[Layer(**layer_data) for layer_data in data["layers"]],
        bias=1.0,
        network_type=data["network_type"]
    )
    loss_function = MSELoss()
    return network, loss_function, data["learning_rate"], data["test_cases"]

@neural_network_task("Task 2", repeat_times=1000, print_weights=False)
def execute_regression_task2():
    data = TestData.REGRESSION
    network = Network(
        layers=[Layer(**layer_data) for layer_data in data["layers"]],
        bias=1.0,
        network_type=data["network_type"]
    )
    loss_function = MSELoss()
    return network, loss_function, data["learning_rate"], data["test_cases"]

@neural_network_task("Model 2\nTask 1", repeat_times=1, print_weights=True)
def execute_binary_classification_task1():
    data = TestData.BINARY_CLASSIFICATION
    network = Network(
        layers=[Layer(**layer_data) for layer_data in data["layers"]],
        bias=1.0,
        network_type=data["network_type"]
    )
    loss_function = BCELoss()
    return network, loss_function, data["learning_rate"], data["test_cases"]

@neural_network_task("Task 2", repeat_times=1000, print_weights=False)
def execute_binary_classification_task2():
    data = TestData.BINARY_CLASSIFICATION
    network = Network(
        layers=[Layer(**layer_data) for layer_data in data["layers"]],
        bias=1.0,
        network_type=data["network_type"]
    )
    loss_function = BCELoss()
    return network, loss_function, data["learning_rate"], data["test_cases"]

if __name__ == '__main__':
    print("------ Model 1 ------")
    execute_regression_task1()
    execute_regression_task2()
    print("\n------ Model 2 ------")
    execute_binary_classification_task1()
    execute_binary_classification_task2()
