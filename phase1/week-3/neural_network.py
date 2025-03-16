from dataclasses import dataclass
from typing import Callable
from functools import wraps

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

def neural_network_task(task_name: str) -> Callable:
    def decorator(task_func: Callable) -> Callable:
        @wraps(task_func)
        def wrapper():
            print(f'{task_name}:')
            network, test_inputs = task_func()
            for input_values in test_inputs:
                outputs = network.forward(input_values)
                print(outputs)
        return wrapper
    return decorator

@neural_network_task('Task1')
def execute_task1():
    network = Network(
        layers=[
            Layer(weights=[
                [0.5, 0.2, 0.3],
                [0.6, -0.6, 0.25]
            ]),
            Layer(weights=[
                [0.8, 0.4, -0.5]
            ])
        ],
        bias=1.0
    )
    
    test_inputs = [
        [1.5, 0.5],
        [0, 1]
    ]
    
    return network, test_inputs

@neural_network_task('Task2')
def execute_task2():
    network = Network(
        layers=[
            Layer(weights=[
                [0.5, 1.5, 0.3],
                [0.6, -0.8, 1.25]
            ]),
            Layer(weights=[
                [0.6, -0.8, 0.3]
            ]),
            Layer(weights=[
                [0.5, 0.2],
                [-0.4, 0.5]
            ])
        ],
        bias=1.0
    )
    
    test_inputs = [
        [0.75, 1.25],
        [-1, 0.5]
    ]
    
    return network, test_inputs

if __name__ == '__main__':
    execute_task1()
    print()
    execute_task2()
