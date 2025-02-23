from __future__ import annotations
from abc import ABC, abstractmethod
from functools import wraps
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Iterator
import csv
import math
import os
import random


def he_init(input_size: int, output_size: int) -> list[list[float]]:
    """He 初始化，適用於 ReLU 層"""
    limit = math.sqrt(2 / input_size)
    return [
        [random.uniform(-limit, limit) for _ in range(input_size)]
        for _ in range(output_size)
    ]


def xavier_init(input_size: int, output_size: int) -> list[list[float]]:
    """Xavier 初始化，適用於線性層"""
    limit = math.sqrt(6 / (input_size + output_size))
    return [
        [random.uniform(-limit, limit) for _ in range(input_size)]
        for _ in range(output_size)
    ]


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
        self.weight_gradients = [
            [0.0] * len(self.weights[0]) for _ in range(len(self.weights))
        ]
        self.bias_gradients = [0.0] * len(self.bias_weights)

    def inner_product(self, weights: list[float], inputs: list[float]) -> float:
        return sum(weight * input_value for weight, input_value in zip(weights, inputs))

    def forward(
        self, input_values: list[float], activation_func: Callable
    ) -> list[float]:
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

    def backward(
        self, next_layer_gradients: list[float], activation_derivative: Callable
    ) -> list[float]:
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
    def calculate_gradients(
        self, outputs: list[float], expects: list[float]
    ) -> list[float]:
        pass


@dataclass(slots=True)
class MSELoss(Loss):
    def calculate_total_loss(self, outputs: list[float], expects: list[float]) -> float:
        errors = [(output - expect) ** 2 for output, expect in zip(outputs, expects)]
        return sum(errors) / len(outputs)

    def calculate_gradients(
        self, outputs: list[float], expects: list[float]
    ) -> list[float]:
        n = len(outputs)
        return [(2 / n) * (output - expect) for output, expect in zip(outputs, expects)]


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


@dataclass
class DataSet:
    def __init__(self, file_path: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, file_path)
        self.data = self._read_csv(data_path)
        self.height_mean, self.height_std = self._calculate_stats("Height")
        self.weight_mean, self.weight_std = self._calculate_stats("Weight")

    def _read_csv(self, file_path: str) -> List[dict]:
        with open(file_path, "r") as file:
            return list(csv.DictReader(file))

    def _calculate_stats(self, column: str) -> Tuple[float, float]:
        values = [float(row[column]) for row in self.data]
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        std = math.sqrt(squared_diff_sum / len(values))
        return mean, std

    def normalize_value(self, value: float, mean: float, std: float) -> float:
        return (value - mean) / std

    def denormalize_weight(self, normalized_weight: float) -> float:
        return normalized_weight * self.weight_std + self.weight_mean

    def transform_loss_to_weights(self, loss: float) -> float:
        """將MSE損失轉換為實際體重誤差"""
        loss = loss * (self.weight_std**2)
        return math.sqrt(loss)

    def encode_data(self) -> Tuple[List[List[float]], List[float]]:
        encoded_inputs = []
        encoded_targets = []

        for row in self.data:
            gender_encoded = [1, 0] if row["Gender"] == "Male" else [0, 1]

            height_normalized = self.normalize_value(
                float(row["Height"]), self.height_mean, self.height_std
            )

            encoded_input = gender_encoded + [height_normalized]
            encoded_inputs.append(encoded_input)

            weight_normalized = self.normalize_value(
                float(row["Weight"]), self.weight_mean, self.weight_std
            )
            encoded_targets.append([weight_normalized])

        return encoded_inputs, encoded_targets

    def get_batches(
        self, xs: List[List[float]], es: List[List[float]], batch_size: int
    ) -> Iterator[Tuple[List[List[float]], List[List[float]]]]:
        indices = list(range(len(xs)))
        random.shuffle(indices)

        for i in range(0, len(xs), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_xs = [xs[j] for j in batch_indices]
            batch_es = [es[j] for j in batch_indices]
            yield batch_xs, batch_es


def create_network() -> Network:
    return Network(
        layers=[
            Layer(
                weights=he_init(3, 4),
                bias_weights=[0.1] * 4,
            ),
            Layer(
                weights=xavier_init(4, 1),
                bias_weights=[0.1] * 1,
            ),
        ],
        network_type="regression",
    )


def neural_network_task(task_name: str) -> Callable:
    def decorator(task_func: Callable) -> Callable:
        @wraps(task_func)
        def wrapper():
            print(f"------ {task_name} Tasks -------")
            task_func()
            print(f"------ {task_name} Done -------")

        return wrapper

    return decorator


@neural_network_task("Weight Prediction")
def execute_weight_prediction():
    dataset = DataSet("data/gender-height-weight.csv")
    network = create_network()
    loss_function = MSELoss()
    learning_rate = 0.01
    batch_size = 32
    training_iterations = 50

    training_data, expected_values = dataset.encode_data()

    print("Training...")
    for iteration in range(training_iterations):
        total_loss = 0
        batch_count = 0

        for batch_inputs, batch_expects in dataset.get_batches(
            training_data, expected_values, batch_size
        ):
            batch_outputs = [network.forward(inputs) for inputs in batch_inputs]
            batch_loss = sum(
                loss_function.calculate_total_loss(outputs, expects)
                for outputs, expects in zip(batch_outputs, batch_expects)
            )

            for inputs, expects in zip(batch_inputs, batch_expects):
                outputs = network.forward(inputs)
                gradients = loss_function.calculate_gradients(outputs, expects)
                network.backward(gradients)
                network.zero_grad(learning_rate)

            total_loss += batch_loss
            batch_count += 1

        if (iteration + 1) % 10 == 0:
            average_loss = total_loss / (batch_count * batch_size)
            error = dataset.transform_loss_to_weights(average_loss)
            print(f"Iteration {iteration + 1}, Average Error: {error:.2f} pounds")

    print("\nFinal Evaluation...")
    final_loss = 0
    final_count = 0
    for batch_inputs, batch_expects in dataset.get_batches(
        training_data, expected_values, batch_size
    ):
        batch_outputs = [network.forward(inputs) for inputs in batch_inputs]
        batch_loss = sum(
            loss_function.calculate_total_loss(outputs, expects)
            for outputs, expects in zip(batch_outputs, batch_expects)
        )
        final_loss += batch_loss
        final_count += len(batch_inputs)

    average_loss = final_loss / final_count
    final_error = dataset.transform_loss_to_weights(average_loss)
    print(f"Final Average Error: {final_error:.2f} pounds")


if __name__ == "__main__":
    execute_weight_prediction()
