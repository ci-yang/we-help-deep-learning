import torch

def task1():
    print("Task 1: Create Tensor from list")
    data = [[2, 3, 1], [5, -2, 1]]
    tensor = torch.tensor(data)
    print(f"Tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"Data Type: {tensor.dtype}\n")

def task2():
    print("Task 2: Create random Tensor")
    tensor = torch.rand(3, 4, 2)
    print(f"Shape: {tensor.shape}")
    print(f"Content:\n{tensor}\n")

def task3():
    print("Task 3: Create ones Tensor")
    tensor = torch.ones(2, 1, 5)
    print(f"Shape: {tensor.shape}")
    print(f"Content:\n{tensor}\n")

def task4():
    print("Task 4: Matrix Multiplication")
    tensor1 = torch.tensor([[1, 2, 4], [2, 1, 3]])
    tensor2 = torch.tensor([[5], [2], [1]])
    result = torch.matmul(tensor1, tensor2)
    print(f"Result:\n{result}\n")

def task5():
    print("Task 5: Element-wise Product")
    tensor1 = torch.tensor([[1, 2], [2, 3], [-1, 3]])
    tensor2 = torch.tensor([[5, 4], [2, 1], [1, -5]])
    result = tensor1 * tensor2
    print(f"Result:\n{result}\n")

if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
    task5()