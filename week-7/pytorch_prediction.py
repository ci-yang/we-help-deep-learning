import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import csv
import os
import math

class WeightDataset(Dataset):
    def __init__(self, file_path: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, file_path)
        
        self.data = []
        self.height_mean = 0
        self.height_std = 1
        self.weight_mean = 0
        self.weight_std = 1
        
        self._load_data(data_path)
    
    def _load_data(self, file_path: str):
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            raw_data = list(reader)
            
            heights = [float(row['Height']) for row in raw_data]
            weights = [float(row['Weight']) for row in raw_data]
            
            self.height_mean = sum(heights) / len(heights)
            self.weight_mean = sum(weights) / len(weights)
            
            self.height_std = math.sqrt(sum((x - self.height_mean) ** 2 for x in heights) / len(heights))
            self.weight_std = math.sqrt(sum((x - self.weight_mean) ** 2 for x in weights) / len(weights))
            
            for row in raw_data:
                gender = [1.0, 0.0] if row['Gender'] == 'Male' else [0.0, 1.0]
                height = (float(row['Height']) - self.height_mean) / self.height_std
                weight = (float(row['Weight']) - self.weight_mean) / self.weight_std
                self.data.append((gender, height, weight))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        gender, height, weight = self.data[idx]
        features = torch.tensor(gender + [height], dtype=torch.float32)
        target = torch.tensor([weight], dtype=torch.float32)
        return features, target
    
    def denormalize_weight(self, normalized_weight: float) -> float:
        return normalized_weight * self.weight_std + self.weight_mean
    
    def transform_loss_to_weights(self, loss: float) -> float:
        loss = loss * (self.weight_std ** 2)
        return math.sqrt(loss)

class TitanicDataset(Dataset):
    def __init__(self, file_path: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, file_path)
        
        self.data = []
        self.age_mean = 0
        self.age_std = 1
        self.fare_mean = 0
        self.fare_std = 1
        
        self._load_data(data_path)
    
    def _load_data(self, file_path: str):
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            raw_data = list(reader)
            
            ages = [float(row['Age']) for row in raw_data if row['Age']]
            fares = [float(row['Fare']) for row in raw_data if row['Fare']]
            
            self.age_mean = sum(ages) / len(ages)
            self.fare_mean = sum(fares) / len(fares)
            
            self.age_std = math.sqrt(sum((x - self.age_mean) ** 2 for x in ages) / len(ages))
            self.fare_std = math.sqrt(sum((x - self.fare_mean) ** 2 for x in fares) / len(fares))
            
            for row in raw_data:
                features = []
                features.append(1.0 if row['Sex'] == 'male' else 0.0)
                
                pclass = int(row['Pclass'])
                features.extend([1.0 if i == pclass else 0.0 for i in range(1, 4)])
                
                age = float(row['Age']) if row['Age'] else self.age_mean
                features.append((age - self.age_mean) / self.age_std)
                
                fare = float(row['Fare']) if row['Fare'] else self.fare_mean
                features.append((fare - self.fare_mean) / self.fare_std)
                
                embarked = row['Embarked'] if row['Embarked'] else 'S'
                embarked_map = {'S': [1,0,0], 'C': [0,1,0], 'Q': [0,0,1]}
                features.extend(embarked_map[embarked])
                
                self.data.append((features, float(row['Survived'])))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, target = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)

class WeightPredictionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 4)
        self.layer2 = nn.Linear(4, 1)
        
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.0)
        
        with torch.no_grad():
            self.layer1.bias.fill_(0.1)
            self.layer2.bias.fill_(0.1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class TitanicModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 8)
        self.layer2 = nn.Linear(8, 1)
        
        nn.init.kaiming_uniform_(self.layer1.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.0)
        
        with torch.no_grad():
            self.layer1.bias.fill_(0.1)
            self.layer2.bias.fill_(0.1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        return x

def execute_weight_prediction():
    dataset = WeightDataset('data/gender-height-weight.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = WeightPredictionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print("Before Training Evaluation...")
    model.eval()
    with torch.no_grad():
        total_loss = sum(criterion(model(features), targets).item() * len(features)
                        for features, targets in dataloader)
        average_loss = total_loss / len(dataset)
        initial_error = dataset.transform_loss_to_weights(average_loss)
        print(f"Initial Average Error: {initial_error:.2f} pounds\n")
    
    print("Training...")
    training_iterations = 50
    
    for iteration in range(training_iterations):
        model.train()
        total_loss = 0
        
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(features)
    
    print("\nFinal Evaluation...")
    model.eval()
    with torch.no_grad():
        total_loss = sum(criterion(model(features), targets).item() * len(features)
                        for features, targets in dataloader)
        average_loss = total_loss / len(dataset)
        final_error = dataset.transform_loss_to_weights(average_loss)
        print(f"Final Average Error: {final_error:.2f} pounds")

def execute_titanic_prediction():
    dataset = TitanicDataset('data/titanic.csv')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = TitanicModel(input_size=9)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    print("Before Training Evaluation...")
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_loss = sum(criterion(model(features), targets).item() * len(features)
                        for features, targets in dataloader)
        
        for features, targets in dataloader:
            outputs = model(features)
            predictions = (outputs >= 0.5).float()
            total_correct += (predictions == targets).sum().item()
        
        average_loss = total_loss / len(dataset)
        accuracy = total_correct / len(dataset)
        print(f"Initial Loss: {average_loss:.4f}")
        print(f"Initial Accuracy: {accuracy:.2%}\n")
    
    print("Training...")
    training_iterations = 100
    
    for iteration in range(training_iterations):
        model.train()
        total_loss = 0
        total_correct = 0
        
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(features)
            predictions = (outputs >= 0.5).float()
            total_correct += (predictions == targets).sum().item()
    
    print("\nFinal Evaluation...")
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_loss = sum(criterion(model(features), targets).item() * len(features)
                        for features, targets in dataloader)
        
        for features, targets in dataloader:
            outputs = model(features)
            predictions = (outputs >= 0.5).float()
            total_correct += (predictions == targets).sum().item()
        
        average_loss = total_loss / len(dataset)
        accuracy = total_correct / len(dataset)
        print(f"Final Loss: {average_loss:.4f}")
        print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    print("=== Weight Prediction ===")
    execute_weight_prediction()
    print("\n=== Titanic Prediction ===")
    execute_titanic_prediction()
