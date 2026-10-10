import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import optuna
import mlflow
import mlflow.pytorch

from data import get_data_hash
from model import SimpleCNN, set_seed

# 1. Setup MLflow
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_experiment("ho2-fashion-mnist")

# 2. Siapkan Data (Global agar tidak diload berulang-ulang tiap trial)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
train_subset, val_subset = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(42))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # 3. Sugesti Hyperparameter oleh Optuna
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    epochs = 3 # Di-set 3 epoch per trial agar 20 trial tidak memakan waktu berjam-jam
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    # Kunci seed agar hasil adil antar trial
    set_seed(42) 
    model = SimpleCNN(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. MLflow Child Run (Nested = True)
    with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
        mlflow.log_params({"lr": lr, "batch_size": batch_size, "dropout_rate": dropout_rate, "epochs": epochs})
        mlflow.set_tag("candidate", "true")
        
        val_acc = 0
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / 10000
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        
        # Simpan format model MLflow pt2
        example_input = torch.randn(1, 1, 28, 28).to(device)
        mlflow.pytorch.log_model(model, "model", input_example=example_input.cpu().numpy())
        
        return val_acc

if __name__ == "__main__":
    print("Memulai Hyperparameter Tuning dengan Optuna...")
    # 5. Parent Run
    with mlflow.start_run(run_name="02-optuna-sweep"):
        mlflow.log_param("data_md5", get_data_hash())
        mlflow.set_tag("stage", "tuning")
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20) # Sesuai syarat >= 20 trial
        
        # Catat hasil terbaik di parent run
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_accuracy", study.best_value)
        
        print("\n=== TUNING SELESAI ===")
        print(f"Konfigurasi Terbaik: {study.best_params}")
        print(f"Akurasi Terbaik: {study.best_value}")
