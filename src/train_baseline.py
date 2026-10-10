import os
import sys
# Hindari error emoji MLflow di Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch

from data import get_data_hash
from model import SimpleCNN, set_seed

# 1. Setup Konfigurasi MLflow
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_experiment("ho2-fashion-mnist")

config = {
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 5, # Baseline cukup 5 epoch
    "seed": 42,
    "dropout_rate": 0.0 # Baseline murni tanpa dropout
}
set_seed(config["seed"])

# 2. Siapkan Data (Data Validation dipisah, JANGAN sentuh Test Set!)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)

# Membagi 60.000 data train menjadi 50.000 train dan 10.000 validasi
train_subset, val_subset = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)

# 3. Inisialisasi Model, Loss, dan Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(dropout_rate=config["dropout_rate"]).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# 4. Training Loop & MLflow Logging
with mlflow.start_run(run_name="01-cnn-baseline"):
    # Log Konteks Lengkap
    mlflow.log_params(config)
    mlflow.log_param("data_md5", get_data_hash())
    mlflow.set_tag("stage", "baseline")
    mlflow.set_tag("model_family", "cnn")
    
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validasi Model
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / 10000
        
        # Log metrik per epoch
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={val_acc:.4f}")
        mlflow.log_metrics({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_accuracy": val_acc}, step=epoch)
        
    # Simpan Bobot Model di akhir dengan contoh input untuk format pt2
    example_input = torch.randn(1, 1, 28, 28).to(device)
    mlflow.pytorch.log_model(model, "model", input_example=example_input.cpu().numpy())
    print("Training baseline selesai. Model dan metrik tercatat di MLflow!")
