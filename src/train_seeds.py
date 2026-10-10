import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import numpy as np

from data import get_data_hash
from model import SimpleCNN, set_seed

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_experiment("ho2-fashion-mnist")

print("Mencari model terbaik dari hasil Optuna di MLflow...")
# 1. Cari Best Params secara otomatis menggunakan API MLflow (L5 Level)
runs = mlflow.search_runs(
    experiment_names=["ho2-fashion-mnist"],
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.val_accuracy DESC"]
)

if runs.empty:
    print("Error: Tidak ada hasil Optuna yang ditemukan!")
    sys.exit(1)

best_run = runs.iloc[0]
best_lr = float(best_run["params.lr"])
best_batch_size = int(best_run["params.batch_size"])
best_dropout = float(best_run["params.dropout_rate"])

print(f"Konfigurasi Terbaik Ditemukan:\n LR = {best_lr:.5f}\n Batch Size = {best_batch_size}\n Dropout = {best_dropout}")

# 2. Persiapan Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
full_train = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=False)
train_subset, val_subset = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(42))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seeds = [42, 123, 999]
accuracies = []

print("\nMemulai Evaluasi 3 Seed...")
# 3. Parent Run untuk 3 Seed
with mlflow.start_run(run_name="03-seed-evaluation"):
    mlflow.log_params({"lr": best_lr, "batch_size": best_batch_size, "dropout_rate": best_dropout})
    mlflow.log_param("data_md5", get_data_hash())
    mlflow.set_tag("stage", "seed_evaluation")
    
    for idx, seed in enumerate(seeds):
        # Kunci seed untuk eksperimen ini
        set_seed(seed)
        train_loader = DataLoader(train_subset, batch_size=best_batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=best_batch_size, shuffle=False)
        
        model = SimpleCNN(dropout_rate=best_dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_lr)
        
        # Kita train 5 epoch per seed agar stabil
        epochs = 5
        
        # Child run per seed
        with mlflow.start_run(run_name=f"seed-{seed}", nested=True):
            mlflow.log_param("seed", seed)
            
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
            
            # Simpan model child
            example_input = torch.randn(1, 1, 28, 28).to(device)
            mlflow.pytorch.log_model(model, "model", input_example=example_input.cpu().numpy())
            
            accuracies.append(val_acc)
            print(f"  Seed {seed} Selesai -> Val Acc: {val_acc:.4f}")
    
    # 4. Hitung Mean dan Standar Deviasi
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    mlflow.log_metric("mean_val_accuracy", mean_acc)
    mlflow.log_metric("std_val_accuracy", std_acc)
    
    print("\n=== EVALUASI SEED SELESAI ===")
    print(f"Akurasi Rata-rata (Mean) : {mean_acc:.4f}")
    print(f"Standar Deviasi (Std)    : {std_acc:.4f}")
