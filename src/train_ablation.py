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

from data import get_data_hash
from model import SimpleCNN, set_seed

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_experiment("ho2-fashion-mnist")

# 1. Ambil parameter terbaik dari MLflow
runs = mlflow.search_runs(
    experiment_names=["ho2-fashion-mnist"],
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.val_accuracy DESC"]
)
best_run = runs.iloc[0]
best_lr = float(best_run["params.lr"])
best_batch_size = int(best_run["params.batch_size"])
best_dropout = float(best_run["params.dropout_rate"])

# Pastikan dropout ada nilainya agar bisa di-ablation
if best_dropout < 0.05:
    best_dropout = 0.25 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_and_eval(run_name, use_aug, drop_rate):
    # Setup Transforms (Dengan atau Tanpa Augmentasi)
    if use_aug:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), # Augmentasi: Flip gambar
            transforms.RandomRotation(10),     # Augmentasi: Putar gambar 10 derajat
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    # Load datasets
    full_train = datasets.FashionMNIST(root='./data', train=True, transform=train_transform, download=False)
    full_val = datasets.FashionMNIST(root='./data', train=True, transform=val_transform, download=False)
    
    # Split (Pastikan indeks konsisten dengan set_seed)
    set_seed(42)
    train_subset, _ = random_split(full_train, [50000, 10000], generator=torch.Generator().manual_seed(42))
    _, val_subset = random_split(full_val, [50000, 10000], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_subset, batch_size=best_batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=best_batch_size, shuffle=False)
    
    model = SimpleCNN(dropout_rate=drop_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)
    
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({"use_augmentation": use_aug, "dropout_rate": drop_rate})
        mlflow.set_tag("ablation", "true")
        
        val_acc = 0
        for epoch in range(5):
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
            
        example_input = torch.randn(1, 1, 28, 28).to(device)
        mlflow.pytorch.log_model(model, "model", input_example=example_input.cpu().numpy())
        
        print(f"  {run_name} -> Val Acc: {val_acc:.4f}")
        return val_acc

if __name__ == "__main__":
    print("Memulai Ablation Study (Evaluasi Komponen Model)...")
    with mlflow.start_run(run_name="04-ablation-study"):
        mlflow.log_param("data_md5", get_data_hash())
        mlflow.set_tag("stage", "ablation")
        
        print("\n1. Melatih Full Model (Augmentasi + Dropout)")
        acc_full = train_and_eval("Full Model", use_aug=True, drop_rate=best_dropout)
        
        print("2. Melatih Model TANPA Augmentasi")
        acc_no_aug = train_and_eval("- Augmentasi", use_aug=False, drop_rate=best_dropout)
        
        print("3. Melatih Model TANPA Dropout")
        acc_no_drop = train_and_eval("- Dropout", use_aug=True, drop_rate=0.0)
        
        print("\n=== ABLATION STUDY SELESAI ===")
        print(f"Full Model        : {acc_full:.4f}")
        print(f"Tanpa Augmentasi  : {acc_no_aug:.4f} (Selisih: {acc_no_aug - acc_full:.4f})")
        print(f"Tanpa Dropout     : {acc_no_drop:.4f} (Selisih: {acc_no_drop - acc_full:.4f})")
