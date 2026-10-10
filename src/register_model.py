import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
from mlflow import MlflowClient

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
mlflow.set_experiment("ho2-fashion-mnist")

print("1. Mencari Model Terbaik dari MLflow...")
runs = mlflow.search_runs(
    experiment_names=["ho2-fashion-mnist"],
    filter_string="tags.candidate = 'true'",
    order_by=["metrics.val_accuracy DESC"]
)
best_run = runs.iloc[0]
best_run_id = best_run["run_id"]
best_model_uri = f"runs:/{best_run_id}/model"
print(f"Model terbaik ditemukan di Run ID: {best_run_id}")

print("\n2. Evaluasi Final pada TEST SET...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

# KUNCI FIX: Ubah batch_size menjadi 1. 
# Saat melatih model, kita menyimpannya dengan `input_example` berukuran 1 gambar (batch=1).
# Oleh karena itu, grafik model PyTorch terkunci secara permanen di ukuran 1 gambar.
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

loaded_model = mlflow.pytorch.load_model(best_model_uri)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Panggil model secara langsung (tanpa .eval() karena sudah dikunci saat export)
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

test_acc = correct / len(test_set)
print(f"Akurasi Test Set: {test_acc:.4f}")

# Catat metrik test ke Run yang bersangkutan
with mlflow.start_run(run_id=best_run_id):
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.set_tag("test_evaluated", "true")

print("\n3. Quality Gate & Model Registry...")
MIN_ACCURACY = 0.85
NAME = "fashion-mnist-cnn"
client = MlflowClient()

if test_acc >= MIN_ACCURACY:
    print(f"✅ Quality Gate Lulus (Test Acc {test_acc:.4f} >= {MIN_ACCURACY})")
    
    mv = mlflow.register_model(best_model_uri, NAME)
    client.set_model_version_tag(NAME, mv.version, "validation_status", "approved")
    client.set_registered_model_alias(NAME, "champion", mv.version)
    
    print(f"🎉 Model berhasil diregistrasi dengan nama '{NAME}' versi {mv.version}!")
    print(f"🎉 Alias @champion telah diberikan!")
else:
    print(f"❌ Quality Gate Gagal (Test Acc < {MIN_ACCURACY}).")
