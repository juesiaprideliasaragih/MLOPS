import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from torchvision import datasets, transforms
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

print("1. Memuat model @champion dari Model Registry...")
# Keajaiban MLOps: Kita tidak perlu tahu lokasi file pastinya atau versi ke berapa,
# cukup panggil nama model dan aliasnya (@champion).
model_uri = "models:/fashion-mnist-cnn@champion"
model = mlflow.pytorch.load_model(model_uri)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n2. Mengambil 5 gambar sampel dari dataset...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_set = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=False)

# Kamus nama kelas untuk Fashion-MNIST
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("\n3. Melakukan Prediksi (Inferensi) di Produksi...\n")
with torch.no_grad():
    for i in range(5): # Ambil 5 gambar pertama
        image, actual_label = test_set[i]
        
        # Tambahkan dimensi batch menjadi (1, 1, 28, 28)
        image_batch = image.unsqueeze(0).to(device)
        
        # Prediksi
        output = model(image_batch)
        _, predicted_idx = torch.max(output.data, 1)
        
        pred_class = classes[predicted_idx.item()]
        actual_class = classes[actual_label]
        
        status = "✅ BENAR" if pred_class == actual_class else "❌ SALAH"
        print(f"Gambar {i+1} | Jawaban Asli: {actual_class:15} | Prediksi AI: {pred_class:15} | {status}")

print("\n=== INFERENSI SELESAI ===")
