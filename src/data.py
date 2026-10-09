import torchvision
import os
import yaml

def get_data_hash():
    """Membaca MD5 Hash dari file DVC sebagai versi dataset (Reproducibility L1)"""
    dvc_file = 'data/FashionMNIST.dvc'
    if os.path.exists(dvc_file):
        with open(dvc_file, 'r') as f:
            dvc_data = yaml.safe_load(f)
            return dvc_data['outs'][0]['md5']
    return "unknown_hash"

if __name__ == "__main__":
    # Eksekusi untuk mendownload dataset
    os.makedirs("data", exist_ok=True)
    print("Mengunduh Fashion-MNIST...")
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)
    
    print(f"Jumlah Train: {len(train_set)}")
    print(f"Jumlah Test: {len(test_set)}")
    print("Dataset berhasil diunduh ke folder ./data/FashionMNIST")