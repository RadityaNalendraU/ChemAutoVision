import os
import pandas as pd

# Konfigurasi URL Dataset
DATASETS = {
    "clintox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "smiles_col": "smiles",
        "target_col": "FDA_APPROVED"
    },
    "sider": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "smiles_col": "smiles",
        "target_col": "Hepatobiliary disorders"
    },
    "tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "smiles_col": "smiles",
        "target_col": "SR-p53"
    },
    "bace": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "smiles_col": "smiles",
        "target_col": "Class"
    }
}

ROOT_DIR = "/chemAutoVision"

ROOT_DIR = "/chemAutoVision"

def download_datasets():
    print("=== MEMULAI PROSES UNDUH DATASET ===")
    for data_name, config in DATASETS.items():
        raw_dir = os.path.join(ROOT_DIR, "data", "raw", data_name)
        
        # Ekstensi menyesuaikan apakah itu .csv atau .csv.gz
        ext = ".csv.gz" if config['url'].endswith(".gz") else ".csv"
        csv_path = os.path.join(raw_dir, f"{data_name}{ext}")
        
        # Buat direktori jika belum ada
        os.makedirs(raw_dir, exist_ok=True)
        
        if not os.path.exists(csv_path):
            print(f"[+] Mengunduh {data_name.upper()} dari {config['url']}...")
            try:
                df = pd.read_csv(config['url'])
                df.to_csv(csv_path, index=False)
                print(f"    -> Berhasil disimpan di: {csv_path}")
            except Exception as e:
                print(f"    -> [ERROR] Gagal mengunduh {data_name}: {e}")
        else:
            print(f"[-] {data_name.upper()} sudah ada di {csv_path}. Melewati proses unduh.")

if __name__ == "__main__":
    download_datasets()
    print("=== PROSES UNDUH SELESAI ===")