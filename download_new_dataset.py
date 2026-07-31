import os
import pandas as pd

# Konfigurasi seluruh dataset yang ingin diunduh
DATASETS_CONFIG = {
    # "BBB": {
    #     "train": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/BBB_permeability/BBB_train_set.csv",
    #     "test": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/BBB_permeability/BBB_test_set.csv",
    #     "expected_count": 2041
    # },
    # "Caco2": {
    #     "train": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/Caco2_permeability/Papp_train_set.csv",
    #     "test": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/Caco2_permeability/Papp_test_set.csv",
    #     "expected_count": 3475
    # },
    "RLM": {
        "train": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/RLM_stability/RCLint_train_set.csv",
        "test": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/RLM_stability/RCLint_test_set.csv",
        "expected_count": 3108
    },
    "HLM": {
        "train": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/HLM_stability/HCLint_train_set.csv",
        "test": "https://raw.githubusercontent.com/CADD-SC/ADMET_Prediction_Models/main/data/HLM_stability/HCLint_test_set.csv",
        "expected_count": 5902
    }
}

ROOT_DIR = "/chemAutoVision"

def download_and_merge_datasets():
    print("=== MEMULAI UNDUH DAN GABUNG MULTIPLE DATASET HAN ET AL. ===")
    
    for data_name, config in DATASETS_CONFIG.items():
        print(f"\n--- Memproses Dataset: {data_name} ---")
        
        raw_dir = os.path.join(ROOT_DIR, "data", "raw", data_name)
        os.makedirs(raw_dir, exist_ok=True)
        
        final_csv_path = os.path.join(raw_dir, f"{data_name}.csv")
        
        if not os.path.exists(final_csv_path):
            try:
                print(f"[+] Mengunduh data latih (train) untuk {data_name}...")
                df_train = pd.read_csv(config["train"])
                
                print(f"[+] Mengunduh data uji (test) untuk {data_name}...")
                df_test = pd.read_csv(config["test"])
                
                # Menggabungkan kedua data menjadi satu data raw utuh
                df_combined = pd.concat([df_train, df_test], ignore_index=True)
                
                # Menyimpan data gabungan
                df_combined.to_csv(final_csv_path, index=False)
                
                print(f"    -> Berhasil! Data digabungkan dan disimpan di: {final_csv_path}")
                print(f"    -> Total molekul: {len(df_combined)} (Seharusnya ~{config['expected_count']} molekul)")
                
            except Exception as e:
                print(f"    -> [ERROR] Gagal memproses data {data_name}: {e}")
                print("    -> Pastikan tautan Raw URL di dalam script sudah benar sesuai nama file di GitHub.")
        else:
            print(f"[-] Data gabungan {data_name} sudah ada di {final_csv_path}.")

if __name__ == "__main__":
    download_and_merge_datasets()