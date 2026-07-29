import os
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw

# Konfigurasi Kolom Dataset Drug-Based (Tanpa BBBP)
DATASETS = {
    "clintox": {"smiles_col": "smiles", "target_col": "FDA_APPROVED"},
    "sider": {"smiles_col": "smiles", "target_col": "Hepatobiliary disorders"},
    "tox21": {"smiles_col": "smiles", "target_col": "SR-p53"},
    "bace": {"smiles_col": "mol", "target_col": "Class"}
}

ROOT_DIR = "/chemAutoVision"

def render_images():
    print("=== MEMULAI PROSES RENDER GAMBAR DRUG-BASED ===")
    for data_name, config in DATASETS.items():
        raw_dir = os.path.join(ROOT_DIR, "data", "raw", data_name)
        image_dir = os.path.join(ROOT_DIR, "data", "images", data_name)
        
        csv_path_gz = os.path.join(raw_dir, f"{data_name}.csv.gz")
        csv_path_normal = os.path.join(raw_dir, f"{data_name}.csv")
        
        if os.path.exists(csv_path_gz):
            csv_path = csv_path_gz
        elif os.path.exists(csv_path_normal):
            csv_path = csv_path_normal
        else:
            print(f"\n[ERROR] File sumber untuk {data_name.upper()} tidak ditemukan. Lewati.")
            continue
            
        os.makedirs(image_dir, exist_ok=True)
        
        df = pd.read_csv(csv_path)
        print(f"\n[+] Memproses {data_name.upper()} ({len(df)} baris)")
        
        success_count = 0
        failed_count = 0
        failed_smiles = []
        smiles_col = config['smiles_col']

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Rendering {data_name}"):
            try:
                smiles = str(row[smiles_col])
                img_name = f"{data_name}_{idx}.png"
                img_path = os.path.join(image_dir, img_name)
                
                if os.path.exists(img_path):
                    success_count += 1
                    continue

                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(256, 256))
                    img.save(img_path)
                    success_count += 1
                else:
                    failed_count += 1
                    failed_smiles.append((idx, smiles))
                    
            except Exception:
                failed_count += 1
                failed_smiles.append((idx, smiles))
                
        print(f"    -> Gambar Berhasil : {success_count}")
        print(f"    -> Gambar Gagal    : {failed_count}")
        
        if failed_count > 0:
            error_log_path = os.path.join(ROOT_DIR, "data", "raw", data_name, "failed_smiles.csv")
            failed_df = pd.DataFrame(failed_smiles, columns=["index", "smiles"])
            failed_df.to_csv(error_log_path, index=False)
            print(f"    -> Log Error disimpan di: {error_log_path}")

if __name__ == "__main__":
    render_images()
    print("\n=== PROSES RENDER SELESAI ===")