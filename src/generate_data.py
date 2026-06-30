import argparse
import math
import os

import deepchem as dc
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolStandardize
from sklearn.model_selection import train_test_split

from data.images import padding_image, smi_to_img
from seed import set_seeds
from settings import IMG_SIZE, IS_DOWNSCALE_BEFORE_DAUG

RAW_HERG_CSV_PATH = "../data/raw/hERG_inhibition/hERG.csv"
RAW_CYP3A4_CSV_PATH = "../data/raw/CYP3A4_inhibition/CYP3A4.csv"
RAW_PGP_CSV_PATH = "../data/raw/P-gp_substrate/P-gp.csv"


def _generate_image(_df_row: pd.Series, task_name: str) -> pd.Series:
    set_seeds(99)

    if task_name in ["london", "lz", "dm"]:
        img_dir = "qm"
    else:
        img_dir = task_name
    downscale_ratio = 1 / 1.5 if IS_DOWNSCALE_BEFORE_DAUG else 1
    original_img_size = (
        math.floor(IMG_SIZE[0] * downscale_ratio),
        math.floor(IMG_SIZE[1] * downscale_ratio),
    )

    img_file_path = f"../data/images/{img_dir}/{_df_row.idx}.png"

    original_img = smi_to_img(
        _df_row.smiles,
        original_img_size,
    )
    padded_img = padding_image(original_img, (IMG_SIZE[0], IMG_SIZE[1]))
    padded_img.save(img_file_path)

    _df_row["img_path"] = img_file_path

    return _df_row


def _exc_dup_mol(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    if "smiles" not in df.columns or task_name not in df.columns:
        raise ValueError("Invalid DataFrame columns")

    dup_smiles = set(df[df.duplicated(subset=["smiles"], keep=False)]["smiles"])
    if not dup_smiles:
        return df

    smiles_to_remove = set()
    for smiles in dup_smiles:
        subset = df[df["smiles"] == smiles]
        uni_tasks = subset[task_name].unique()
        if len(uni_tasks) > 1:
            smiles_to_remove.add(smiles)

    if smiles_to_remove:
        # Remove all rows with conflicting SMILES
        result_df = df[~df["smiles"].isin(smiles_to_remove)]
    else:
        result_df = df

    # 目的変数の値が同一なのであれば、1件だけ残す
    result_df = result_df.drop_duplicates(subset=["smiles"], keep="first")
    return result_df


def _find_largest_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        lfc = MolStandardize.fragment.LargestFragmentChooser()
        mol = lfc.choose(mol)
        smi = Chem.MolToSmiles(mol)
    else:
        print(f"invalid smiles: {smiles}")
        return None
    return smi


def _preprocess_smiles(smiles: str) -> str:
    smiles = _find_largest_smiles(smiles)
    if smiles is None:
        return None
    smiles = _smi_to_canonical_smi(smiles)
    return smiles


def _load_data(task_name: str) -> pd.DataFrame:
    if task_name == "FreeSolv":
        _, datasets, _ = dc.molnet.load_freesolv(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    elif task_name == "ESOL":
        _, datasets, _ = dc.molnet.load_delaney(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    elif task_name == "Lipo":
        _, datasets, _ = dc.molnet.load_lipo(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    elif task_name == "CYP3A4":
        datasets = pd.read_csv(RAW_CYP3A4_CSV_PATH)
    elif task_name == "P-gp":
        datasets = pd.read_csv(RAW_PGP_CSV_PATH)
    elif task_name == "hERG":
        datasets = pd.read_csv(RAW_HERG_CSV_PATH)
    elif task_name == "BBBP":
        _, datasets, _ = dc.molnet.load_bbbp(
            featurizer=dc.feat.RawFeaturizer(), split=None
        )
    else:
        raise ValueError("Invalid task name is input")

    if not isinstance(datasets, pd.DataFrame):
        datasets = pd.DataFrame(
            {"smiles": datasets[0].ids, task_name: datasets[0].y.flatten()}
        )

    datasets["index"] = datasets.index
    return datasets


def _split_data(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    _df_train, df_test = train_test_split(df, test_size=ratio, random_state=99)
    df_train, df_val = train_test_split(_df_train, test_size=ratio, random_state=99)

    df_train["group"] = "train"
    df_val["group"] = "val"
    df_test["group"] = "test"
    df = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True)
    return df


def _smi_to_canonical_smi(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

def _save_split_csvs(df: pd.DataFrame, task_name: str, prefix: str) -> None:
    cols = ["idx", "smiles", task_name, "group", "img_path"]
    for split in ("train", "val", "test"):
        name = f"{split}_{prefix}_{task_name}_img.csv" if prefix else f"{split}_{task_name}_img.csv"
        out = f"../data/{name}"
        df[df["group"] == split][cols].to_csv(out, index=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="generate data")
    parser.add_argument("--task_name", type=str, help="the task name")
    args = parser.parse_args()
    task_name = args.task_name

    output_path = f"../data/{task_name}_img.pkl"

    _df = _load_data(task_name)

    # desalts
    _df["smiles"] = _df["smiles"].apply(_preprocess_smiles)
    _df = _df[_df["smiles"].notna()]
    # remove duplicated mol
    _df = _exc_dup_mol(_df, task_name)
    # data split
    _df = _split_data(_df, 0.2)
    _df.index.name = 'idx'
    _df = _df.reset_index()

    df = pd.DataFrame(
        {"idx": [], "smiles": [], task_name: [], "group": [], "img_path": []}
    )
    for _, _df_row in _df.iterrows():
        _df_row_tmp = _generate_image(_df_row, task_name)
        df = pd.concat([df, pd.DataFrame([_df_row_tmp])], axis=0)

    df.to_pickle(output_path)
    _save_split_csvs(df, task_name, "")

    print("End Creating Data")
