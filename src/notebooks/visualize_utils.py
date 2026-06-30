import autokeras as ak
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from typing import Optional, Literal
import tensorflow as tf
from tensorflow import keras

def find_feature_layer(model):
    """
    全結合層（Dense）の直前のレイヤーを見つける。
    ネストされた Functional モデルにも対応。
    """

    def _find_dense_recursive(m):
        for i, layer in enumerate(m.layers):
            # Dense層を見つけたら、その直前のレイヤーを返す
            if isinstance(layer, keras.layers.Dense):
                if i > 0:
                    prev_layer = m.layers[i - 1]
                    return prev_layer.name
                else:
                    raise ValueError("Dense layer is the first layer - no previous layer.")
            # ネストされた Functional モデルにも潜る
            if hasattr(layer, "layers"):
                found = _find_dense_recursive(layer)
                if found:
                    return found
        return None

    layer_name = _find_dense_recursive(model)
    if layer_name is None:
        print("No Dense layer found. Using second-to-last layer as fallback.")
        if len(model.layers) >= 2:
            return model.layers[-2].name
        else:
            return model.layers[-1].name

    print(f"Found Dense layer → using previous layer: {layer_name}")
    return layer_name


def load_autokeras_model(model_path):
    """
    AutoKerasのh5形式のモデルを読み込み、特徴抽出用のモデルを作成
    
    Args:
        model_path: h5モデルファイルのパス
    
    Returns:
        original_model: 元のモデル
    """
    print(f"Loading AutoKeras model from {model_path}...")
    
    # AutoKerasのカスタムオブジェクトを定義
    original_model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS, compile=False)

    print("\n=== Model Structure Analysis ===")
    print(f"Number of inputs: {len(original_model.inputs)}")
    print(f"Input shapes: {[inp.shape for inp in original_model.inputs]}")
    print(f"Input names: {[inp.name for inp in original_model.inputs]}")

        # レイヤー接続の確認
    print("\n=== Layer Connections ===")
    for i, layer in enumerate(original_model.layers[:10]):  # 最初の10レイヤーのみ表示
        print(f"Layer {i}: {layer.name} ({type(layer).__name__})")
        if hasattr(layer, 'input_spec') and layer.input_spec:
            print(f"  Input spec: {layer.input_spec}")
        try:
            print(f"  Input shape: {layer.input_shape}")
            print(f"  Output shape: {layer.output_shape}")
        except:
            print("  Shape info unavailable")
    
    # モデル構造を表示
    print("\nModel Summary:")
    original_model.summary()
    return original_model

def extract_model_path(
    results_csv_path: str,  
    run_id: str
):
    df_results = pd.read_csv(results_csv_path)
    result_row = df_results[df_results["Run ID"]==run_id]
    original_path = result_row.model_path.iloc[0]
    
    if original_path.startswith("./logs"):
        model_path = original_path.replace("./logs", "../../logs")
    else:
        model_path = original_path.replace("../models", "../../models")
        
    return model_path

def get_best_run_id(df: pd.DataFrame, 
                   target: str, 
                   metric: str, 
                   direction: Literal['max', 'min'] = 'min',
                   model_name: Optional[str] = None, 
                   is_daug: Optional[bool] = False,
                   is_weighted_balanced: Optional[bool] = False, 
                   ) -> Optional[str]:

    # NaN や 'None' 文字列を統一しておく
    df["model_name"] = df["model_name"].replace(["None", "nan", "NaN"], None)

    required_cols = ['Run ID', 'target', 'img_size', metric]
    if 'model_name' in df.columns:
        required_cols.append('model_name')
    if 'hp_tuning' in df.columns:
        required_cols.append('hp_tuning')
    if 'automl' in df.columns:
        required_cols.append('automl')
    if 'execute_data_aug' in df.columns:
        required_cols.append('execute_data_aug')
    if 'weighted_balanced' in df.columns:
        required_cols.append('weighted_balanced')

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必要な列が見つかりません: {missing_cols}")
    print(f"model_name: {model_name}")

    daug_column_name = "execute_data_aug"
        
    # === 条件分岐 ===
    if model_name is None:
        # model_name が None のときのみ hp_tuning_filter を使う
        filtered_df = df[
            (df['target'] == target) &
            (df['hp_tuning'] == (hp_tuning_filter or 'grid_search')) & 
            (df[metric].notna())
        ]

    else:
        if is_daug:
            daug_cond = df["execute_data_aug"] == True
        else:
            daug_cond = df["execute_data_aug"].isna() | df["execute_data_aug"] == False

        if model_name == "ImageMol":
            automl_cond = df["automl"].isna() | (df["automl"] == False)
            # ImageMolのデータにはimg_size=224のものも混ざっているため絞り込む
            filtered_df = df[
                (df['model_name'] == model_name) &
                (df['target'] == target) &
                automl_cond &
                (df['img_size'] == "256") &
                (df['weighted_balanced'] == is_weighted_balanced) &
                (df[metric].notna())
            ]

        elif model_name in ("autokeras", "resnet18"):
            filtered_df = df[
                (df['model_name'] == model_name) &
                (df['target'] == target) &
                daug_cond &
                (df[metric].notna())
            ]
        else:
            filtered_df = df[
                (df['model_name'] == model_name) &
                (df['target'] == target) &
                (df[metric].notna())
            ]

    # === 該当なしのチェック ===
    if filtered_df.empty:
        print("❌ 該当データが見つかりません:")
        print(f"  model_name: {model_name}")
        print(f"  target: {target}")
        # print(f"  hp_tuning: {hp_tuning_filter}")
        print(f"  metric: {metric}")
        return None

    filtered_df[metric] = filtered_df[metric].astype(str).str.strip("'")
    try:
        filtered_df[metric] = filtered_df[metric].astype(float)
    except ValueError as e:
        invalid_vals = filtered_df.loc[
            ~filtered_df[metric].astype(str).str.match(r"^-?\d+(\.\d+)?$")
        ][metric].unique()
        raise ValueError(f"`r2` に変換できない値があります: {invalid_vals}") from e

    # === 最適化方向で選択 ===
    if direction == 'max':
        best_row = filtered_df.loc[filtered_df[metric].idxmax()]
    elif direction == 'min':
        best_row = filtered_df.loc[filtered_df[metric].idxmin()]
    else:
        raise ValueError("direction は 'max' または 'min' のみ指定可能です")

    # === 結果出力 ===
    print(f"✅ 最適なRun ID: {best_row['Run ID']}")
    print(f"   model_name: {best_row['model_name']}")
    print(f"   target: {best_row['target']}")
    # print(f"   hp_tuning: {best_row.get('hp_tuning', None)}")
    print(f"   {metric}: {best_row[metric]}")

    return best_row['Run ID']
