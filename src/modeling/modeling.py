import tensorflow_hub as hub
import autokeras as ak
from typing import List

from keras import regularizers
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Input, InputSpec
from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    ReLU,
    Add,
    GlobalAveragePooling2D,
    Dense,
    Input,
)
import tensorflow_models as tfm
from official.vision.modeling import backbones

def create_convnextbase_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
) -> Model:
    
    # with strategy.scope():
    model = keras.applications.convnext.ConvNeXtBase(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(model.output)
    if is_regression:
        x = Dense(1)(x)
    else:
        x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_efficientnetb0_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
) -> Model:
    
    # with strategy.scope():
    model = keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(model.output)
    if is_regression:
        x = Dense(1)(x)
    else:
        x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)
    return model

def create_densenet121_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
    # strategy: tf.distribute.Strategy,
    do_rate: float | None = None,
    l1: float | None = None,
) -> Model:
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    model = keras.applications.densenet.DenseNet121(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(model.output)
    if is_regression:
        if do_rate:
            x = Dropout(do_rate)(x)
        if l1:
            x = Dense(1, kernel_regularizer=regularizers.L1(l1))(x)
        else:
            x = Dense(1)(x)
    else:
        if do_rate:
            x = Dropout(do_rate)(x)
        if l1:
            x = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L1(l1))(
                x
            )
        else:
            x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_densenet201_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
) -> Model:
    
    # with strategy.scope():
    model = keras.applications.densenet.DenseNet201(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(model.output)
    if is_regression:
        x = Dense(1)(x)
    else:
        x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_resnet152_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
) -> Model:
    
    # with strategy.scope():
    model = keras.applications.resnet_v2.ResNet152V2(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(model.output)
    if is_regression:
        x = Dense(1)(x)
    else:
        x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_resnet18_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
) -> Model:

    def basic_block(x, filters, stride=1):
        shortcut = x

        x = Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)

        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x
    
    inputs = Input(shape=input_shape)

    # conv1
    x = Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # conv2_x
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    x = basic_block(x, 64, stride=1)
    x = basic_block(x, 64, stride=1)

    # conv3_x
    x = basic_block(x, 128, stride=2)
    x = basic_block(x, 128, stride=1)

    # conv4_x
    x = basic_block(x, 256, stride=2)
    x = basic_block(x, 256, stride=1)

    # conv5_x
    x = basic_block(x, 512, stride=2)
    x = basic_block(x, 512, stride=1)

    x = GlobalAveragePooling2D()(x)

    if is_regression:
        outputs = Dense(1)(x)
    else:
        outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss=loss_alg, metrics=metrics)

    return model

def create_resnet50_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
    # strategy: tf.distribute.Strategy,
    do_rate: float | None = None,
    l1: float | None = None,
) -> Model:
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    model = keras.applications.resnet50.ResNet50(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = GlobalAveragePooling2D()(model.output)
    if do_rate:
            x = Dropout(do_rate)(x)

    if is_regression:
        if l1:
            x = Dense(1, kernel_regularizer=regularizers.L1(l1))(x)
        else:
            x = Dense(1)(x)
    else:
        if l1:
            x = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L1(l1))(
                x
            )
        else:
            x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_vgg16_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
    # strategy: tf.distribute.Strategy,
    do_rate: float | None = None,
    l1: float | None = None,
) -> Model:
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # with strategy.scope():
    model = keras.applications.vgg16.VGG16(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = Flatten()(model.output)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    if is_regression:
        if do_rate:
            x = Dropout(do_rate)(x)
        if l1:
            x = Dense(1, kernel_regularizer=regularizers.L1(l1))(x)
        else:
            x = Dense(1)(x)
    else:
        if do_rate:
            x = Dropout(do_rate)(x)
        if l1:
            x = Dense(1, activation="sigmoid", kernel_regularizer=regularizers.L1(l1))(
                x
            )
        else:
            x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_vgg19_model(
    input_shape: tuple[int, int, int],
    loss_alg: str,
    metrics: List,
    is_regression: bool,
) -> Model:
    model = keras.applications.vgg19.VGG19(
        include_top=False, weights=None, input_shape=input_shape
    )
    x = Flatten()(model.output)
    x = Dense(4096, activation="relu")(x)
    x = Dense(4096, activation="relu")(x)
    if is_regression:
        x = Dense(1)(x)
    else:
        x = Dense(1, activation="sigmoid")(x)
    model = Model(model.inputs, x)

    return model

def create_vit_model(input_shape=(256, 256, 3), loss_alg="mean_squared_error", metrics=None, is_regression=True):
    inputs = tf.keras.Input(shape=input_shape)
    
    x = tf.keras.layers.Resizing(224, 224)(inputs)
    x = tf.keras.layers.Rescaling(1./255)(x)
    
    vit_layer = hub.KerasLayer("https://tfhub.dev/sayakpaul/vit_b16_fe/1", trainable=False)
    x = vit_layer(x)
    
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(1, name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ViT_ChemAutoVision")
    return model
    """
    TensorFlow Model GardenのVision Transformerを使用してモデルを作成
    
    Args:
        input_shape: 入力画像のサイズ (height, width, channels)
        loss_alg: 損失関数
        metrics: 評価指標のリスト
        is_regression: 回帰タスクかどうか
        patch_size: パッチサイズ
        hidden_size: 隠れ層のサイズ
        num_layers: Transformerレイヤー数
        num_heads: アテンションヘッド数
        mlp_dim: MLP次元数
        dropout_rate: ドロップアウト率
        representation_size: 中間表現層のサイズ (0なら中間層なし)
        num_classes: 分類クラス数（分類の場合）または出力次元数（回帰の場合）
        
    Returns:
        コンパイル済みVision Transformerモデル
    """
    
    # print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # with strategy.scope():
    
    # transformer_config = tfm.vision.configs.backbones.Transformer(
    #     num_layers=num_layers,
    #     num_heads=num_heads,
    #     mlp_dim=mlp_dim,
    #     dropout_rate=dropout_rate,
    # )
    
    # vit_config = tfm.vision.configs.backbones.VisionTransformer(
    #     pooler='token',  # CLSトークンを使用
    #     representation_size=representation_size, 
    #     hidden_size=hidden_size,
    #     patch_size=patch_size,
    #     transformer=transformer_config,
    #     init_stochastic_depth_rate=0.0,
    #     original_init=True,
    #     output_encoded_tokens=True,
    #     output_2d_feature_maps=False,
    # )

    input_specs = InputSpec(shape=(None,) + input_shape)

    vit_config = {
        'input_specs': input_specs,
        'pooler': 'token',  # CLSトークンを使用
        'representation_size': representation_size, 
        'hidden_size': hidden_size,
        'patch_size': patch_size,
        'mlp_dim': mlp_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'attention_dropout_rate': attention_do_rate,
        'dropout_rate': do_rate,
        'init_stochastic_depth_rate': 0.0,
        'original_init': True,
        'pos_embed_shape': None,
        'output_encoded_tokens': True,
        'output_2d_feature_maps': False,
        'layer_scale_init_value': 0.0,
        'transformer_partition_dims': None,
    }
    
    inputs = Input(shape=input_shape, name="image")
    # vit_config_dict = vit_config.as_dict()
    # vit_config_dict.pop("model_name", None) 
    # backbone = backbones.VisionTransformer.from_config(vit_config_dict)
    backbone = backbones.VisionTransformer(**vit_config)
    backbone.build(input_shape=(None,) + input_shape)
    features = backbone(inputs)
    
    # CLSトークンを取得。ここ大丈夫？
    if isinstance(features, dict):
        x = features['pre_logits']  # 辞書からpre_logitsテンソルを取得
    else:
        x = features

    x = tf.squeeze(x, axis=[1, 2])
    print(f"Features shape: {x.shape}")
    
    if is_regression:
        outputs = Dense(1, name="regression_head")(x)
    else:
        outputs = Dense(1, activation="sigmoid", name="classification_head")(x)
    
    model = Model(inputs, outputs, name="vision_transformer")
    model.summary()
    
    return model
