import datetime
import os
from typing import Any, Dict

import tensorflow as tf
import wandb
from keras import callbacks
from keras.models import Model, model_from_json
from tensorflow import keras
from tensorflow.keras import layers


def model_v1(num_classes: int, width: int, height: int, trainable: bool = False):
    # Load Base Model
    base_model = keras.applications.ResNet50V2(
        include_top=False,  # Exclude ImageNet classifier at the top
        weights="imagenet",
        input_shape=(width, height, 3),
    )
    # Freeze all parameters of the base model
    base_model.trainable = trainable
    inputs = keras.Input(shape=(width, height, 3))
    # Apply specific pre-processing function for ResNet v2
    x = keras.applications.resnet_v2.preprocess_input(inputs)
    # Keep base model batch normalization layers in inference mode (instead of training mode)
    x = base_model(x, training=False)
    # Rebuild top layers
    x = layers.GlobalAveragePooling2D()(x)  # Average pooling operation
    x = layers.BatchNormalization()(x)  # Introduce batch norm
    x = layers.Dropout(0.2)(x)  # Regularize with dropout

    # Flattening to final layer - Dense classifier with 37 units (multi-class classification)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def model_v2(num_classes: int, width: int, height: int, trainable: bool = False):
    # Load Base Model
    base_model = keras.applications.ResNet50V2(
        include_top=False,  # Exclude ImageNet classifier at the top
        weights="imagenet",
        input_shape=(width, height, 3),
    )
    # Freeze all parameters of the base model
    base_model.trainable = trainable
    inputs = keras.Input(shape=(width, height, 3))
    # Apply specific pre-processing function for ResNet v2
    x = keras.applications.resnet_v2.preprocess_input(inputs)
    # Keep base model batch normalization layers in inference mode (instead of training mode)
    x = base_model(x, training=False)
    # Rebuild top layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)  # Regularize with dropout

    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)  # Regularize with dropout

    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)  # Regularize with dropout

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def model_v3(num_classes: int, width: int, height: int, trainable: bool = False):
    # Load Base Model
    base_model = keras.applications.VGG19(
        include_top=False,  # Exclude ImageNet classifier at the top
        weights="imagenet",
        input_shape=(width, height, 3),
    )
    # Freeze all parameters of the base model
    base_model.trainable = trainable
    inputs = keras.Input(shape=(width, height, 3))
    # Apply specific pre-processing function for ResNet v2
    x = keras.applications.vgg19.preprocess_input(inputs)
    # Keep base model batch normalization layers in inference mode (instead of training mode)
    x = base_model(x, training=False)
    # Rebuild top layers
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def model_V4(num_classes: int, width: int, height: int, trainable: bool = False):
    # Load Base Model
    base_model = keras.applications.VGG19(
        include_top=False,  # Exclude ImageNet classifier at the top
        weights="imagenet",
        input_shape=(width, height, 3),
    )
    # Freeze all parameters of the base model
    base_model.trainable = trainable
    inputs = keras.Input(shape=(width, height, 3))
    # Apply specific pre-processing function for ResNet v2
    x = keras.applications.vgg19.preprocess_input(inputs)
    # Keep base model batch normalization layers in inference mode (instead of training mode)
    x = base_model(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)  # Average pooling operation
    x = layers.BatchNormalization()(x)  # Introduce batch norm
    x = layers.Dropout(0.2)(x)  # Regularize with dropout
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)  # Introduce batch norm
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)  # Regularize with dropout

    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def run(
    model: tf.keras.Model,
    epochs: int,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    config_logs: Dict[str, Any],
    checkpoint_filename: str,
    save_dir: str,
    compile_config: Dict[str, Any],
    wandb_config: Dict,
    hub_token: str,
    hub_model_id: str,
):

    run = wandb.init(
        sync_tensorboard=True, reinit=True, **config_logs, config=wandb_config
    )
    model.compile(**compile_config)
    earlystopping = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True
    )

    # Persist the model as checkpoint
    checkpoint_dir = os.path.join(save_dir, checkpoint_filename)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, checkpoint_filename + "model.keras"),
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
    )
    #     #Push to Hub Callback
    #     model_push_to_hub = PushToHubCallback(output_dir=checkpoint_dir,
    #                                             save_strategy= "epoch",
    #                                             hub_model_id = hub_model_id,
    #                                             hub_token= hub_token,
    #                                             checkpoint = True
    #                                             )
    # Tensorboard Tracking run Callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=[earlystopping, tensorboard_callback, checkpoint],
    )
    run.finish()
    return history


def save_model(path: str, model: Model) -> None:

    model_yaml = model.to_json()
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, "model.json"), "w") as json_file:
        json_file.write(model_yaml)
    model.save_weights(os.path.join(path, "model.weights.h5"))
    print("Saved model to disk")


def load_model(path_json: str, path_weights: str) -> Model:
    json_file = open(path_json, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    return loaded_model
