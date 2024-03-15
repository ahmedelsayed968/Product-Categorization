import datetime
import uuid

import tensorflow as tf
from dataset import ImageDataSet, ImageProcessor
from modelling import model_v2, run
from tensorflow import keras

dataset_args = {
    "path": "/kaggle/input/amazon-products-images-v2",
    "train_size": 0.8,
    "val_size": 0.1,
    "test_size": 0.1,
}


dataset = ImageDataSet(**dataset_args)
train_split_args = {"batch_size": 16, "width": 224, "height": 224}
train, val, test = dataset.get_train_val_test(**train_split_args)

epochs = 50
learning_rate = 0.001
model = model_v2(
    len(ImageProcessor.CLASS_NAMES),
    train_split_args["width"],
    train_split_args["height"],
)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
loss = keras.losses.CategoricalCrossentropy()
metrics = [
    tf.keras.metrics.CategoricalCrossentropy(name="categorical_crossentropy"),
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision1", top_k=1),
    tf.keras.metrics.Precision(name="precision3", top_k=3),
    tf.keras.metrics.Recall(name="recall1", top_k=1),
    tf.keras.metrics.Recall(name="recall3", top_k=3),
    tf.keras.metrics.F1Score(average="macro", name="f1_score"),
]

project = "Slash"
id_ = f"V2-{str(uuid.uuid4())}"
save_dir, checkpoint_filename = "./model", "model-" + str(datetime.datetime.now())
config_logs = {"project": project, "id": id_}

compile_config = {"optimizer": optimizer, "loss": loss, "metrics": metrics}
wandb_config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": ImageProcessor.BATCH_SIZE,
    "architecture": "ModelV4",
    "dataset": "V2",
}
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# hub_token = user_secrets.get_secret("HF")
# hub_model_id = f"{config['architecture']}-{config['dataset']}-{config['epochs']}"
history1 = run(
    model,
    epochs,
    train,
    val,
    config_logs,
    checkpoint_filename,
    save_dir,
    compile_config,
    wandb_config,
)
