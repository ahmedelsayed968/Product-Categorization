import pathlib
from typing import Tuple

# from tensorflow. import Dataset
import numpy as np
import tensorflow as tf


class ImageProcessor:
    IMG_WIDTH, IMG_HEIGHT = 224, 224
    BATCH_SIZE = 32
    CLASS_NAMES = None
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    @classmethod
    def get_label(cls, file_path: pathlib.Path):
        assert ImageProcessor.CLASS_NAMES is not None
        parts = tf.strings.split(file_path, "/")
        return tf.cast(parts[-2] == ImageProcessor.CLASS_NAMES, tf.float32)

    @classmethod
    def decode_img(cls, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return tf.image.resize(
            img, [ImageProcessor.IMG_WIDTH, ImageProcessor.IMG_HEIGHT]
        )

    @classmethod
    def process_path(cls, file_path: pathlib.Path):
        label = ImageProcessor.get_label(file_path)
        img = tf.io.read_file(file_path)
        img = ImageProcessor.decode_img(img)
        return img, label

    @classmethod
    def prepare_for_training(cls, ds: tf.data.Dataset, shuffle_buffer_size=1000):

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.batch(ImageProcessor.BATCH_SIZE)
        ds = ds.prefetch(buffer_size=ImageProcessor.AUTOTUNE)
        return ds


class ImageDataSet:
    def __init__(
        self, path: str, train_size: float, test_size: float, val_size: float
    ) -> None:
        self.path = path
        self.data_dir = pathlib.Path(path)
        self.dataset_size = len(list(self.data_dir.glob("*/*.jpg")))
        self.list_ds = tf.data.Dataset.list_files(str(self.data_dir / "*/*"))
        self.train = None
        self.test = None
        self.val = None
        self.train_sz = int(train_size * self.dataset_size)
        self.test_sz = int(test_size * self.dataset_size)
        self.val_sz = int(val_size * self.dataset_size)
        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob("*")])

    def get_train_val_test(self, batch_size, width, height) -> Tuple[tf.data.Dataset]:
        ImageProcessor.CLASS_NAMES = self.CLASS_NAMES
        ImageProcessor.IMG_WIDTH, ImageProcessor.IMG_HEIGHT = width, height
        ImageProcessor.BATCH_SIZE = batch_size
        if not self.train:
            self.train = self.list_ds.take(self.train_sz).map(
                ImageProcessor.process_path
            )
            self.train = ImageProcessor.prepare_for_training(self.train)

        remaining = self.list_ds.skip(self.train_sz)
        if not self.val:
            self.val = remaining.take(self.val_sz).map(ImageProcessor.process_path)
            self.val = ImageProcessor.prepare_for_training(self.val)

        if not self.test:
            self.test = remaining.skip(self.val_sz).map(ImageProcessor.process_path)
            self.test = ImageProcessor.prepare_for_training(self.test)
        return self.train, self.val, self.test
