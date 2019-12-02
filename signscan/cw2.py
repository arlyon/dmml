import hashlib
import os.path as path
import pickle
import os
from enum import Enum

import tensorflow as tf
import tensorflow.keras as k
import click
import pandas
from sklearn.model_selection import StratifiedKFold

from signscan.cli import signscan, load_data


class EnumType(click.Choice):
    def __init__(self, enum):
        self._enum = enum
        super().__init__([e.value for e in enum])

    def convert(self, value, param, ctx):
        return self._enum(super().convert(value, param, ctx))


class ClassifierType(Enum):
    LINEAR = "linear"
    MULTILAYER = "multilayer"


class TestType(Enum):
    KFOLD = "kfold"
    TRAINTEST = "train-test"


def f_score(y_true, y_pred):
    from tensorflow.keras.backend import sum, round, clip, epsilon

    true_positives = sum(round(clip(y_true * y_pred, 0, 1)))
    possible_positives = sum(round(clip(y_true, 0, 1)))
    predicted_positives = sum(round(clip(y_pred, 0, 1)))

    recall = true_positives / (possible_positives + epsilon())
    precision = true_positives / (predicted_positives + epsilon())
    return 2 * ((precision * recall) / (precision + recall + epsilon()))


@signscan.group()
@click.option("--classifier", type=EnumType(ClassifierType), required=True)
@click.option("--model-cache", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option("--batch-size", type=int, default=32)
@click.pass_context
def neural_net(ctx, classifier: ClassifierType, model_cache: str, batch_size: int):
    """Analysis with varied-depth neural networks."""
    ctx.obj["classifier"] = classifier
    ctx.obj["model_dir"] = model_cache
    ctx.obj["batch_size"] = batch_size


@neural_net.command()
@click.option("--splits", help="number of groups to split the data into", type=int, default=10)
@click.pass_context
def kfold(ctx, splits):
    """Train the model using k-fold method on one data set."""
    print("loading data...")
    images, labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    print("")
    print(f"running k-fold with 10 folds on a {ctx.obj['classifier'].value} model...")
    scores = pandas.DataFrame()
    for fold, (train_indices, test_indices) in enumerate(
        StratifiedKFold(n_splits=splits, random_state=ctx.obj["seed"]).split(images, labels)):
        print(f" - training fold {fold+1}")
        train_images = images.iloc[train_indices]
        train_labels = labels.iloc[train_indices]
        test_images = images.iloc[test_indices]
        test_labels = k.utils.to_categorical(labels.iloc[test_indices])
        model, hist = build_model(ctx, train_images, train_labels, kfold.name, batch_size=ctx.obj["batch_size"])

        print(f"   evaluating fold {fold+1}")
        data = dict(zip(model.metrics_names, model.evaluate(test_images, test_labels, batch_size=ctx.obj["batch_size"],
                                     verbose=ctx.obj["verbosity"] > 1)))

        scores = scores.append(data, ignore_index=True)

    print("")
    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
        print(scores)


@neural_net.command()
@click.argument("test_dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option("--train-data-offset", type=int, default=0)
@click.pass_context
def train_test(ctx, test_dir, train_data_offset: int):
    """Train the model using two training and testing data sets."""
    print("loading data...")
    train_images, train_labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])
    test_images, test_labels = load_data(test_dir, shuffle_seed=ctx.obj["seed"])

    train_images, train_labels, test_images, test_labels = move_data(train_data_offset, train_images, train_labels,
                                                                     test_images, test_labels)

    print("")
    print(f"training {ctx.obj['classifier'].value} model with {train_data_offset} moved...")
    model, hist = build_model(ctx, train_images, train_labels, train_test.name, batch_size=ctx.obj["batch_size"])

    test_images = (test_images / 255)
    test_labels = k.utils.to_categorical(test_labels)

    print("")
    print("evaluating model...")
    for key, value in zip(model.metrics_names, model.evaluate(test_images, test_labels, batch_size=ctx.obj["batch_size"], verbose=ctx.obj["verbosity"] > 1)):
        print(f" - {key}: {value}")


class CacheDigest:
    def __init__(self, images: pandas.DataFrame, labels: pandas.DataFrame, train_strategy: str,
                 classifier: ClassifierType, batch_size: int, epochs: int):
        self._hash = hashlib.sha256()
        self._hash.update(bytearray(pandas.util.hash_pandas_object(images).values))
        self._hash.update(bytearray(pandas.util.hash_pandas_object(labels).values))
        self._hash.update(train_strategy.encode('utf8'))
        self._hash.update(classifier.value.encode('utf8'))
        self._hash.update(batch_size.to_bytes(2, byteorder='big'))
        self._hash.update(epochs.to_bytes(2, byteorder='big'))

        self._train_strategy = train_strategy
        self._classifier = classifier

    @property
    def model(self) -> str:
        return f"{self._hash.digest().hex()}.h5"

    @property
    def hist(self) -> str:
        return f"{self._hash.digest().hex()}.hist"

    @property
    def image(self) -> str:
        return f"{self._train_strategy}-{self._classifier.value}.png"


def build_model(ctx, train_images, train_labels, cache_name, batch_size=32, epochs=10):
    digest = CacheDigest(train_images, train_labels, cache_name, ctx.obj["classifier"], batch_size, epochs)
    model_path = path.join(ctx.obj["model_dir"], digest.model) if ctx.obj["model_dir"] is not None else None
    hist_path = path.join(ctx.obj["model_dir"], digest.hist) if ctx.obj["model_dir"] is not None else None
    model, history = None, None

    if model_path is not None and path.exists(model_path):
        model = k.models.load_model(model_path, custom_objects={"f_score": f_score})

    if hist_path is not None and path.exists(hist_path):
        with open(hist_path, "rb") as file:
            history = pickle.load(file)

    train_images = train_images / 255
    train_labels = k.utils.to_categorical(train_labels)

    if model is None or history is None:
        model = compile_model(train_images, ctx.obj["classifier"])
        history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, shuffle=False).history

        if model_path is not None:
            model.save(model_path)
        if hist_path is not None:
            with open(hist_path, "wb") as file:
                pickle.dump(history, file)

    if ctx.obj["save_plot"] is not None:
        image_path = path.join(ctx.obj["save_plot"], digest.image)
        k.utils.plot_model(model, to_file=image_path)

    return model, history


def compile_model(train_images, classifier) -> k.Sequential:
    model = k.Sequential()
    model.add(k.Input(train_images.shape[1], name="in", dtype=tf.float16))
    if classifier is ClassifierType.LINEAR:
        pass
    elif classifier is ClassifierType.MULTILAYER:
        for x in range(2):
            model.add(k.layers.Dense(units=20, activation="relu", use_bias=True, name=f"layer{x}"))
    else:
        raise Exception("Unhandled classifier type. Please report this error.")
    model.add(k.layers.Dense(units=10, activation="softmax", use_bias=True, name="out"))
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=[
        k.metrics.TruePositives(), k.metrics.FalsePositives(), k.metrics.TrueNegatives(), k.metrics.FalseNegatives(),
        k.metrics.Precision(), k.metrics.Recall(), k.metrics.AUC(), k.metrics.categorical_accuracy, f_score
    ])

    return model


def move_data(train_elements_move, train_images, train_labels, test_images, test_labels):
    if train_elements_move > 0:
        move_train_images = train_images[:train_elements_move]
        train_images = train_images[train_elements_move:]
        test_images = test_images.append(move_train_images)

        move_train_labels = train_labels[:train_elements_move]
        train_labels = train_labels[train_elements_move:]
        test_labels = test_labels.append(move_train_labels)

    return train_images, train_labels, test_images, test_labels
