from enum import Enum
import logging

import click
import tensorflow as tf

from signscan.cli import signscan, load_data


class EnumType(click.Choice):
    def __init__(self, enum):
        self._enum = enum
        super().__init__([e.value for e in enum])

    def convert(self, value, param, ctx):
        return self._enum(super().convert(value, param, ctx))


class TrainingType(Enum):
    CROSS_VALIDATION = "cross-validation"
    TRAIN_TEST = "train-test"

@signscan.command()
@click.option("--train-type", type=EnumType(TrainingType), required=True)
@click.option("--train-elements-move", type=int, default=0)
@click.option("--model-dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.pass_context
def neural_net(ctx, train_type: TrainingType, train_elements_move: int, model_dir: str):
    logger = tf.get_logger()
    logger.setLevel(logging.INFO)
    print("loading data...")
    x_train, y_train, labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    if train_type is TrainingType.CROSS_VALIDATION:


        pass

    elif train_type is TrainingType.TRAIN_TEST:
        output = process_data(x_train, y_train, labels, train_elements_move)

    feature_columns = [tf.feature_column.numeric_column(x, dtype=tf.float32) for x in x_train.columns.values]
    linear_est = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        model_dir=model_dir,
    )
    train_input_func = make_input_func(x_train, labels)
    linear_est.train(train_input_func)
    print("success")


def make_input_func(x_train, y_train, epochs=10, batch_size=32, shuffle=True):
    def input_func():
        dataset = tf.data.Dataset.from_tensor_slices((dict(x_train), y_train.values))
        if shuffle:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size).repeat(epochs)
        return dataset
    return input_func
