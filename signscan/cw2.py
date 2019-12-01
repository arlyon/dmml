from enum import Enum

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
    tf.debugging.set_log_device_placement(True)

    print("loading data...")
    train_images, train_labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    if train_type is TrainingType.CROSS_VALIDATION:
        pass
    elif train_type is TrainingType.TRAIN_TEST:
        # todo(arlyon) this
        test_images, test_labels = load_data("./cw2", shuffle_seed=ctx.obj["seed"])

    feature_columns = (tf.feature_column.numeric_column(x, dtype=tf.uint8) for x in train_images.columns.values)
    linear_est = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        n_classes=10,
        model_dir=model_dir,
    )

    linear_est.train(tf.compat.v1.estimator.inputs.pandas_input_fn(
        train_images, train_labels,
        1024, 1, True
    ))

    print("success")