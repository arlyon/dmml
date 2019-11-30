from enum import Enum

import click
import tensorflow

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
@click.pass_context
def neural_net(ctx, training_type: TrainingType, train_elements_move: int):

    print("loading data...")
    x_train, y_train, labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    if training_type is TrainingType.CROSS_VALIDATION:


    elif training_type is TrainingType.TRAIN_TEST:
        output = process_data(x_train, y_train, labels, train_elements_move)

    linear_classifier = tensorflow.estimator.LinearClassifier()

