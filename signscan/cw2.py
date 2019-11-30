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
@click.option("--model-dir", type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.pass_context
def neural_net(ctx, train_type: TrainingType, train_elements_move: int, model_dir: str):

    print("loading data...")
    x_train, y_train, labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    if train_type is TrainingType.CROSS_VALIDATION:

        pass

    elif train_type is TrainingType.TRAIN_TEST:
        output = process_data(x_train, y_train, labels, train_elements_move)

    dataset = tensorflow.data.Dataset.from_tensor_slices((dict(x_train), labels))
    feature_columns = [tensorflow.feature_column.numeric_column(x, dtype=tensorflow.float32) for x in x_train.columns.values]
    linear_est = tensorflow.estimator.LinearClassifier(
        feature_columns=feature_columns,
        model_dir=model_dir,
    )

    linear_est.train(dataset)

    pass
