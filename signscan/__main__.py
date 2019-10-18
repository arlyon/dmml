import itertools
import os
import re
from collections import namedtuple
from dataclasses import dataclass
from os import path
from typing import Tuple, Dict, Optional

import click
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn import naive_bayes
from sklearn.base import BaseEstimator
from sklearn.utils import column_or_1d

label_mapping = [
    "limit_60", "limit_80", "limit_80_lifted",
    "right_of_way_crossing", "right_of_way", "give_way", "stop",
    "no_speed_limit", "turn_right_down", "turn_left_down"
]

YTrain = namedtuple("YTrain", label_mapping)


class CommonPixels(list):
    def pixel_coords(self):
        return [f'{x % 48}x{x // 48}' for x in self]

    def __getitem__(self, y):
        return CommonPixels(list.__getitem__(self, y))


@dataclass
class BayesAnalysis:
    classifier: BaseEstimator
    correct_predictions: int
    total_predictions: int
    top_features: CommonPixels
    heat_map: Optional[numpy.ndarray]


def load_data(folder: str, *, shuffle=True) -> Tuple[pandas.DataFrame, YTrain]:
    """
    Loads the data from the provided folder.
    We have chosen not to shrink the data set.

    :param folder: The directory to load from.
    :param shuffle: Whether to shuffle the data.
    :returns:
        A tuple containing the images, and a YTrain mapping the images to labels..
        x_train is a dataframe of 12660 images and the greyscale values of its pixel data (normalized).
        y_train is a set of dataframes that associate an given image with a label.
    """

    x_train = None
    y_train = {}

    with click.progressbar(os.listdir(folder)) as bar:
        for file in bar:
            if re.match(".+([0-9]).csv", file):
                y_train[label_mapping[int(file[-5])]] = pandas.read_csv(
                    path.join(folder, file), true_values="0", false_values="1",
                    names=["label"], header=0
                )
            elif "x_train" in file:
                x_train = pandas.read_csv(path.join(folder, file))

    if shuffle:
        x_train.sample(frac=1)
        for x in y_train.__dict__.values():
            x.sample(frac=1)

    return x_train, YTrain(**y_train)


def bayesian_classification(train: pandas.DataFrame, labels: pandas.DataFrame, n_correlated=10) -> BayesAnalysis:
    classifier = naive_bayes.MultinomialNB()
    classifier.fit(train, column_or_1d(labels))

    labels = labels.copy()
    labels["prediction"] = classifier.predict(train)
    labels["correct"] = labels["label"] == labels["prediction"]

    correct_count, total_count = sum(labels["correct"]), len(labels)

    success_probability = classifier.coef_[0]

    # get most correlated features for each label
    top_features = CommonPixels(numpy.argsort(success_probability)[:n_correlated])

    # generate heatmap of correlation
    heat_map = numpy.reshape(success_probability, (-1, 48)) if len(success_probability) == 48**2 else None

    return BayesAnalysis(classifier, correct_count, total_count, top_features, heat_map)


@click.command()
@click.option('--save_plot', help='The folder to output plots to.', default=None)
@click.option('--show_plot', help='Whether to show plots.', is_flag=True)
@click.argument("data_folder")
def cmd(data_folder, save_plot, show_plot):
    """Coursework one tool."""

    print("loading data...")
    x_train, y_train = load_data(data_folder, shuffle=False)

    #
    # Naive Bayesian Classification and Deeper Analysis
    # https://github.com/arlyon/dmml/issues/3
    # https://github.com/arlyon/dmml/issues/4
    #

    print("running bayesian classification on all features...")
    label_classifications: Dict[str, BayesAnalysis] = {
        label: bayesian_classification(x_train, frame, n_correlated=len(x_train.columns))
        for label, frame in y_train._asdict().items()
    }

    for label, analysis in label_classifications.items():
        accuracy = f"{analysis.correct_predictions} out of {analysis.total_predictions} ({analysis.correct_predictions / analysis.total_predictions * 100:.2f}%)"
        print(f" - accuracy for label {click.style(label, fg='green')}: {click.style(accuracy, fg='bright_black')}")
        print(f"   {click.style(str(len(analysis.top_features[:10])), fg='yellow')} most correlated pixels: {click.style(', '.join(analysis.top_features[:10].pixel_coords()), fg='bright_black')}")

        plt.imshow(analysis.heat_map, cmap='hot', interpolation='lanczos')
        plt.title("Heatmap for " + label)

        if show_plot:
            plt.show()

        if save_plot is not None:
            os.makedirs(save_plot, exist_ok=True)
            plt.savefig(os.path.join(save_plot, label + ".png"))

    print(f" - average accuracy: {sum(analysis.correct_predictions / analysis.total_predictions for analysis in label_classifications.values()) / len(label_classifications) * 100:.2f}%")

    #
    # Attempt to improve classification and make conclusions
    # https://github.com/arlyon/dmml/issues/5
    #

    print("building accuracy graph over features ordered by correlation...")
    feature_analyses = []
    with click.progressbar(range(1, 20)) as bar:
        for n in bar:
            top_n_pixels = set(itertools.chain.from_iterable(x.top_features[:n] for x in label_classifications.values()))
            limited_label_classifications: Dict[str, BayesAnalysis] = {
                label: bayesian_classification(x_train[(str(x) for x in top_n_pixels)], frame)
                for label, frame in y_train._asdict().items()
            }
            feature_analyses.append((limited_label_classifications, n))

    data = zip(
        (y for _, y in feature_analyses),
        (sum(y.correct_predictions / y.total_predictions for y in x.values()) / len(x) for x, _ in feature_analyses),
    )

    accuracy = pandas.DataFrame(data=data, columns=["number of features", "prediction accuracy"])
    accuracy.plot(kind='scatter', x='number of features', y='prediction accuracy')
    plt.title("Accuracy using n top correlating features for each label")

    if show_plot:
        plt.show()

    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        plt.savefig(os.path.join(save_plot, "feature_accuracy.png"))


if __name__ == "__main__":
    cmd()
