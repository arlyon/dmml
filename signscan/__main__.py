import itertools
import os
import re
from collections import namedtuple, Counter
from dataclasses import dataclass
from os import path
from typing import Tuple, Dict, Optional, List

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
    total_count: int
    correct_count: int
    top_features: CommonPixels
    heat_map: Optional[numpy.ndarray]
    mistake_indices: List[int]  # photos that were mistaken for another

    @property
    def mistake_count(self):
        return self.total_count - self.correct_count

    @property
    def correct_indices(self):
        return [x for x in range(self.total_count) if x not in self.mistake_indices]


def load_data(folder: str, *, shuffle=True) -> Tuple[pandas.DataFrame, YTrain, pandas.DataFrame]:
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
    all_labels = None

    with click.progressbar(os.listdir(folder)) as bar:
        for file in bar:
            if re.match(".+([0-9]).csv", file):
                frame = pandas.read_csv(path.join(folder, file), names=["label"], header=0)
                frame["label"] = frame["label"] == 0  # convert to bool
                y_train[label_mapping[int(file[-5])]] = frame
            elif "x_train" in file:
                x_train = pandas.read_csv(path.join(folder, file))
            elif "y_train" in file:
                all_labels = pandas.read_csv(path.join(folder, file), names=["label"], header=0)

    if shuffle:
        shuffled_indices = numpy.random.permutation(x_train.index)
        x_train = x_train.reindex(shuffled_indices)
        all_labels = all_labels.reindex(shuffled_indices)
        for key, y in y_train.items():
            y_train[key] = y.reindex(shuffled_indices)

    return x_train, YTrain(**y_train), all_labels


def fit_labels(x_train, y_train: YTrain) -> Dict[str, BayesAnalysis]:
    """For each label, create a bayes classifier for it."""
    return {
        label: bayesian_classification(x_train, frame, n_correlated=len(x_train.columns))
        for label, frame in y_train._asdict().items()
    }


def bayesian_classification(train: pandas.DataFrame, labels: pandas.DataFrame, n_correlated=10) -> BayesAnalysis:
    classifier = naive_bayes.MultinomialNB()
    classifier.fit(train, column_or_1d(labels))

    labels = labels.copy()  # make a copy to avoid editing the given labels
    labels["prediction"] = classifier.predict(train).astype(bool)
    labels["correct"] = labels["label"] == labels["prediction"]

    correct_count, total_count = sum(labels["correct"]), len(labels)

    # get the various probabilities from the classifier
    success_probability = classifier.coef_[0]

    # signs most confused with this one
    mistaken_photos = labels[(labels.prediction == True) & (labels.correct == False)].index

    # get most correlated features for each label
    top_features = CommonPixels(numpy.argsort(success_probability)[:n_correlated])

    # generate heat map of correlation when dealing with the full image
    heat_map = numpy.reshape(success_probability, (-1, 48)) if len(success_probability) == 48 ** 2 else None

    return BayesAnalysis(classifier, total_count, correct_count, top_features, heat_map, mistaken_photos)


@click.group()
@click.argument("data_folder")
@click.option('--save-plot', help='The folder to output plots to.', default=None)
@click.option('--show-plot', help='Whether to show plots.', is_flag=True)
@click.pass_context
def signscan(ctx, data_folder, save_plot, show_plot):
    """Tool for demonstrating the various analyses required for coursework 1."""

    # store data so that other commands can use it
    ctx.ensure_object(dict)
    ctx.obj["data_folder"] = data_folder
    ctx.obj["save_plot"] = save_plot
    ctx.obj["show_plot"] = show_plot


@signscan.command()
@click.pass_context
def bayes_simple(ctx):
    """
    Naive Bayesian Classification and Deeper Analysis.

    - https://github.com/arlyon/dmml/issues/3
    - https://github.com/arlyon/dmml/issues/4
    """

    print("loading data...")
    x_train, y_train, labels = load_data(ctx.obj["data_folder"])

    print("")
    print("running bayesian classification on all features...")

    save_plot = ctx.obj["save_plot"]
    show_plot = ctx.obj["show_plot"]

    label_classifiers = fit_labels(x_train, y_train)

    for label, analysis in label_classifiers.items():
        accuracy = f"{analysis.correct_count} out of {analysis.total_count} ({analysis.correct_count / analysis.total_count * 100:.2f}%)"
        print(f" - {click.style(label, fg='green')}: {click.style(accuracy, fg='bright_black')}")
        print(f"   {click.style(str(len(analysis.top_features[:10])), fg='yellow')} most correlated pixels: {click.style(', '.join(analysis.top_features[:10].pixel_coords()), fg='bright_black')}")

        plt.imshow(analysis.heat_map, cmap='hot', interpolation='lanczos')
        plt.title("Heatmap for " + label)

        if save_plot is not None:
            os.makedirs(save_plot, exist_ok=True)
            plt.savefig(os.path.join(save_plot, label + ".png"))

        if show_plot:
            plt.show()

        

    print(f" - average accuracy: {sum(analysis.correct_predictions / analysis.total_predictions for analysis in label_classifiers.values()) / len(label_classifiers) * 100:.2f}%")


@signscan.command()
@click.option("-n", default=10, help="How many of the n most correlated values to graph.")
@click.pass_context
def bayes_complex(ctx, n):
    """
    Improve bayesian classification and make conclusions.

    - https://github.com/arlyon/dmml/issues/5
    """

    print("loading data...")
    x_train, y_train, _ = load_data(ctx.obj["data_folder"])

    print("")
    print(f"building accuracy graph over {n} features sorted by correlation...")

    save_plot = ctx.obj["save_plot"]
    show_plot = ctx.obj["show_plot"]

    label_classifiers = fit_labels(x_train, y_train)

    feature_analyses = []
    with click.progressbar(range(1, n + 1)) as bar:
        for n in bar:
            top_n_pixels = set(itertools.chain.from_iterable(x.top_features[:n] for x in label_classifiers.values()))
            limited_label_classifications = fit_labels(x_train[(str(x) for x in top_n_pixels)], y_train)
            feature_analyses.append((limited_label_classifications, n))

    # for each of the analyses get the a pair of the (average accuracy, index)
    data = (
        (sum(y.correct_count / y.total_count for y in x.values()) / len(x), y)
        for x, y in feature_analyses
    )

    accuracy = pandas.DataFrame(data=data, columns=["prediction accuracy", "number of features"])
    accuracy.plot(kind='scatter', x='number of features', y='prediction accuracy')
    plt.title("Accuracy using n top correlating features for each label")

    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        path = os.path.join(save_plot, "feature_accuracy.png")
        plt.savefig(path)
        print("")
        print("saved figure to " + path)

    if show_plot:
        plt.show()

if __name__ == "__main__":
    signscan()
