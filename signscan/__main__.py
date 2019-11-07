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

    print(f"average accuracy: {sum(analysis.correct_count / analysis.total_count for analysis in label_classifiers.values()) / len(label_classifiers) * 100:.2f}%")

    print("")
    print("mistaken classifications:")
    most_mistaken = calculate_most_mistaken_heatmap(label_classifiers, labels)
    plt.imshow(most_mistaken, cmap='hot')
    plt.title("Which signs are most frequently mislabeled as another?")
    plt.xlabel("mistaken label")
    plt.ylabel("actual label")

    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        plt.savefig(os.path.join(save_plot, "mislabeled.png"))

    if show_plot:
        plt.show()

    for label, analysis in label_classifiers.items():
        most_mistaken_label = Counter(label_mapping[x] for x in labels.loc[analysis.mistake_indices].label)
        n_most_mistaken = sorted(most_mistaken_label.items(), key=lambda x: x[1], reverse=True)[:3]
        most_mistaken = ", ".join(f"{key} ({count})" for key, count in n_most_mistaken)
        print(f" - mistaken with {click.style(label, fg='green')}: {click.style(most_mistaken, fg='bright_black')}")

    print("")
    print("10 most frequently influential features:")
    counter = Counter(itertools.chain.from_iterable(x.top_features[:10] for x in label_classifiers.values()))
    for key, count in itertools.islice(sorted(counter.items(), key=lambda x: x[1], reverse=True), 10):
        print(f" - {key % 48}x{key // 48} (top feature {count} times)")


def calculate_most_mistaken_heatmap(label_classifiers: Dict[str, pandas.DataFrame], labels) -> pandas.DataFrame:
    """
    Generates a 2d table describing how many times each column is mistaken for a given index.
    :param label_classifiers: A number of classifiers over the labels.
    :param labels: A DataFrame which matches each image index with an index into label_mapping.
    :return: A DataFrame table.
    """
    return pandas.DataFrame(
        data=(
            Counter(label_mapping[x] for x in labels.loc[y.mistake_indices].label)
            for y in label_classifiers.values()
        ),
        columns=label_classifiers.keys(),
        index=label_classifiers.keys(),
        dtype=int
    ).fillna(value=0)


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

    # dictionary mapping subsets of n features to the analyses generated from them
    feature_analyses = {}
    with click.progressbar(range(1, n + 1)) as bar:
        for n in bar:
            top_n_pixels = set(itertools.chain.from_iterable(x.top_features[:n] for x in label_classifiers.values()))
            feature_analyses[n] = fit_labels(x_train[(str(x) for x in top_n_pixels)], y_train)

    # for each of the analyses get the a pair of the (average accuracy, index)
    average_data = (
        (sum(y.correct_count / y.total_count for y in x.values()) / len(x), y)
        for y, x in feature_analyses.items()
    )

    average_accuracy = pandas.DataFrame(data=average_data, columns=["prediction accuracy", "number of features"])

    print("")
    print("accuracy for 2, 5, and 10 top features per label:")
    for label in label_mapping:
        features = " / ".join(
            f"{x} features {100 * feature_analyses[x][label].correct_count / feature_analyses[x][label].total_count:.2f}%"
            for x in (2, 5, 10)
        )
        print(f" - {label} / {features}")

    average_accuracy.plot(kind='scatter', x='number of features', y='prediction accuracy')
    plt.title("Accuracy using n top correlating features for each label")

    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        path = os.path.join(save_plot, "feature_accuracy.png")
        plt.savefig(path)
        print("")
        print("saved figure to " + path)

    if show_plot:
        plt.show()


@signscan.command()
@click.pass_context
def count_samples(ctx):
    """
    Outputs the number of samples for each discovered label.
    """
    print("loading data...")
    x_train, y_train, _ = load_data(ctx.obj["data_folder"])

    save_plot = ctx.obj["save_plot"]
    show_plot = ctx.obj["show_plot"]

    label_classifiers = fit_labels(x_train, y_train)

    print("")
    print("enumerated sample counts:")
    for key, frame in y_train._asdict().items():
        print(f" - {key}: {frame[frame.label==0].shape[0]}")
    print("total: ", len(x_train))

    plt.scatter(
        x=[frame[frame.label==0].shape[0] for frame in y_train._asdict().values()],
        y=[c.correct_predictions/c.total_predictions for c in label_classifiers.values()]
    )
    plt.xlabel("number of samples")
    plt.ylabel("prediction accuracy")

    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        plt.savefig(os.path.join(save_plot, "sample_size_correlation.png"))

    if show_plot:
        plt.show()


if __name__ == "__main__":
    signscan()
