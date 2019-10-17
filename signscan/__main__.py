import os
import re
from collections import namedtuple
from os import path
from typing import Tuple, Dict

import click
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn import naive_bayes
from sklearn.utils import column_or_1d

label_mapping = [
    "limit_60", "limit_80", "limit_80_lifted",
    "right_of_way_crossing", "right_of_way", "give_way", "stop",
    "no_speed_limit", "turn_right_down", "turn_left_down"
]

YTrain = namedtuple("YTrain", label_mapping)
BayesAnalysis = namedtuple("BayesAnalysis",
                           ["classifier", "correct_predictions", "total_predictions", "top_features", "heat_map"])


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
    x_train = pandas.read_csv(path.join(folder, "x_train_gr_smpl.csv"))
    files = [x for x in os.listdir(folder) if re.match(".+([0-9]).csv", x)]
    y_train = YTrain(**{
        label_mapping[int(file[-5])]: pandas.read_csv(
            path.join(folder, file), true_values="0", false_values="1",
            names=["label"], header=0
        )
        for file in files
    })

    if shuffle:
        x_train.sample(frac=1)
        for x in y_train.__dict__.values():
            x.sample(frac=1)

    return x_train, y_train


def bayesian_classification(x_train, frame, n_correlated=10) -> BayesAnalysis:
    classifier = naive_bayes.MultinomialNB()
    classifier.fit(x_train, column_or_1d(frame))

    frame["prediction"] = classifier.predict(x_train)
    frame["correct"] = frame["label"] == frame["prediction"]

    correct_count, total_count = sum(frame["correct"]), len(frame)

    success_probability = classifier.coef_[0]

    # get most correlated features for each label
    top_features = numpy.argsort(success_probability)[:n_correlated]
    pixel_likelihood = [f'{x % 48}x{x // 48}' for x in top_features]

    # generate heatmap of correlation
    heat_map = numpy.reshape(success_probability, (-1, 48))

    return BayesAnalysis(classifier, correct_count, total_count, pixel_likelihood, heat_map)


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

    label_classifications: Dict[str, BayesAnalysis] = {}
    with click.progressbar(y_train._asdict().items(), label="performing bayesian classification") as bar:
        for label, frame in bar:
            label_classifications[label] = bayesian_classification(x_train, frame)

    for label, analysis in label_classifications.items():
        accuracy = f"{analysis.correct_predictions} out of {analysis.total_predictions} ({analysis.correct_predictions / analysis.total_predictions * 100:.2f}%)"
        print(f" - accuracy for label {click.style(label, fg='green')}: {click.style(accuracy, fg='bright_black')}")
        print(f"   {click.style(str(len(analysis.top_features)), fg='yellow')} most correlated pixels: {click.style(', '.join(analysis.top_features), fg='bright_black')}")

        plt.imshow(analysis.heat_map, cmap='hot', interpolation='lanczos')
        plt.title("Heatmap for " + label)

        if show_plot:
            plt.show()

        if save_plot is not None:
            os.makedirs(save_plot, exist_ok=True)
            plt.savefig(os.path.join(save_plot, label + ".png"))

    #
    # Attempt to improve classification and make conclusions
    #
    #


if __name__ == "__main__":
    cmd()
