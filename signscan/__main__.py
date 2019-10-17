import os
import re
from collections import namedtuple
from os import path
from sys import argv
from typing import Tuple

import pandas
from sklearn import naive_bayes
from sklearn.utils import column_or_1d

label_mapping = [
    "limit_60", "limit_80", "limit_80_lifted",
    "right_of_way_crossing", "right_of_way", "give_way", "stop",
    "no_speed_limit", "turn_right_down", "turn_left_down"
]

YTrain = namedtuple("YTrain", label_mapping)


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


if __name__ == "__main__":
    print("loading data")
    x_train, y_train = load_data(argv[1], shuffle=False)

    #
    # Naive Bayesian Classification and Deeper Analysis
    # https://github.com/arlyon/dmml/issues/3
    # https://github.com/arlyon/dmml/issues/4
    #

    print("performing bayesian classification:")
    for label, df in y_train._asdict().items():
        bayes = naive_bayes.BernoulliNB()
        bayes.fit(x_train, column_or_1d(df))

        df["prediction"] = bayes.predict(x_train)
        df["correct"] = df["label"] == df["prediction"]

        correct_count, total_count = sum(df["correct"]), len(df)
        print(
            f" - accuracy for label {label}: {correct_count} out of "
            f"{total_count} ({correct_count / total_count * 100:.2f}%)"
        )

        pixel_likelihood = pandas.DataFrame(data=bayes.feature_log_prob_[1]).sort_values(0)[0:10]
        print(f"   most correlated pixels: {list(pixel_likelihood.index)}")
