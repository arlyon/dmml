import os
import re
from os import path

import click
import numpy
import pandas
from collections import namedtuple
from typing import Tuple

label_mapping = [
    "limit_60", "limit_80", "limit_80_lifted",
    "right_of_way_crossing", "right_of_way", "give_way", "stop",
    "no_speed_limit", "turn_right_down", "turn_left_down"
]

YTrain = namedtuple("YTrain", label_mapping)


def load_data(folder: str, *, shuffle=True, shuffle_seed=None) -> Tuple[pandas.DataFrame, YTrain, pandas.DataFrame]:
    """
    Loads the data from the provided folder.
    We have chosen not to shrink the data set.

    :param folder: The directory to load from.
    :param shuffle: Whether to shuffle the data.
    :param shuffle_seed: The seed to use when generating.
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
        random = numpy.random.RandomState()
        if shuffle_seed is not None:
            random.seed(shuffle_seed)

        shuffled_indices = random.permutation(x_train.index)
        x_train = x_train.reindex(shuffled_indices)
        all_labels = all_labels.reindex(shuffled_indices)
        for key, y in y_train.items():
            y_train[key] = y.reindex(shuffled_indices)

    return x_train, YTrain(**y_train), all_labels


@click.group()
@click.argument("data_folder")
@click.option('--seed', help='The random seed. This program is deterministic, and so the seed must be set.', default=0)
@click.option('--save-plot', help='The folder to output plots to.', default=None)
@click.option('--show-plot', help='Whether to show plots.', is_flag=True)
@click.pass_context
def signscan(ctx, data_folder, seed, save_plot, show_plot):
    """Tool for demonstrating the various analyses required for coursework 1."""

    # store data so that other commands can use it
    ctx.ensure_object(dict)
    ctx.obj["data_folder"] = data_folder
    ctx.obj["seed"] = seed
    ctx.obj["save_plot"] = save_plot
    ctx.obj["show_plot"] = show_plot


@signscan.command()
@click.pass_context
def count_samples(ctx):
    """
    Outputs the number of samples for each discovered label.
    """
    print("loading data...")
    x_train, y_train, _ = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    print("")
    print("enumerated sample counts:")
    for key, frame in y_train._asdict().items():
        print(f" - {key}: {frame[frame.label==0].shape[0]}")
    print("total: ", len(x_train))
