import os
import re
from os import path
import pickle

import click
import numpy
import pandas
from collections import namedtuple
from typing import Tuple

from tensorflow import keras

label_mapping = [
    "limit_60", "limit_80", "limit_80_lifted",
    "right_of_way_crossing", "right_of_way", "give_way", "stop",
    "no_speed_limit", "turn_right_down", "turn_left_down"
]

tf_verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]

YTrain = namedtuple("YTrain", label_mapping)


def load_data(folder: str, *, shuffle=True, shuffle_seed=None) -> Tuple[pandas.DataFrame, pandas.Series]:
    """
    Loads the data from the provided folder.
    We have chosen not to shrink the data set.

    :param folder: The directory to load from.
    :param shuffle: Whether to shuffle the data.
    :param shuffle_seed: The seed to use when generating.
    :returns:
        A tuple of (images, labels)
        images is a dataframe of 12660 images and the greyscale values of its pixel data (normalized).
        labels is a series that associate an given image with a label.
    """

    images = None
    labels = None
    cache = path.join(folder, "data.cache")

    if path.exists(cache):
        with open(cache, "rb") as cache:
            return pickle.load(cache)

    with click.progressbar(os.listdir(folder)) as bar:
        for file in bar:
            if re.match("x_([a-z]+)_gr_smpl.csv", file):
                images = pandas.read_csv(path.join(folder, file), dtype='uint8')
            if re.match("y_([a-z]+)_smpl.csv", file):
                labels = pandas.read_csv(path.join(folder, file), header=0, dtype='int32').iloc[:,0]

    if shuffle:
        random = numpy.random.RandomState()
        if shuffle_seed is not None:
            random.seed(shuffle_seed)

        shuffled_indices = random.permutation(images.index)
        images = images.reindex(shuffled_indices)
        labels = labels.reindex(shuffled_indices)

    data = (images, labels)
    with open(cache, "wb") as cache:
        pickle.dump(data, cache)

    return data


@click.group()
@click.argument("data_folder")
@click.option('--seed', help='The random seed. This program is deterministic, and so the seed must be set.', default=0)
@click.option('--save-plot', help='The folder to output plots to.', default=None, type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('--show-plot', help='Whether to show plots.', is_flag=True)
@click.option('-v', '--verbosity', help="The verbosity of the output.", count=True)
@click.pass_context
def signscan(ctx, data_folder, seed, save_plot, show_plot, verbosity):
    """Tool for demonstrating the various analyses required for coursework 1."""

    # store data so that other commands can use it
    ctx.ensure_object(dict)
    ctx.obj["data_folder"] = data_folder
    ctx.obj["seed"] = seed
    ctx.obj["save_plot"] = save_plot
    ctx.obj["show_plot"] = show_plot
    ctx.obj["verbosity"] = verbosity

    import tensorflow as tf
    tf.get_logger().setLevel(tf_verbosity[min(ctx.obj["verbosity"], 3)])


@signscan.command()
@click.pass_context
def count_samples(ctx):
    """
    Outputs the number of samples for each discovered label.
    """
    print("loading data...")
    images, labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    print("")
    print("enumerated sample counts:")
    for key, arr in list(zip(label_mapping, numpy.transpose(keras.utils.to_categorical(labels)))):
        print(f" - {key}: {int(sum(arr))}")
    print("total: ", len(images))
