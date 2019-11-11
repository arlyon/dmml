import itertools
import os
import re
from collections import namedtuple, Counter
from dataclasses import dataclass
from os import path
from typing import Tuple, Dict, Optional, List
from pprint import pprint

import click
import matplotlib.pyplot as plt
import numpy
import pandas
from sklearn import naive_bayes, metrics
from sklearn.base import BaseEstimator
from sklearn.utils import column_or_1d
from sklearn.cluster import KMeans, k_means
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest, VarianceThreshold

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
        numpy.random.seed(seed=42)
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

    result = BayesAnalysis(classifier, total_count, correct_count, top_features, heat_map, mistaken_photos)
    return result

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
@click.option("--sweep-features", is_flag=True, help="Compares various feature selection methods")
@click.option("--sweep-variance", is_flag=True, help="Compares datasets where low variance features have been removed")
@click.option("--sweep-clusters", is_flag=True, help="Compares k-means algorithm with different number of clusters")
@click.pass_context
def k_clustering(ctx, sweep_features, sweep_variance, sweep_clusters):
    """
    K-means clustering function to be run on dataset.
    Includes simple analysis of results.
    """
    save_plot = ctx.obj["save_plot"]
    show_plot = ctx.obj["show_plot"]

    print("loading data...")
    features, boolean_labels, labels = load_data(ctx.obj["data_folder"])
    n_samples, n_features = features.shape
    features_with_labels = features.copy()
    features_with_labels[n_features] = labels

    # Save seed for consistent runs for data analysis
    seed = numpy.random.get_state()


    # Run k-clustering excluding the class attribute
    model = KMeans(n_clusters=10)
    print("running k-means clustering on all features except class...")
    numpy.random.set_state(seed)
    base_predictions = model.fit_predict(features)
    score_clustering(labels, base_predictions, print_score=True)

    # Run k-clustering including the class attribute
    print("running k-means clustering on all features including class...")
    numpy.random.set_state(seed)
    score_clustering(labels, model.fit_predict(features_with_labels), print_score=True)

    # Perform Analytical sweeps of features, variance and clusters
    best_feature_n = None
    if sweep_features:
        best_feature_n = feature_sweep(features, boolean_labels, labels, seed, save_plot, show_plot, n_features=500)

    if sweep_variance:
        variance_sweep(features, labels, seed, save_plot, show_plot, step=50)

    best_cluster_n = None
    if sweep_clusters:
        best_cluster_n = cluster_sweep(features, labels, seed, save_plot, show_plot, n_clusters=50, step=1)

    # Plotting the contingency matrix for the base prediction
    matrix = metrics.cluster.contingency_matrix(column_or_1d(labels), base_predictions)
    plt.imshow(matrix, cmap="hot")
    plt.title("Base Prediction mapping centroids against class labels")
    plt.xlabel("Cluster Centroid Label")
    plt.ylabel("Actual Label")
    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        path = os.path.join(save_plot, "base_prediction_matrix.png")
        plt.savefig(path)
        print("")
        print("saved figure to " + path)
    if show_plot:
        plt.show()
    plt.clf()

    # Running k-clustering with results from sweep analysis
    print("Running k-means clustering with optimal settings")
    model.set_params(n_clusters=28)
    selector = SelectKBest(k=122)
    numpy.random.set_state(seed)
    optimal_predictions = model.fit_predict(selector.fit_transform(features, column_or_1d(labels)))
    score_clustering(labels, optimal_predictions, print_score=True)

    # Plotting contingency matrix from optimal predictions
    matrix = metrics.cluster.contingency_matrix(column_or_1d(labels), optimal_predictions)
    plt.imshow(matrix, cmap="hot")
    plt.title("Optimal Prediction mapping centroids against class labels")
    plt.xlabel("Cluster Centroid Label")
    plt.ylabel("Actual Label")
    if save_plot is not None:
        os.makedirs(save_plot, exist_ok=True)
        path = os.path.join(save_plot, "optimal_prediction_matrix.png")
        plt.savefig(path)
        print("")
        print("saved figure to " + path)
    if show_plot:
        plt.show()
    plt.clf()

    # Print out optimal results from sweep analysis
    if best_feature_n:
        print(f"Ideal number of k-best features is {best_feature_n}.")
    if best_feature_n:
         print(f"Ideal number of clusters is {best_cluster_n}.")
    
    print("Analysis Completed.")
  
def feature_sweep(features, boolean_labels, labels, seed, save_plot, show_plot, n_features=20):
    """
    Performs a sweep of top 'n_features' bayesian features per label, top 'n_features' bayesian features overall
    and the top 'n_features' selected by the k-best selector using the f_classifier function.
    """
    model = KMeans(n_clusters=10)
    
    # Finds the best performing features per label from the naive bayesian net
    print("Performing sweep of top bayesian features per label...")
    bayes_classifiers = fit_labels(features, boolean_labels)
    bayes_per_label_analysis = []
    with click.progressbar(range(n_features//10)) as feature_range:
        for n in feature_range:
            top_n_features = set(itertools.chain.from_iterable(x.top_features[:n+1] for x in bayes_classifiers.values()))
            numpy.random.set_state(seed)
            predictions = model.fit_predict(features[(str(x) for x in top_n_features)])
            bayes_per_label_analysis.append((len(top_n_features), score_clustering(labels, predictions)))

    # Finds the best performing features overall from the naive bayesian net
    print("Performing sweep of top bayesian features overall...")
    bayes_classifier = bayesian_classification(features, labels, n_correlated=n_features + 1)
    bayes_overall_analysis = []
    with click.progressbar(range(1, n_features + 1)) as feature_range:
        for n in feature_range:
            top_n_features = bayes_classifier.top_features[:n]
            numpy.random.set_state(seed)
            predictions = model.fit_predict(features[(str(x) for x in top_n_features)])
            bayes_overall_analysis.append((n, score_clustering(labels, predictions)))

    # Finds the best perfroming features from the K-Best algorithm using the f_classifier scoring function
    print("Performing sweep of K-best features...")
    k_best_analysis = []
    selector = SelectKBest()
    selector.fit(features, column_or_1d(labels))
    with click.progressbar(range(1, n_features + 1)) as feature_range:
        for n in feature_range:
            selector.set_params(k = n)
            selected_features = selector.transform(features)
            numpy.random.set_state(seed)
            predictions = model.fit_predict(selected_features)
            k_best_analysis.append((n, score_clustering(labels, predictions)))
    
    # Plots the data gathered from the sweeps aboveß
    if show_plot or save_plot:
        handles = []
        plot_info = [
            (bayes_per_label_analysis, 'r', "Bayes per Label"),
            (bayes_overall_analysis, 'g', "Bayes Overall"),
            (k_best_analysis, 'b', "K Best")
        ]
        for data, colour, name in plot_info:
            data = list(zip(*[(x, *y.values()) for x, y in data]))
            handles+= plt.plot(data[0], data[3], '-' + colour, label = name+ " V Score")
            handles+= plt.plot(data[0], data[4], '--' + colour, label = name + " Rand")
        plt.legend(handles, loc="lower left")
        plt.xlabel("Number of Features")
        plt.title("Comparison of Feature Selection Algorithms")
        if save_plot is not None:
            os.makedirs(save_plot, exist_ok=True)
            path = os.path.join(save_plot, "feature_sweep.png")
            plt.savefig(path)
            print("")
            print("saved figure to " + path)
        if show_plot:
            plt.show()
        plt.clf()

    # Plots a heatmap of the results of the f_classifier scoring function
    plt.imshow(selector.scores_.reshape(48, -1), cmap='hot', interpolation='lanczos')
    plt.title("K-Best Feature Heatmap")
    if save_plot is not None:
            path = os.path.join(save_plot, "k_best_heatmap.png")
            plt.savefig(path)
            print("")
            print("saved figure to " + path)
    if show_plot:
        plt.show()
    plt.clf()

    # Returns the best performing number of features for the K-best selector
    score = [v_score + rand for _, _, v_score, rand in [scores.values() for (_, scores) in k_best_analysis]]
    score = numpy.argmax(score)
    print(f"Best performance out of {n_features} features: {score}")
    return score      

def variance_sweep(features, labels, seed, save_plot, show_plot, step=500):
    """
    Performs a sweep across the range of variance to establish the value of removing
    low variance pixels
    """
    model = KMeans(n_clusters=10)
    variance_analysis = []
    selector = VarianceThreshold()
    selector.fit(features)

    # Sweeps through variance range
    print("Performing sweep of variance thresholding...")
    # The bounds correspond to approximately 0 features selected and all features selected
    # lower bound 2900
    # upper bound 6450
    with click.progressbar(range(2900, 6450, step)) as variance_range:
        for variance in variance_range:
            selector.set_params(threshold=variance)
            selected_features = selector.transform(features)
            numpy.random.set_state(seed)
            predictions = model.fit_predict(selected_features)
            variance_analysis.append((variance, score_clustering(labels, predictions)))
    
    # Plots results from variance sweep
    if show_plot or save_plot:
        data = list(zip(*[(x, *y.values()) for x, y in variance_analysis]))
        name = "Variance"
        handles = plt.plot(data[0], data[3], '-b', label = name+ " V Score")
        handles += plt.plot(data[0], data[4], '--b', label = name + " Rand")
        plt.legend(handles, loc="lower left")
        plt.xlabel("Variance Threshold for Feature Selection")
        plt.title("Effect of Variance Threshold Feature Selection")
        if save_plot is not None:
            os.makedirs(save_plot, exist_ok=True)
            path = os.path.join(save_plot, "variance_sweep.png")
            plt.savefig(path)
            print("")
            print("saved figure to " + path)
        if show_plot:
            plt.show()
        plt.clf()

    # Plots the variances of each pixel as a heatmap
    plt.imshow(selector.variances_.reshape(48, -1), cmap='hot', interpolation='lanczos')
    plt.title("Heatmap of Variances between Images")
    if save_plot is not None:
            path = os.path.join(save_plot, "variance_heatmap.png")
            plt.savefig(path)
            print("")
            print("saved figure to " + path)
    if show_plot:
        plt.show()
    plt.clf()

    

def cluster_sweep(features, labels, seed, save_plot, show_plot, n_clusters=20, step=1):
    """
    Performs a sweep across different numbers of clusters to determine the optimal number of
    for classification for this data set. 
    """
    cluster_analysis = []
    model = KMeans()

    # Sweeps through the range of numbers of clusters, starting at 10 because of the 10 initial classes.
    print("Performing sweep of clusters...")
    with click.progressbar(range(10, n_clusters + 1, step)) as cluster_range:
        for cluster_size in cluster_range:
            model.set_params(n_clusters = cluster_size)
            numpy.random.set_state(seed)
            predictions = model.fit_predict(features)
            cluster_analysis.append((cluster_size, score_clustering(labels, predictions)))
    
    # Plots the results from the cluster sweep
    data = list(zip(*[(x, *y.values()) for x, y in cluster_analysis]))
    if show_plot or save_plot:
        name = "Cluster"
        handles = plt.plot(data[0], data[3], '-b', label = name+ " V Score")
        handles += plt.plot(data[0], data[4], '--b', label = name + " Rand")
        plt.legend(handles, loc="lower left")
        plt.xlabel("Number of Clusters")
        plt.title("Performance Comparison with K-Clusters")
        if save_plot is not None:
            os.makedirs(save_plot, exist_ok=True)
            path = os.path.join(save_plot, "cluster_sweep.png")
            plt.savefig(path)
            print("")
            print("saved figure to " + path)
        if show_plot:
            plt.show()
        plt.clf()
    
    # Returns the best scoring number of clusters based only on the adjusted_random_score
    # because this score is more accurate for high numbers of clusters.
    score = [rand for _, _, _, rand in [scores.values() for (_, scores) in cluster_analysis]]
    score = numpy.argmax(score) + 10
    print(f"Best performance out of {n_clusters} clusters: {score}")
    return score

def score_clustering(true_labels, predicted_labels, print_score=False):
    """
    Returns the homogeneity, completeness, v_score and adjusted random score for an list of classes
    and a list of predictions.
    """
    score = {}
    score["homogeneity"], score["completeness"], score["v_score"] = metrics.cluster.homogeneity_completeness_v_measure(column_or_1d(true_labels), predicted_labels)
    score["adjusted"] = metrics.cluster.adjusted_rand_score(column_or_1d(true_labels), predicted_labels)
    if print_score:
        print(f"Homogeneity: {score['homogeneity']}\tCompleteness: {score['completeness']}")
        print(f"V Score: {score['v_score']}\t\tAdjusted Rand Score: {score['adjusted']}\n")
    return score


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
        y=[c.correct_count/c.total_count for c in label_classifiers.values()]
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
