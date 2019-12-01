import click
import numpy as np
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from enum import Enum

from sklearn.model_selection import train_test_split

from signscan.cli import signscan, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import tree

@signscan.command()
@click.pass_context
def cw2(ctx):
    """
    Naive Bayesian Classification and Deeper Analysis.

    - https://github.com/arlyon/dmml/issues/3
    - https://github.com/arlyon/dmml/issues/4
    """

    print("loading data...")
    x_train, y_train, labels = load_data(ctx.obj["data_folder"])


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
@click.pass_context
def randomforest(ctx, train_type: TrainingType):
    """
    Running Random Forest Classifier on the data
    """

    print("loading data...")

    x_train, y_train = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    # to switch between cross-validation and train-test Classifier
    if train_type is TrainingType.CROSS_VALIDATION:
        pass
    elif train_type is TrainingType.TRAIN_TEST:
        x_test, y_test = load_data("./cw2", shuffle_seed=ctx.obj["seed"])

    x_train, x_split, y_train, y_split = train_test_split(x_train, y_train, test_size=0.3, random_state=42)

    x_test = x_test.append(x_split)
    y_test = y_test.append(y_split)

    print("Running Random Forest...")

    clf = RandomForestClassifier(oob_score=True, n_estimators=15, min_samples_split=50, min_samples_leaf=75)

    clf.fit(x_train, y_train)

    print("Random Forest Parameters: n_estimators=", len(clf.estimators_), ", min_samples_split=", clf.min_samples_split, ", min_samples_leaf=", clf.min_samples_leaf)

    predicted_labels = clf.predict(x_test)

    '''
    TP and FP Rate
    '''

    cm = confusion_matrix(y_test, predicted_labels)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')

    # labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')

    plt.show()

    '''
    Precision, Recall and, F Measure
    '''

    class_precision = metrics.precision_score(y_test, predicted_labels, average=None)

    print('Precision for Each Class:', class_precision)
    print('Mean Precision:', metrics.precision_score(y_test, predicted_labels, average='micro'))

    class_recall = metrics.recall_score(y_test, predicted_labels, average=None)

    print('Recall for Each Class:', class_recall)
    print('Mean Recall:', metrics.recall_score(y_test, predicted_labels, average='micro'))

    class_f_measure = metrics.f1_score(y_test, predicted_labels, average=None)

    print('F Measure for Each Class:', class_f_measure)
    print('Mean F Measure:', metrics.f1_score(y_test, predicted_labels, average='micro'))

    plt.plot(class_precision, label='Class Precision')
    plt.plot(class_recall, label='Class Recall')
    plt.plot(class_f_measure, label='Class F Measure')
    plt.legend()
    plt.grid()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.title("Metric Data for each Class")
    plt.xlabel("Class")
    plt.ylabel("Value")
    plt.show()

    '''
    ROC area
    '''



    '''
    Misc Metrics
    '''

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predicted_labels))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predicted_labels))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted_labels)))

    print('Accuracy on Training Set: ', clf.score(x_train, y_train))
    print('OOB Score: ', clf.oob_score_)
    print('Accuracy on Test Set: ', clf.score(x_test, y_test))

    '''
    Visualise trees in forest
    '''
    i_tree = 0
    for tree_in_forest in clf.estimators_:
        dot_data = tree.export_graphviz(tree_in_forest, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(f'{"tree_"}{i_tree}')
        i_tree = i_tree + 1
