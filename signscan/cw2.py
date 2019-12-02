from enum import Enum
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from signscan.cli import signscan, load_data
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

#from sklearn.cross_validation import cross_val_score
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score, cross_val_predict # Import cross_val_score function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix

from sklearn.externals.six import StringIO
from sklearn import preprocessing

class EnumType(click.Choice):
    def __init__(self, enum):
        self._enum = enum
        super().__init__([e.value for e in enum])

    def convert(self, value, param, ctx):
        return self._enum(super().convert(value, param, ctx))

class TrainingType(Enum):
    CROSS_VALIDATION = "cross-validation"
    TRAIN_TEST = "train-test"

def detailed_accuracy(images, labels, predicted_labels):
    print("running confusion_matrix for TP Rate and FP Rate...")
    cm = confusion_matrix(labels, predicted_labels)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    # labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    #Precision, Recall and, F Measure
    print("")
    print("running Precision, Recall and, F Measure...")
    class_precision = metrics.precision_score(labels, predicted_labels, average=None)
    print('Precision for Each Class:', class_precision)
    print('Mean Precision:', metrics.precision_score(labels, predicted_labels, average='micro'))
    class_recall = metrics.recall_score(labels, predicted_labels, average=None)
    print('Recall for Each Class:', class_recall)
    print('Mean Recall:', metrics.recall_score(labels, predicted_labels, average='micro'))
    class_f_measure = metrics.f1_score(labels, predicted_labels, average=None)
    print('F Measure for Each Class:', class_f_measure)
    print('Mean F Measure:', metrics.f1_score(labels, predicted_labels, average='micro'))
    # to generate the plot
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

    #Misc Metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(labels, predicted_labels))
    print('Mean Squared Error:', metrics.mean_squared_error(labels, predicted_labels))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(labels, predicted_labels)))


@signscan.command()
@click.option("--train-type", type=EnumType(TrainingType), required=True)
@click.pass_context
def decisiontree_j48(ctx, train_type: TrainingType):
    """
    Dercision Tree Classifier - J48 Algorithm == C45Algorithm.
    """
    print("loading data...")
    train_images, train_labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])

    #to switch between cross-validation and train-test Classifier
    if train_type is TrainingType.CROSS_VALIDATION:
        pass
    elif train_type is TrainingType.TRAIN_TEST:
        test_images, test_labels = load_data("./cw2", shuffle_seed=ctx.obj["seed"])
        # Perform traim-test split
        # 100% 70% training and 30% test (Task 5, 6 and 7)
        train_images, split_images, train_labels, split_labels = train_test_split(train_images, train_labels, test_size=0.7, random_state=42)
        test_images = test_images.append(split_images)
        test_labels = test_labels.append(split_labels)

    print("")
    print("Running Decision Tree J48: KFold...")
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=100, min_samples_leaf=75)
    clf.fit(train_images, train_labels)

    if train_type is TrainingType.CROSS_VALIDATION:

        print("")
        print("Making KFold CROSS_VALIDATION:")
        print("J48 Parameters: Moved data", clf.min_samples_split, ", min samples leaf: ", clf.min_samples_leaf)
        # Perform 10-fold cross validation
        scores = cross_val_score(clf, train_images, train_labels, cv=10)
        predicted_labels = clf.predict(train_images)
        #Visualisation trees
        print("making visualisation of a tree...")
        #for tree_in_forest in clf.estimators_:
        #dot_data = tree.export_graphviz(clf, out_file=None)
        #graph = graphviz.Source(dot_data)
        #graph.render(f'{"tree_"}')
        print("running detailed accuracy on KFold...")
        detailed_accuracy(train_images, train_labels, predicted_labels)
        print('Accuracy: ', clf.score(train_images, train_labels))
    elif train_type is TrainingType.TRAIN_TEST:
        print("")
        print("Make Train-Test:")
        print("J48 Parameters: Moved data", clf.min_samples_split, ", min samples leaf: ", clf.min_samples_leaf)
        predicted_labels = clf.predict(test_images)
        print("running detailed accuracy on Train-Test...")
        detailed_accuracy(test_images, test_labels, predicted_labels)
        print('Accuracy: ', clf.score(test_images, test_labels))

"""
    def tree_model_for_j48():
        #Visualisation trees
        print("making visualisation of a tree...")
        #for tree_in_forest in clf.estimators_:
        dot_data = tree.export_graphviz(clf, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(f'{"tree_"}')
"""
