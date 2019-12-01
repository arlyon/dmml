from enum import Enum
import click
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from signscan.cli import signscan, load_data
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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

@signscan.command()
@click.option("--train-type", type=EnumType(TrainingType), required=True)
@click.pass_context
def decisiontree_j48(ctx, train_type: TrainingType):
    """
    Dercision Tree Classifier - J48 Algorithm == C45Algorithm.
    """
    print("loading data...")
    train_images, train_labels = load_data(ctx.obj["data_folder"], shuffle_seed=ctx.obj["seed"])
    assert train_images is not None

    #to switch between cross-validation and train-test Classifier
    if train_type is TrainingType.CROSS_VALIDATION:
        pass
    elif train_type is TrainingType.TRAIN_TEST:
        test_images, test_labels = load_data("./cw2", shuffle_seed=ctx.obj["seed"])

    # to split the data into 70% training and 30% test
    #train_images, split_image, train_labels, split_labels = train_test_split(x_train, y_train, test_size=0.3, random_state=42)


    print("")
    print("running decision tree j48...")
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=100, min_samples_leaf=75)
    clf.fit(train_images, train_labels)

    print("")
    print("Predicting using CROSS_VALIDATION...")
    predicted_labels = clf.predict(train_images)
    print("Accuracy: ",metrics.accuracy_score(train_labels, predicted_labels))
    # TP Rate and FP Rate
    print("running confusion_matrix...")
    cm = confusion_matrix(train_labels, predicted_labels)
    #fp_rate = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    # labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Trained labels')
    ax.set_title('Confusion Matrix')
    plt.show()

    print("")
    print("Predicting using TRAIN_TEST...")
    predicted_labels = clf.predict(test_images)
    print("Accuracy: ",metrics.accuracy_score(test_labels, predicted_labels))

    #score = cross_val_score(predicted_labels, test_labelscv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #scores = cross_val_score(clf, train_images, train_labels, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    print("")
    print("Visualizing Decision Trees...")

    #print("spliting data into training & testing...")
    # 70% training and 30% test
    #X_train, X_test, Y_train, Y_test = train_test_split(x_train, labels, test_size=0.3, random_state=0)
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train_transformed = scaler.transform(X_train)

    #print("train-test validation...")
    #clf = clf.fit(X_train_transformed, y_test) # Train Decision Tree
    #X_test_transformed = scaler.transform(X_test)
    #clf.score(X_test_transformed, y_test)

    #y_pred = clf.predict(labels) # Predict the response for test dataset
    #print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
