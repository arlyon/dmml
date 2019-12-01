from enum import Enum
import click

from signscan.cli import signscan, load_data
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score # Import cross_val_score function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
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

    #to switch between cross-validation and train-test Classifier
    if train_type is TrainingType.CROSS_VALIDATION:
        pass
    elif train_type is TrainingType.TRAIN_TEST:
        test_images, test_labels = load_data("./cw2", shuffle_seed=ctx.obj["seed"])

    print("", train_labels)
    print("running decision tree j48...")
    #clf = DecisionTreeClassifier(random_state=0) # Create Decision Tree


    #print("spliting data into training & testing...")
    # 70% training and 30% test
    #X_train, X_test, Y_train, Y_test = train_test_split(x_train, labels, test_size=0.3, random_state=0)
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train_transformed = scaler.transform(X_train)

    #print("Building Dercision Tree Model using J48 or C45  ")
    #clf = DecisionTreeClassifier(random_state=0) # Create Decision Tree
    #print("cross validation...")
    #scores = cross_val_score(clf, x_train, labels, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #print("train-test validation...")
    #clf = clf.fit(X_train_transformed, y_test) # Train Decision Tree
    #X_test_transformed = scaler.transform(X_test)
    #clf.score(X_test_transformed, y_test)

    #y_pred = clf.predict(labels) # Predict the response for test dataset
    #print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
