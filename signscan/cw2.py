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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score # Import cross_val_score function
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

    # to change the testing size of the dataset
    #train_images, split_images, train_labels, split_labels = train_test_split(x_train, y_train, test_size=1, random_state=42)
    #test_images = test_images.append(split_images)
    #test_labels = test_labels.append(split_labels)

    print("")
    print("Running Decision Tree J48...")
    clf = DecisionTreeClassifier(random_state=0, min_samples_split=100, min_samples_leaf=75)
    clf.fit(train_images, train_labels)
    print("J48 Parameters: min_samples_split=", clf.min_samples_split, ", min_samples_leaf=", clf.min_samples_leaf)

    print("")
    print("Task 3: KFold CROSS_VALIDATION:")
    predicted_labels = clf.predict(train_images)
    # TP Rate and FP Rate for K-Fold Cross Val
    '''
    TP Rate and FP Rate

    print("running confusion_matrix for TP Rate and FP Rate...")
    #kfold split???
    cm = confusion_matrix(train_labels, predicted_labels)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g')
    # labels and title
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    plt.show()
    '''
    '''
    Precision, Recall and, F Measure

    print("")
    print("running Precision, Recall and, F Measure...")
    class_precision = metrics.precision_score(train_labels, predicted_labels, average=None)
    print('Precision for Each Class:', class_precision)
    print('Mean Precision:', metrics.precision_score(train_labels, predicted_labels, average='micro'))
    class_recall = metrics.recall_score(train_labels, predicted_labels, average=None)
    print('Recall for Each Class:', class_recall)
    print('Mean Recall:', metrics.recall_score(train_labels, predicted_labels, average='micro'))

    class_f_measure = metrics.f1_score(train_labels, predicted_labels, average=None)

    print('F Measure for Each Class:', class_f_measure)
    print('Mean F Measure:', metrics.f1_score(train_labels, predicted_labels, average='micro'))

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
    '''
    ROC area
    '''



    '''
    Misc Metrics
    '''

    print('Mean Absolute Error:', metrics.mean_absolute_error(train_labels, predicted_labels))
    print('Mean Squared Error:', metrics.mean_squared_error(train_labels, predicted_labels))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(train_labels, predicted_labels)))

    print('Accuracy on Training Set: ', clf.score(train_images, train_labels))
    print("Accuracy on KFold CROSS_VALIDATION: ",metrics.accuracy_score(train_labels, predicted_labels))
    print('Accuracy on Test Set: ', clf.score(train_images, train_labels))

    '''
    Visualise Decision Trees
    '''
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render(f'{"tree_"}')



    #print("")
    #print("Task 5: TRAIN_TEST:")
    #predicted_labels = clf.predict(test_images)
    #print("Accuracy:",metrics.accuracy_score(test_labels, predicted_labels))

    #score = cross_val_score(predicted_labels, test_labelscv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #scores = cross_val_score(clf, train_images, train_labels, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


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
