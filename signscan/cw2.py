import click
import graphviz
import pydotplus
import os

from signscan.cli import signscan, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import export_graphviz


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


@signscan.command()
@click.pass_context
def randomforest(ctx):
    """
    Running Random Forest Classifier on the data
    """

    print("loading data...")
    x_train, y_train, true_labels = load_data(ctx.obj["data_folder"])

    print("Running Random Forest...")

    clf = RandomForestClassifier(n_estimators=10)
    clf.fit(x_train, true_labels)

    print(clf.predict(x_train))
    print(clf.decision_path(x_train))
    print(clf.apply(x_train))
    print(clf.score(x_train, true_labels))

    i_tree = 0
    for tree_in_forest in clf.estimators_:
        dot_data = tree.export_graphviz(tree_in_forest, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render(f'{"tree_"}{i_tree}')
        i_tree = i_tree + 1
