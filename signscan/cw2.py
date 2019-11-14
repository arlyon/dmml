import click

from signscan.cli import signscan, load_data


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

    print("")
    print("hello world")


