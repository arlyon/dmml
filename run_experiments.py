#!/usr/bin/env python3

from itertools import product
import sys
import subprocess

cli_template = "pipenv run signscan cw1 neural-net --classifier={} --model-cache=models {}{}"

latex_template = """
\\begin{{figure}}
    \\begin{{Verbatim}}[fontsize=\\scriptsize]
{}
\\end{{Verbatim}}
    \\caption{{Sample Output From The {} {} Neural Net With {}}}
    \\label{{script:{}-{}-{}}}
\\end{{figure}}
"""


figures = []
for (command, arg), classifier in product((*product(("kfold",), (10,)), *product(("train-test",), (0, 4000, 9000))), ("linear", "multilayer")):
    if command == "kfold":
        cli = cli_template.format(classifier, command, f" --splits={arg}")
        description = f"{arg} Folds"
    elif command == "train-test":
        cli = cli_template.format(classifier, command, f" cw2 --train-data-offset={arg}")
        description = f"{arg} Moved To Test Set"
    else:
        sys.exit(1)

    out = subprocess.check_output(cli.split(" "))
    print(latex_template.format(
        out.decode('utf8'), classifier.capitalize(), command.capitalize(), description,
        classifier, command, description.replace(" ", "-").lower()
    ))

