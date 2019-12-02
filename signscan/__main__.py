import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from signscan.cli import signscan
from signscan.cw1 import bayes_simple, bayes_complex, k_clustering, em_clustering, agglo_clustering, bayes_tan
from signscan.cw2 import neural_net


if __name__ == "__main__":
    signscan(prog_name="signscan")
