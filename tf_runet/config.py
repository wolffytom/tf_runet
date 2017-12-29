import argparse
import numpy as np
import os

cwd = os.getcwd()
parser = argparse.ArgumentParser()

# train parameters
parser.add_argument('--learning_rate', type=float, default=0.002)
parser.add_argument('--optimizer', type=str, default='Adam')

parser.add_argument('--regularizer', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.5)

args = parser.parse_args()

# additional parameters