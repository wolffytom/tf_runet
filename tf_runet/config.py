import argparse
import numpy as np
import os

cwd = os.getcwd()
parser = argparse.ArgumentParser()

# train parameters
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--optimizer', type=str, default='Adam')

args = parser.parse_args()

# additional parameters