import argparse
import numpy as np

parser = argparse.ArgumentParser()

# data parameters
parser.add_argument('--channels', type=int, default=3)
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--use_mark', type=bool, default=False)
parser.add_argument('--zeromark_percentage', type=float, default=1)
parser.add_argument('--use_max_size', type=bool, default=True)
parser.add_argument('--max_size_x', type=int, default=300)
parser.add_argument('--max_size_y', type=int, default=300)
parser.add_argument('--norm_input', type=bool, default=True)
parser.add_argument('--norm_input_minus', type=bool, default=True)

# net architecture parameters
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--features_root', type=int, default=16)
parser.add_argument('--cnn_kernel_size', type=int, default=3)
parser.add_argument('--pool_size', type=int, default=2)
parser.add_argument('--LSTM', type=bool, default=True)
parser.add_argument('--regularizer_scale', type=float, default=0.01)

# model parameters
parser.add_argument('--base_net_size', type=int, default=100)

# cost architecture parameters
parser.add_argument('--cost_name', type=str, default='log')
parser.add_argument('--regularizer', type=bool, default=True)
parser.add_argument('--class_weights', type=list, default=[1,1])
parser.add_argument('--use_class_weights', type=bool, default=False)

# train parameters
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--keep_prob', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--max_step', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=0.002)

parser.add_argument('--useGPU', type=bool, default=True)

# print for debug
parser.add_argument('--print_marks_distribution', type=bool, default=True)
parser.add_argument('--print_batchidx', type=bool, default=False)
parser.add_argument('--print_dataloading', type=bool, default=False)

cfg = parser.parse_args()
# additional parameters
