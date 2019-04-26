cfg = {
'channels' : 3,
'use_mark' : False,
'use_max_size' : False,
'max_size_x' : 128,
'max_size_y' : 128,
'norm_input' : True,
'norm_input_minus' : False,

# net architecture parameters
'layers' : 1,
'features_root' : 16,
'cnn_kernel_size' : 3,
'pool_size' : 2,
'LSTM' : True,
'egularizer_scale': 0.1,

# model
'optimizer' : 'Adam',
'base_net_size' : 128,
'batch_size' : 1,
'max_step' : 30,
'learning_rate' : 0.001,
'keep_prob' : 0.9,
'regularizer' : False,
'useGPU' : True,

}
