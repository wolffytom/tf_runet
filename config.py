cfg = {
'channels' : 3,
'use_mark' : False,
'use_max_size' : False,
'max_size_x' : 300,
'max_size_y' : 300,
'norm_input' : True,
'norm_input_minus' : False,

# net architecture parameters
'layers' : 4,
'features_root' : 16,
'cnn_kernel_size' : 3,
'pool_size' : 2,
'LSTM' : True,
'egularizer_scale': 0.1,

# model
'optimizer' : 'Adam',
'base_net_size' : 256,
'batch_size' : 1,
'max_step' : 4,
'learning_rate' : 0.002,
'keep_prob' : 0.8,
'regularizer' : False,
'useGPU' : True,

}
