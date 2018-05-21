import tensorflow as tf

def get_sess():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    session = tf.InteractiveSession(config=config)
    return session
