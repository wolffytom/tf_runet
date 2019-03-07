import tensorflow as tf

def get_cost(logits, labels, regularizer, cfg):
    with tf.variable_scope('cost', reuse = tf.AUTO_REUSE):
        flat_logits = tf.reshape(logits, [-1])
        flat_labels = tf.reshape(labels, [-1])
        loss = tf.losses.log_loss(flat_labels, flat_logits)

        if cfg.regularizer:
            #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            #loss += reg_term
            l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
            loss += l2_loss * cfg.regularizer_scale
            
        return loss
