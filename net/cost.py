import tensorflow as tf

def get_cost(logits, labels, n_class, marks, regularizer, cfg):
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are: 
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """
    with tf.variable_scope('cost', reuse = tf.AUTO_REUSE):
        flat_logits = tf.reshape(logits, [-1, n_class])
        flat_labels = tf.reshape(labels, [-1, n_class])
        
        class_accuracy_list = []
        labels_split = tf.split(flat_labels,n_class,axis=1)
        correct_prediction = tf.cast(tf.equal(tf.argmax(flat_logits, axis=1), tf.argmax(flat_labels, axis=1)), tf.float32)
        for i_class in range(n_class):
            labels_split[i_class] = tf.reshape(labels_split[i_class], [-1])
            i_class_correct = tf.cast(tf.tensordot(labels_split[i_class], correct_prediction, axes=1), tf.float32)
            i_class_times = tf.cast(tf.reduce_sum(labels_split[i_class]), tf.float32)
            i_class_accuracy = i_class_correct / i_class_times
            class_accuracy_list.append(tf.reshape(i_class_accuracy, [1]))
        class_accuracy = tf.concat(class_accuracy_list,axis = 0)
        total_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if cfg.cost_name == "cross_entropy":
            lossmap = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,labels=flat_labels)

            if cfg.use_class_weights:
                class_weights = cfg.class_weights
                class_weights = tf.constant(
                    np.array(class_weights, dtype=np.float32))

                weight_map = tf.multiply(flat_labels, class_weights)
                weight_map = tf.reduce_sum(weight_map, axis=1)

                lossmap = tf.multiply(lossmap, weight_map)

            if cfg.use_mark:
                flat_marks = tf.reshape(marks, [-1])
                lossmap = tf.tensordot(lossmap, flat_marks, axes=1)
                loss = tf.reduce_sum(lossmap) / tf.reduce_sum(flat_marks)

            else:
                loss = tf.reduce_mean(lossmap)

        elif cost_name == "cross_entropy_with_class_ave_weights":
            classes_distrib_inv = 1 / tf.reduce_sum(flat_labels, axis=0)
            classes_weights = classes_distrib_inv / tf.reduce_sum(classes_distrib_inv)
            weight_map = tf.reduce_sum(flat_labels * classes_weights, axis=1)

            loss_map = tf.nn.softmax_cross_entropy_with_logits(
                logits=flat_logits, labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)

            loss = tf.reduce_mean(weighted_loss)

        else:
            raise ValueError("Unknown cost function: " % cost_name)

        if cfg.regularizer:
            #reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            #reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
            #loss += reg_term
            l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
            loss += l2_loss * cfg.regularizer_scale
            
        return loss, class_accuracy, total_accuracy


