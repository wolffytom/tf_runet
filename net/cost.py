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

def calc_auc(pred, label):
    pred = pred.reshape((-1))
    label = label.reshape((-1))
    assert len(pred) == len(label)
    pred_label_tuples = []
    for i in range(len(pred)):
        pred_label_tuples.append((pred[i], label[i]))
    pos_cnt = sum(label)
    neg_cnt = len(label) - pos_cnt
    if pos_cnt == 0 or neg_cnt == 0:
        return -1

    neg_before = 0
    res = []
    for i in range(len(pred)):
        this_label =pred_label_tuples[i][1] 
        if this_label > 0:
            res.append(float(neg_before)/neg_cnt)
        else:
            neg_before += 1
    res = sum(res) / len(res)
    return res

if __name__ == '__main__':
    print (calc_auc([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1]))
