def calc_auc(pred, label):
    res = {}
    if type(label) is not list:
        pred = pred.reshape((-1))
        label = label.reshape((-1))
    assert len(label) == len(pred)
    positions = []
    for i in range(len(label)):
        positions.append((pred[i], label[i]))
    positions.sort()
    sum_pos, num_pos, num_neg = 0., 0., 0.
    for i in range(len(label)):
        if positions[i][1] > 0:
            sum_pos += i+1
            num_pos += 1
    num_neg = len(label) - num_pos
    res['pos'] = num_pos
    res['neg'] = num_neg
    if num_pos * num_neg == 0:
        res['auc'] = 1
    else:
        res['auc'] =  (sum_pos - (num_pos) * (num_pos + 1) / 2) / float(num_pos * num_neg)
    return res

def print_step_auc(predict, labels):
    assert predict.shape == labels.shape
    batch_size, step, size_x, size_y = predict.shape
    res = []
    for istep in range(step):
        stepauc = calc_auc(predict[:,istep,:,:], labels[:,istep,:,:])
        res.append(stepauc['auc'])
        print ('step: %5d, auc: %f, pos:%f, neg:%f' % (istep, stepauc['auc'],
            stepauc['pos'], stepauc['neg']))
    return res


if __name__ == '__main__':
    print (calc_auc([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1])['auc'])
