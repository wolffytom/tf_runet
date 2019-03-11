def calc_auc(pred, label):
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
    if num_pos * num_neg == 0:
        return 1
    else:
        return (sum_pos - (num_pos) * (num_pos + 1) / 2) / float(num_pos * num_neg)

if __name__ == '__main__':
    print (calc_auc([0.1, 0.2, 0.3, 0.4, 0.5], [0, 1, 0, 1, 1]))
