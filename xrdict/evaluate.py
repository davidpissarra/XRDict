def evaluate(ground_truth, prediction):
    acc_1 = 0.
    acc_10 = 0.
    acc_100 = 0.
    length = len(ground_truth)
    for i in range(length):
        if ground_truth[i] in prediction[i][:100]:
            acc_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                acc_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    acc_1 += 1
    return acc_1/length*100, acc_10/length*100, acc_100/length*100
