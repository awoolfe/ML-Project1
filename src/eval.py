# This is where we evaluate the accuracy of our models
# Input:
# target_y : result that we wish to obtain based on real data
# true_y : results obtained by the model


def evaluate_acc(target_y, true_y):
    correct_labels = 0
    if len(target_y) != len(true_y):  # to prevent indexing exceptions
        print("can't compare those sets, not the same size")
        return -1  # return error code
    for i in range(len(target_y)):
        if target_y[i] == true_y[i]:
            correct_labels += 1  # we count how many labels the model got right
    return correct_labels/len(target_y)  # we return the ratio over correct over total
