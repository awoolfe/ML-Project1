import numpy as np
import csv
import pandas as fb
import matplotlib.pyplot as plt

with open ('../data/winequality-red.csv', 'r') as f:
    wines = list(csv.reader(f, delimiter=';'))
    wines_attributes = wines[0]
    wines = np.array(wines[1:], dtype=np.float)
    for i in wines:
        i[11] = 1 if i[11] > 5 else 0

    reviews = ["Positive", "Negative"]
    good = 0
    bad = 0
    for i in wines:
        if i[11] == 1:
            good += 1
        else:
            bad += 1

    values = [good, bad]

    # plt.figure(figsize=(4, 4))
    # plt.bar(reviews, values)
    # plt.suptitle('Review Distribution')
    # plt.tight_layout(w_pad=1)
    fig, axs = plt.subplots(len(wines[0]), 2, figsize=(15, 60))

    for i, attribute in enumerate(wines_attributes):
        good_wine_attribute = [x for j, x in enumerate(wines[:, i]) if wines[j, -1] == 1]
        bad_wine_attribute = [x for j, x in enumerate(wines[:, i]) if wines[j, -1] == 0]

        axs[i][0].hist(good_wine_attribute)
        axs[i][0].set_title(f'{attribute} good wine')

        axs[i][1].hist(bad_wine_attribute)
        axs[i][1].set_title(f'{attribute} bad wine')

        mean_attribute_bad = np.mean(good_wine_attribute)
        mean_attribute_good = np.mean(bad_wine_attribute)

        std_attribute_bad = np.std(good_wine_attribute)
        std_attribute_good = np.std(bad_wine_attribute)

        print('Attribute: ', attribute, '. mean good wine', attribute, ': ', mean_attribute_bad, ' +- ',
              std_attribute_bad, '\nmean bad wine', attribute, ': ', mean_attribute_good, ' += ',
              std_attribute_good, '\n')

    plt.show()
