import math

# Takes in a bunch of data, and computes the entropy
# The data is a set of records, each with some binary attributes and a binary label
def computeTheEntropy(data):

    entropy = -1

    num_rows = len(data)
    num_cols = len(data[0])

    proportion_of_samples_with_attribute_zero = 0
    proportion_of_samples_with_attribute_one = 0

    for j in range(num_rows):
        if data[j][4] == 0:
            proportion_of_samples_with_attribute_zero += 1
        if data[j][4] == 1:
            proportion_of_samples_with_attribute_one += 1

    proportion_of_samples_with_attribute_zero /= num_rows
    proportion_of_samples_with_attribute_one /= num_rows

    # We handle when proportion is 0 separately
    if proportion_of_samples_with_attribute_zero != 0:
        contrib_zeroes = proportion_of_samples_with_attribute_zero*math.log(proportion_of_samples_with_attribute_zero)
    else:
        contrib_zeroes = 0

    if proportion_of_samples_with_attribute_one != 0:
        contrib_ones = proportion_of_samples_with_attribute_one * math.log(proportion_of_samples_with_attribute_one)
    else:
        contrib_ones = 0

    entropy = -(contrib_zeroes + contrib_ones)

    return entropy


def main():
    # data = load_from_text_file("data.txt")

    data = [

        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1]

    ]

    data = [row for row in data if row[1] == 0]

    print(computeTheEntropy(data))

    num_rows = len(data)
    num_cols = len(data[0])

    initial_entropy = computeTheEntropy(data)
    max_info_gain = 0
    best_attribute = -1

    for i in [0,2,3]:
        # We are considering splitting on attribute number i.

        # proportion_of_samples_with_attribute_zero = 0
        # proportion_of_samples_with_attribute_one = 0
        #
        # for j in range(num_rows):
        #     if data[j][i] == 0:
        #         proportion_of_samples_with_attribute_zero += 1
        #     if data[j][i] == 1:
        #         proportion_of_samples_with_attribute_one += 1
        #
        # proportion_of_samples_with_attribute_zero /= num_rows
        # proportion_of_samples_with_attribute_one /= num_rows

        data_but_attribute_is_zero = [row for row in data if row[i] == 0]
        data_but_attribute_is_one = [row for row in data if row[i] == 1]

        proportion_of_samples_with_attribute_zero = len(data_but_attribute_is_zero) / num_rows
        proportion_of_samples_with_attribute_one = len(data_but_attribute_is_one) / num_rows

        expected_new_entropy = (proportion_of_samples_with_attribute_zero*computeTheEntropy(data_but_attribute_is_zero)
                               + proportion_of_samples_with_attribute_one*computeTheEntropy(data_but_attribute_is_one))

        if initial_entropy - expected_new_entropy >= max_info_gain:
            max_info_gain = initial_entropy - expected_new_entropy
            best_attribute = i

    print(best_attribute)
    print(max_info_gain)


if __name__ == '__main__':
    main()