import math
import numpy as np

def standardize(X):
    '''
    Use function standardize_by_mean_and_std to standardize the feature matrix X to have mean 0 and variance 1, and
    append a constant 1 to the feature vector.

    :param
        X: numpy array of size [num_samples, feat_dim]
        where num_samples is the number of samples
        and feat_dim is the dimension of features

    :return: X: numpy array of size [num_samples, feat_dim + 1]
        mean: mean value for each feature. numpy array of size [1, feat_dim]
        std: standard deviation for each feature.
    '''


    n, m = X.shape
    mean_vector = np.zeros(m)

    # Compute means
    for i in range(m):
        sum = 0
        for j in range(n):
            sum = sum + X[i][j]

        average = sum / n
        mean_vector[i] = average

    # Compute stdevs
    stdev_vector = np.zeros(m)
    for i in range(m):
        total_squared_error = 0
        for j in range(n):
            total_squared_error = total_squared_error + (X[i][j] - mean_vector[i])^2


        # Should I be dividing by n-1 or n??
        average_squared_error = total_squared_error / (n-1)
        stdev_vector[i] = math.sqrt(average_squared_error)

    X = standardize_by_mean_and_std(mean_vector,stdev_vector)

    return X, mean_vector, stdev_vector


def standardize_by_mean_and_std(X, mean, std):
    '''
    Standardize the feature matrix X using the passed mean and std.
    And append a constant 1 in the feature.

    :param
        X: numpy array of size [num_samples, feat_dim]
        where num_samples is the number of samples
        and feat_dim is the dimension of features

        mean: numpy array of size [1, feat_dim]. mean value for each feature.

        std: numpy array of size [1, feat_dim]. standard deviation for each feature.

    :return:
        normalized X: numpy array of size [num_samples, feat_dim + 1]
    '''

    n, m = X.shape
    for i in range(m):
        for j in range(n):
            X[i][j] = (X[i][j] - mean[i]) / std[i]

    column_vector_ones = np.ones(n,1)
    X = np.hstack(X,column_vector_ones)

    return X

def perceptron_pred(x, w):
    '''
    Output the prediction result.

    :param
        x: feature vector. numpy array of size [feat_dim + 1,] where feat_dim is the dimension of features
        w: weight vector. numpy array of size [feat_dim + 1,] where feat_dim is the dimension of features
    :return:
        the predicted result. a scalar 1 or -1
    '''

    dot_prodcut = np.dot(x,w)

    if dot_prodcut <= 0:
        return -1
    else:
        return 1

def perceptron(X, Y, w):
    '''
    Perceptron algorithm.

    :param
        X: input features. numpy array of size [num_samples, feat_dim + 1]
        where num_samples is the number of samples
        and feat_dim is the dimension of features

        Y: labels. numpy array of size [num_samples, ]
        where num_samples is the number of samples

        w: initialized weight vector. numpy array of size [feat_dim + 1,]

    :return:
        w: weight vector after training
    '''

    n, m = X.shape
    flag = True

    while flag:
        flag = False
        for i in range(1, n):
            row = X[i]
            prediction = perceptron_pred(row, w)
            if prediction != Y[i]:
                w = w + Y[i]*X[i]
                flag = True

    return w

def main():
    array = np.array([[1,2,3,6,10],[3,4,4,4,4]])
    n,m = np.shape(array)
    print(n)
    print(m)
    print(np.shape(array))

if __name__ == '__main__':
    main()
