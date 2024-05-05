import random

import numpy as np
import matplotlib.pyplot as plt


class StochasticBinaryPerceptron:
    def __init__(self, w, lr):
        self.w = w
        self.lr = lr

    def prob_y_equals_one(self, x):
        z = np.dot(self.w, x)
        return 1 / (1 + np.power(np.e, -z))

    def predict(self, x):
        P = self.prob_y_equals_one(x)
        return random.choices((1, 0), (P, 1 - P))[0]

    def reward(self, y_pred, y_test):
        return int(y_pred == y_test)

    def fit(self, x, y_test):
        y_pred = self.predict(x)
        r = self.reward(y_pred, y_test)
        P = self.prob_y_equals_one(x)
        updated_w = self.w
        for i in range(len(x)):
            e_i = x[i] * (y_pred - P)
            updated_w[i] += self.lr * r * e_i
        self.w = updated_w


def questions(data, labels, my_model, test_data, test_labels):
    accuracy_history = []
    for i, (x, y) in enumerate(zip(data.T, labels)):
        my_model.fit(x, y)
        if (i + 1) % 50 == 0:
            accuracy = evaluate_model_accuracy(my_model, test_data, test_labels)
            accuracy_history.append(accuracy)
    plot_accuracy(accuracy_history)
    plot_weights(my_model.w)


def evaluate_model_accuracy(model, test_data, test_labels):
    correct_predictions = 0
    total_samples = len(test_labels)
    for x, y in zip(test_data.T, test_labels):
        y_pred = model.predict(x)
        if y_pred == y:
            correct_predictions += 1
    accuracy = correct_predictions / total_samples
    return accuracy


def plot_accuracy(accuracy_history):
    iterations = range(50, len(accuracy_history) * 50 + 1, 50)
    plt.plot(iterations, accuracy_history)
    plt.title('Accuracy vs. Number of Samples')
    plt.xlabel('Number of Samples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig("accuracy_number_of_samples.jpg")
    plt.show()


def plot_weights(weights):
    weight_vec = np.reshape(weights, (28, 28))
    plt.imshow(weight_vec, interpolation='nearest')
    plt.savefig("weights_plot.jpg")
    plt.show()


def main():
    data = np.loadtxt('Ex5_data.csv', delimiter=',')
    labels = np.squeeze(np.loadtxt('Ex5_labels.csv', delimiter=','))

    test_data = np.loadtxt('Ex5_test_data.csv', delimiter=',')
    test_labels = np.squeeze(np.loadtxt('Ex5_test_labels.csv', delimiter=','))

    w_len = data.shape[0]
    init_w = np.random.normal(0, 0.01, w_len)
    my_model = StochasticBinaryPerceptron(init_w, 0.01)
    questions(data, labels, my_model, test_data, test_labels)


if __name__ == "__main__":
    main()
