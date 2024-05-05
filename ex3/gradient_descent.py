import numpy as np
import matplotlib.pyplot as plt


class GradientDescentBatch:
    def __init__(self, learning_rate, starting_weights=None):
        self.learning_rate = learning_rate
        self.weights = starting_weights
        self.train_errors = []
        self.generalization_errors = []

    def fit(self, data, labels, target_func, epochs=100):
        if self.weights is None:
            # Initialize weights if not provided
            self.weights = np.zeros(data.shape[1])

        for epoch in range(epochs):
            # Compute predictions for the entire dataset
            predictions = np.dot(np.transpose(data), self.weights)

            # Compute momentary errors for each example
            self.train_errors.append(self.calc_train_errors(predictions, labels))

            # Compute generalization error
            self.generalization_errors.append(self.calculate_gen_error(target_func))

            # Update weights
            a = self.calc_gradient(data, predictions, labels)
            gradient = np.array([x[0] for x in a])
            self.weights = self.weights.astype(data.dtype)  # Ensure weights have the same dtype as data
            self.weights -= self.learning_rate * gradient

    def calc_train_errors(self, pred, target):

        return 0.5 * np.mean(np.square(pred - target))

    def calculate_gen_error(self, target_func):
        x_generalization = np.arange(-5, 5, 0.01).reshape(1, -1)
        ones = np.ones_like(x_generalization)
        samples_generalization = np.concatenate((x_generalization, ones), axis=0)
        pred_labels = np.dot(samples_generalization.T, self.weights)
        target_labels = np.array(target_func(x_generalization))
        generalization_error = 0.5 * np.mean(np.square(pred_labels - target_labels))
        return generalization_error

    def calc_gradient(self, sample, pred, target):
        error = pred - target
        return (1 / len(pred)) * np.dot(sample, np.transpose(error))

    def get_train_errors(self):
        return self.train_errors

    def get_generalization_errors(self):
        return self.generalization_errors


class GradientDescentOnline:
    def __init__(self, learning_rate, starting_weights=None):
        self.learning_rate = learning_rate
        self.weights = starting_weights
        self.momentary_errors = []
        self.generalization_errors = []

    def fit(self, data, labels, target_func, epochs=1):
        if self.weights is None:
            # Initialize weights if not provided
            self.weights = np.zeros(data.shape[1])

        for epoch in range(epochs):
            for i in range(len(labels[0])):
                sample = data[:, i]
                target = target_func(sample[0])

                # Compute prediction and error for the current example
                prediction = np.dot(sample, self.weights)
                self.momentary_errors.append(self.calc_momentery_error(prediction, sample[0]))
                self.generalization_errors.append(self.calculate_gen_error(target_func))

                # Update weights
                gradient = self.calc_gradient(sample, prediction, target)

                self.weights = self.weights.astype(data.dtype)  # Ensure weights have the same dtype as data
                self.weights -= (self.learning_rate * gradient)

    def calc_momentery_error(self, pred, target):
        return np.power((pred - target), 2)

    def calculate_gen_error(self, target_func):
        x_generalization = np.arange(-5, 5, 0.01).reshape(1, -1)
        ones = np.ones_like(x_generalization)
        samples_generalization = np.concatenate((x_generalization, ones), axis=0)
        pred_labels = np.dot(samples_generalization.T, self.weights)
        target_labels = np.array(target_func(x_generalization))
        generalization_error = 0.5 * np.mean(np.square(pred_labels - target_labels))
        return generalization_error

    def calc_gradient(self, sample, pred, target):
        return (pred - target) * sample[:2]

    def get_momentary_errors(self):
        return self.momentary_errors

    def get_generalization_errors(self):
        return self.generalization_errors


class LinearPerceptron:

    def __init__(self, starting_weights=None):
        self.model_weights = starting_weights
        self.train_errors = []
        self.generalization_errors = []

    def fit(self, sample_data, labels, target_func):
        C = self.get_corr_matrix(sample_data)
        u = self.get_corr_vector(sample_data, labels)
        weights = np.dot(np.linalg.inv(C), u)

        self.model_weights = weights

        pred = np.dot(np.transpose(sample_data), weights)
        self.train_errors.append(self.calc_train_errors(pred, labels))
        self.generalization_errors.append(self.calculate_gen_error(target_func))

    def predict(self, sample):
        return np.dot(self.model_weights, sample)

    def get_corr_matrix(self, X):
        p = X.shape[1]
        return (1 / p) * np.dot(X, np.transpose(X))

    def get_corr_vector(self, X, y):
        p = X.shape[1]
        return (1 / p) * np.dot(X, np.transpose(y))

    def get_weights(self):
        """
        Get the weights of the trained perceptron model.

        Returns:
        numpy.ndarray, shape (n,)
            The weights vector representing the decision boundary of the perceptron.
        """
        return self.model_weights

    def calc_train_errors(self, pred, target):
        return 0.05 * np.mean(np.square(pred - target))

    def calculate_gen_error(self, target_func):
        x_generalization = np.arange(-5, 5, 0.01).reshape(1, -1)
        ones = np.ones_like(x_generalization)
        samples_generalization = np.concatenate((x_generalization, ones), axis=0)
        pred_labels = np.dot(samples_generalization.T, self.model_weights)
        target_labels = np.array(target_func(x_generalization))
        generalization_error = 0.05 * np.mean(np.square(pred_labels - target_labels))

        return generalization_error

    def get_train_errors(self):
        return self.train_errors

    def get_generalization_errors(self):
        return self.generalization_errors


def create_data_set(row, col, bias_flag, target_func):
    data = np.random.uniform(low=-5, high=5, size=(row, col))
    ones = np.ones_like(data)
    labels = target_func(data)
    if bias_flag:
        data = np.concatenate((data, ones), axis=0)
    return data, labels


def plot_train_and_gen_error(batch_train, batch_gen, online_train, online_gen, linear_train, linear_gen):
    x_values = np.arange(1, len(batch_train) + 1)
    plt.figure(figsize=(10, 6))

    plt.plot(x_values, batch_train, label='Batch Train Error', color='blue')
    plt.plot(x_values, batch_gen, label='Batch Generalization Error', linestyle='dashed', color='blue')
    plt.plot(x_values, online_train, label='Online Momentary Error', color='green')
    plt.plot(x_values, online_gen, label='Online Generalization Error', linestyle='dashed', color='green')

    # Duplicate values for linear perceptron
    linear_train = np.repeat(linear_train, 100)
    linear_gen = np.repeat(linear_gen, 100)

    plt.plot(x_values, linear_train, label='Linear Perceptron Train Error', linestyle='dashed', color='orange')
    plt.plot(x_values, linear_gen, label='Linear Perceptron Generalization Error', linestyle='dashed', color='red')

    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Train and Generalization Errors Over Iterations')
    plt.legend()
    plt.savefig("train_gene_errors.png")
    plt.show()


def graph_error_per_lr(batch_train, batch_gen, online_gen, learning_rates):
    plt.figure(figsize=(12, 8))

    # Plot Batch Gradient Descent Train Errors
    for i, lr in enumerate(learning_rates):
        x_values_batch = np.arange(1, len(batch_train[i]) + 1)
        plt.plot(x_values_batch, batch_train[i], label=f'Batch GD, LR={lr}')

    plt.xlabel('Iteration')
    plt.ylabel('Train Error')
    plt.title('Batch Gradient Descent Train Errors')
    plt.legend()
    plt.savefig("batch_train_errors_per_lr.png")
    plt.show()

    plt.figure(figsize=(12, 8))

    # Plot Batch Gradient Descent Generalization Errors
    for i, lr in enumerate(learning_rates):
        x_values_batch = np.arange(1, len(batch_gen[i]) + 1)
        plt.plot(x_values_batch, batch_gen[i], label=f'Batch GD, LR={lr}')

    plt.xlabel('Iteration')
    plt.ylabel('Generalization Error')
    plt.title('Batch Gradient Descent Generalization Errors')
    plt.legend()
    plt.savefig("batch_gen_errors_per_lr.png")
    plt.show()

    plt.figure(figsize=(12, 8))

    # Plot Online Gradient Descent Generalization Errors
    for i, lr in enumerate(learning_rates):
        x_values_online = np.arange(1, len(online_gen[i]) + 1)
        plt.plot(x_values_online, online_gen[i], label=f'Online GD, LR={lr}')

    plt.xlabel('Iteration')
    plt.ylabel('Generalization Error')
    plt.title('Online Gradient Descent Generalization Errors')
    plt.legend()
    plt.savefig("online_gen_errors_per_lr.png")
    plt.show()


def q1():
    target_func = lambda x: 1 + x + np.power(x, 2) + np.power(x, 3)
    data, labels = create_data_set(1, 100, True, target_func)
    starting_weights = np.array([1, 1])
    learning_rate = 0.01
    linear_per = LinearPerceptron(starting_weights)
    batch_gd = GradientDescentBatch(learning_rate, starting_weights)
    online_gd = GradientDescentOnline(learning_rate, starting_weights)

    linear_per.fit(data, labels, target_func)
    batch_gd.fit(data, labels, target_func)
    online_gd.fit(data, labels, target_func)

    batch_train = batch_gd.get_train_errors()
    batch_get = batch_gd.get_generalization_errors()

    online_train = online_gd.get_momentary_errors()
    online_get = online_gd.get_generalization_errors()

    linear_train = linear_per.get_train_errors()
    linear_gen = linear_per.get_generalization_errors()

    plot_train_and_gen_error(batch_train, batch_get, online_train, online_get, linear_train, linear_gen)


def q2():
    target_func = lambda x: 1 + x + np.power(x, 2) + np.power(x, 3)
    data, labels = create_data_set(1, 500, True, target_func)
    starting_weights = np.array([1, 1])

    learning_rates = [0.002, 0.005, 0.01, 0.02, 0.05]

    batch_train = []
    batch_gen = []
    online_gen = []

    for lr in learning_rates:
        batch_gd = GradientDescentBatch(lr, starting_weights)
        online_gd = GradientDescentOnline(lr, starting_weights)

        batch_gd.fit(data, labels, target_func, epochs=500)
        online_gd.fit(data, labels, target_func)

        batch_train.append(batch_gd.get_train_errors())
        batch_gen.append(batch_gd.get_generalization_errors())
        online_gen.append(online_gd.get_generalization_errors())

    graph_error_per_lr(batch_train, batch_gen, online_gen, learning_rates)


def main():
    q1()
    q2()


if __name__ == "__main__":
    main()
