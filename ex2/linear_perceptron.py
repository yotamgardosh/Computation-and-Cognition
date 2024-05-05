import numpy as np
import matplotlib.pyplot as plt


class LinearPerceptron:
    model_weights = None

    def fit(self, sample_data, labels):
        C = self.get_corr_matrix(sample_data)
        u = self.get_corr_vector(sample_data, labels)
        weights = np.dot(np.linalg.inv(C), u)

        self.model_weights = weights

    def predict(self, sample):
        return np.dot(self.model_weights, sample)

    def get_corr_matrix(self, X):
        p = X.shape[1]
        return (1 / p) * np.dot(X, np.transpose(X))

    def get_corr_vector(self, X, y):
        p = X.shape[1]
        return (1 / p) * np.dot(X, y)

    def get_weights(self):
        """
        Get the weights of the trained perceptron model.

        Returns:
        numpy.ndarray, shape (n,)
            The weights vector representing the decision boundary of the perceptron.
        """
        return self.model_weights


def MSE(y_pred, y_test):
    return 0.5 * np.mean(np.square(y_pred-y_test))

def calculate_train_error(pred_vals, target_vals):
    train_error = MSE(pred_vals,target_vals)
    return train_error


def calculate_gen_error(model, target_func):
    x_generalization = np.arange(-1, 1.001, 0.001).reshape(1, -1)
    ones = np.ones_like(x_generalization)
    samples_generalization = np.concatenate((x_generalization, ones), axis=0)

    y_target = target_func(x_generalization)
    y_pred_generalization = np.dot(model.get_weights(), samples_generalization)

    generalization_error = MSE(y_pred_generalization,y_target)
    return generalization_error


def main():
    # q1
    P = 500
    N = 1
    ones = np.ones((1, P))
    samples_wo_bias = np.random.uniform(low=-1, high=1, size=(N, P))
    samples = np.concatenate((samples_wo_bias, ones), axis=0)
    target_func = lambda x: np.power(x, 3) - np.power(x, 2)
    labels = target_func(samples_wo_bias).flatten()

    # q2
    model = LinearPerceptron()
    model.fit(samples, labels)
    print("weights")
    print(model.get_weights())
    C = model.get_corr_matrix(samples)
    print("correlation matrix")
    print(C)
    u = model.get_corr_vector(samples, labels)
    print("correlation vector")
    print(u)

    # q3
    pred_values = np.array([model.predict(samples[:, i].reshape(-1, 1)) for i in range(samples.shape[1])])
    target_func = lambda x: np.power(x, 3) - np.power(x, 2)
    y_target = target_func(samples_wo_bias)
    plt.figure(figsize=(10, 6))
    plt.scatter(x=samples_wo_bias, y=y_target, label="target function")
    plt.scatter(x=samples_wo_bias, y=pred_values, label="learned linear function")
    plt.title("Learned vs Target Function in Range [-1,1]")
    plt.xlabel("sample values")
    plt.ylabel("predicted values")
    plt.legend()
    # plt.savefig("learned_vs_target_function_plot.png")
    plt.show()

    # q4
    train_error = calculate_train_error(pred_values, y_target)
    print("train error: {}".format(train_error))

    generalization_error = calculate_gen_error(model, target_func)
    print("Generalization Error: {}".format(generalization_error))

    # q5
    N = 1
    train_error_per_p = []
    gen_error_per_p = []
    M = 100
    target_func = lambda x: np.power(x, 3) - np.power(x, 2)

    model = LinearPerceptron()
    for P in range(5, 101, 5):
        train_error_avg = 0
        gen_error_avg = 0
        for _ in range(M):
            # create samples for train
            ones = np.ones((1, P))
            samples_wo_bias = np.random.uniform(low=-1, high=1, size=(N, P))
            samples = np.concatenate((samples_wo_bias, ones), axis=0)
            labels = target_func(samples_wo_bias).flatten()
            model.fit(samples, labels)

            # calculate labels
            pred_values = np.dot(samples.T, model.get_weights())
            target_values = target_func(samples_wo_bias)

            # calculate error
            train_error_avg += calculate_train_error(pred_values, target_values)
            gen_error_avg += calculate_gen_error(model, target_func)


        train_error_per_p.append(train_error_avg / M)
        gen_error_per_p.append(gen_error_avg / M)

    # Plotting
    p_values = range(5, 101, 5)
    plt.plot(p_values, train_error_per_p, label='Training Error')
    plt.plot(p_values, gen_error_per_p, label='Generalization Error')
    plt.xlabel('P')
    plt.ylabel('Average Error')
    plt.legend()
    plt.title('Average Training and Generalization Error as Function of P')
    # plt.savefig("train_gen_error_plot.png")
    plt.show()




if __name__ == "__main__":
    main()
