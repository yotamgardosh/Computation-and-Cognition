import typing
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    model_weights = None

    def heaviside(slef, z):
        """
        Heaviside step function.

        Parameters:
        - z: float
            The input value.

        Returns:
        int
            The binary output of the Heaviside step function (0 if z < 0, 1 if z >= 0).
        """
        return 0 if z < 0 else 1

    def fit(self, sample_data, labels, weights=None):
        """
        Train the perceptron model using the provided sample data and labels.

        Parameters:
        - sample_data: numpy.ndarray, shape (p, n)
            The input samples where p is the number of samples and n is the number of features.
        - labels: numpy.ndarray, shape (p,)
            The binary labels corresponding to the samples.
        - weights: numpy.ndarray, shape (n,), optional
            Initial weights for the perceptron. If not provided, random weights will be initialized.

        Returns:
        None
        """
        p, n = sample_data.shape

        if weights is None:
            weights = np.random.rand(n)

        while True:
            pre_train_weights = weights.copy()
            for i in range(p):
                y_pred = self.heaviside(np.dot(weights, sample_data[i]))
                if y_pred != labels[i]:
                    weights += (2 * labels[i] - 1) * sample_data[i]

            if np.array_equal(weights, pre_train_weights):
                break

        # Post-training normalization
        # print(weights)
        # weights = weights.astype(float)
        weights /= np.linalg.norm(weights)

        self.model_weights = weights

    def predict(self, sample):
        """
        Predict the binary class of a single input sample using the trained perceptron model.

        Parameters:
        - sample: numpy.ndarray, shape (n,)
            The input sample for prediction.

        Returns:
        int
            The predicted class (0 or 1) based on the perceptron's decision rule.
        """
        return self.heaviside(np.dot(self.model_weights, sample))

    def get_weights(self):
        """
        Get the weights of the trained perceptron model.

        Returns:
        numpy.ndarray, shape (n,)
            The weights vector representing the decision boundary of the perceptron.
        """
        return self.model_weights


def plot_samples_with_predictions(samples, predictions, weights):
    """
    Plot samples with predictions using a scatter plot and the decision boundary.

    Parameters:
    - samples: 2D NumPy array of shape (number of samples, 2)
    - predictions: 1D NumPy array of shape (number of samples)
    - weights: 1D NumPy array of shape (2,)
    """

    plt.figure(figsize=(8,8))

    # Create separate scatter plots for each class
    plt.scatter(samples[predictions == 0, 0], samples[predictions == 0, 1], c='red', marker='o', label="Class 0")
    plt.scatter(samples[predictions == 1, 0], samples[predictions == 1, 1], c='blue', marker='o', label="Class 1")

    # Plot the decision boundary using the weights vector
    if weights is not None:
        x_values = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 1000)
        y_values = -(weights[0] * x_values) / weights[1]
        plt.plot(x_values, y_values, label="Decision Boundary", color='black')

        # Plot the vector generated by weights (perpendicular to the decision boundary)
        vector_weights = np.array([weights[1], -weights[0]])  # Perpendicular vector
        y_values_vector = -(vector_weights[0] * x_values) / vector_weights[1]
        plt.plot(x_values, y_values_vector, label="Vector Generated by Weights", linestyle='--', color='green')


    # Set labels and title
    plt.xlabel('X1 Value')
    plt.ylabel('X2 Value')
    plt.title('Scatter Plot of Samples in the X1, X2 Plane')

    # Add a legend
    plt.legend()

    # Save and display the plot
    plt.savefig("sample_predictions_scatter.png")
    plt.show()


def eval_error(w, optimal_w):
    """
    Evaluate the error metric as the absolute angle (in degrees) between two vectors.

    Parameters:
    - w: numpy.ndarray, shape (n,)
        The first vector.
    - optimal_w: numpy.ndarray, shape (n,)
        The second vector representing the optimal or desired direction.

    Returns:
    - float
        The absolute angle (in degrees) between the two vectors.
    """
    theta = np.dot(w, optimal_w) / (np.linalg.norm(w) * np.linalg.norm(optimal_w))

    angle_rad = np.arccos(np.clip(theta, -1.0, 1.0))  # we use clip to avoid float points errors
    angle_deg = np.rad2deg(angle_rad)

    return np.abs(angle_deg)


def main():
    np.random.seed(10)
    # q2
    ################################################
    # To stick to common convention the sample data
    # is each row a sample and each column a feature
    ################################################
    N = 2
    P = 1000
    samples = np.random.uniform(low=-10, high=10, size=(P, N))
    labels = np.array([int(x[0] > x[1]) for x in samples])
    # samples = np.array([[1, -4], [-10,  6], [6, -1], [4, 7], [9, 7], [-5, 7], [-10, 1], [-2, 6], [-10,  8], [-3, 2]])
    # labels = np.array([1, 0, 1,0, 1, 0, 0, 0, 0, 0])
    # samples = np.array([[-10, -1], [-3, -8], [8, -10], [-7, 3], [4, 4], [2, -1], [-7, 8], [4, -4], [-10, -5], [3, -5]])
    # labels = np.array([0, 1, 1, 0, 1, 1, 0, 1, 0, 1])

    # q3
    weights = [1, 1]
    my_perceptron = Perceptron()
    my_perceptron.fit(samples, labels, weights)
    print(my_perceptron.get_weights())
    predictions = np.array([my_perceptron.predict(x) for x in samples])
    fitted_weights = my_perceptron.get_weights()
    # plot_samples_with_predictions(samples, predictions, fitted_weights)

    # q4
    weights = [1, 1]
    optimal_weights = [1, -1]
    P_values = [5, 20, 30, 50, 100, 150, 200, 500]
    M = 100
    error_arr = []
    for P in P_values:
        error_per_P = 0
        for _ in range(M):
            # generate values
            samples = np.random.uniform(low=-10, high=10, size=(P, N))
            labels = np.array([int(x[0] > x[1]) for x in samples])
            # fit model
            my_perceptron = Perceptron()
            my_perceptron.fit(samples, labels, weights)
            fitted_weights = my_perceptron.get_weights()
            # eval error
            error_per_P += eval_error(fitted_weights, optimal_weights)
        # append avg error after M simulations
        error_arr.append(error_per_P / M)

    print(error_arr)
    # plt.scatter(x=P_values, y=error_arr, marker='o')
    # plt.plot(P_values, error_arr, linestyle='-', color='blue', label='Connect the Dots')
    # plt.xlabel("Number of Samples")
    # plt.xlabel("Average Error (Degrees)")
    # plt.title("Average Error Plot as Function of Number of Samples")
    # plt.savefig("average_error_plot.png")
    # plt.show()


if __name__ == "__main__":
    main()
