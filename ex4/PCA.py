import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def generate_scatter():
    # Number of points
    num_points = 1000

    # Generate random samples from the normal distributions
    x1_samples = np.random.normal(0, 1, num_points)
    x2_samples = np.random.normal(0, 2, num_points)

    # Create 2D points by pairing samples from each distribution
    points = np.column_stack((x1_samples, x2_samples))

    # Plot the points
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5)

    # Plot the vector w = (0, 1) as a red line starting from the origin
    plt.arrow(0, 0, 0, 1, color='red', width=0.05, head_width=0.1, head_length=0.1, length_includes_head=True)
    plt.xlim(-3, 3)
    plt.ylim(-6, 6)
    plt.title('Sample of Points from 2D Normal Distribution with Optimal w')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True)
    # plt.savefig('scatter_with_opt_w.jpeg')
    plt.show()


def plot_q1(df):
    sns.scatterplot(data=df, x='Alcohol', y='Malic.Acid', hue='Class', palette='viridis', legend='full')
    plt.title('Scatter Plot of Wines Dataset')
    # plt.savefig('scatter_q1.jpeg')
    plt.show()


def normalize_features(df):
    # Calculate mean and std for each feature
    feature_means = df.mean()
    feature_stds = df.std()

    # normalize each feature
    normalized_df = (df - feature_means) / feature_stds
    print(normalized_df.mean())
    return normalized_df


def calculate_corr_eig(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Find the eigenvectors and eigenvalues of the correlation matrix
    eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

    return corr_matrix, eigenvalues, eigenvectors


def calculate_explained_var(eigenvalues, idx):
    eig_to_calc = eigenvalues[idx]
    explained_var = eig_to_calc / sum(eigenvalues)
    # print('PC_{idx}: {precent}%'.format(idx=idx, precent=float("{:.2f}".format(explained_var))))
    return float("{:.2f}".format(explained_var))


def plot_cumulative_explained_var(eigenvalues):
    # Calculate cumulative explained variance
    cumulative_var = np.cumsum([calculate_explained_var(eigenvalues, i) for i in range(len(eigenvalues))])

    # Plot the cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(eigenvalues) + 1), cumulative_var, marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.xticks(range(1, len(eigenvalues) + 1), eigenvalues.round(2), rotation=45)
    plt.grid(True)
    # plt.savefig('cumulative_explained_var.jpeg')
    plt.show()


def reduce_to_2_dim(pc1_idx, pc2_idx, wine_class, df, eigenvectors, x_label, y_label):
    # Take the eigenvectors corresponding to the first 2 principal components
    pc1 = eigenvectors[:, pc1_idx]
    pc2 = eigenvectors[:, pc2_idx]

    # Project the data onto the first 2 principal components
    reduced_data = np.dot(df.values, np.vstack((pc1, pc2)).T)

    # Create a DataFrame with the reduced data
    reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

    # Add the wine_class as a column to the reduced DataFrame
    reduced_df['Class'] = wine_class

    # Plot the reduced data with wine_class as the hue
    sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Class', palette='viridis', legend='full')
    plt.title('Data Reduced to 2D with Principal Components')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    file_name = "plot_after_dim_reduction_{}_{}.jpeg".format(x_label, y_label)
    plt.savefig(file_name)
    plt.show()


def wines_df():
    df = pd.read_csv('wines.csv')
    # print(df.shape)
    # print(df.columns)
    plot_q1(df)

    wine_class = df['Class']
    df = df.drop('Class', axis=1)

    normalized_df = normalize_features(df)

    corr_matrix, eigenvalues, eigenvectors = calculate_corr_eig(normalized_df)

    # print("corr_matrix: ", corr_matrix)
    # print("eigenvectors: ", eigenvectors)
    # print("eigenvalues: ", eigenvalues)

    plot_cumulative_explained_var(eigenvalues)

    # first and second PC
    reduce_to_2_dim(0, 1, wine_class, normalized_df, eigenvectors, 'First PC', "Second PC")

    # first and last PC
    reduce_to_2_dim(0, 12, wine_class, normalized_df, eigenvectors, 'First PC', "Last PC")


if __name__ == "__main__":
    wines_df()
