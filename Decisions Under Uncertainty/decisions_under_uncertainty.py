from ex7 import HamsterStudent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MY_ID = 334
U = lambda x, sig: np.power(x, sig)
PI = lambda p, alph: np.exp(-np.power((-np.log(p)), alph))

class HamsterExperiment:
    def __init__(self, student_id):
        self.student_id = student_id
        self.results = []
    def find_Xs(self, Xg, increment=0.01):
        next_p = cur_p = HamsterStudent.myHamster(0, Xg, self.student_id)
        Xs = increment
        while cur_p == next_p:
            cur_p = next_p
            Xs += increment
            next_p = HamsterStudent.myHamster(Xs, Xg, self.student_id)
        return Xs

    def run_experiment(self):
        Xg = 1  # Initial Xg
        Xs_results = [self.find_Xs(Xg)]
        for round in range(4):
            Xg = Xs_results[-1]  # Set Xg for the next round to the equilibrium Xs found
            Xs = self.find_Xs(Xg)
            Xs_results.append(Xs)
            print(f"Round {round + 1}: Xg = {Xg}, Xs equilibrium = {Xs}")

        self.results = Xs_results

    def plot_utility_function(self):
        # Assuming self.results contains the Xs values found in the experiment
        # Incorporate the fixed points (0,0) and (1,1) into the plot data
        Xs_values = [1] + self.results + [0]  # Add 0 at the start and 1 at the end
        utility_values = [1] + [0.5 ** i for i in range(1, len(self.results) + 1)] + [0]  # Adjust for the added points

        # Plotting the utility function
        plt.figure(figsize=(8, 6))
        plt.plot(Xs_values, utility_values, marker='o', linestyle='-', color='b')

        for Xs, utility in zip(Xs_values, utility_values):
            plt.text(Xs, utility, f'({Xs:.2f}, {utility:.2f})')

        # Adding titles and labels
        plt.title(MY_ID)
        plt.xlabel('Grams of Peanuts')
        plt.ylabel('Utility')

        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
        plt.savefig('hamster_utility_function.png')
        plt.show()


def plot_graphs(x_values, func, title, xlabel, ylabel):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns of graphs

    for i, a in enumerate([0.5, 1, 2]):
        y_values = [func(x, a) for x in x_values]

        axs[i].plot(x_values, y_values)
        axs[i].set_title(f'{title} (a={a})')
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].grid(True)

    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.savefig(f'{title}.png')
    plt.show()


def calc_indifference_point(df):
    indifference_points = []

    for index, row in df.iterrows():
        # Filter choices for the current row's conditions
        gamble_xs = df[(df['subject'] == row['subject']) & (df['Xg'] == row['Xg']) &
                       (df['p'] == row['p']) & (df['h'] == row['h']) & (df['choice'] == 1)]['Xs']
        safe_xs = df[(df['subject'] == row['subject']) & (df['Xg'] == row['Xg']) &
                     (df['p'] == row['p']) & (df['h'] == row['h']) & (df['choice'] == 2)]['Xs']

        if len(gamble_xs) == 0:  # All choices are safe
            ind_point = np.min(safe_xs) / 2
        elif len(safe_xs) == 0:  # All choices are for gambling
            xg = row['Xg']
            ind_point = (np.max(gamble_xs) + xg) / 2
        else:  # Mixed choices
            ind_point = (np.max(gamble_xs) + np.min(safe_xs)) / 2

        indifference_points.append(ind_point)

    df['indifference_point'] = indifference_points
    return df


def find_sigma_alpha_with_indifference(df):

    with_indifference = calc_indifference_point(df)
    subject_params_per_trial = {}

    for subject in df['subject'].unique():
        subject_data = with_indifference[with_indifference['subject'] == subject]

        subject_params = {}
        for trial in [1, 2]:
            trial_data = subject_data[subject_data['h'] == trial]

            if trial_data.empty:
                # Ensure that we have a placeholder if no data is available for the trial
                subject_params[trial] = {'alpha': None, 'sigma': None}
                continue

            p_decimal = trial_data['p'] / 100
            p_decimal = np.maximum(p_decimal, np.finfo(float).eps)  # Avoid log(0)
            x = np.log(-np.log(p_decimal))
            y = np.log(-np.log(trial_data['indifference_point'] / trial_data['Xg']))

            polynom = np.polyfit(x, y, 1)
            alpha, neg_log_sigma = polynom[0], polynom[1]
            sigma = np.exp(-neg_log_sigma)

            subject_params[trial] = {'alpha': alpha, 'sigma': sigma}

        subject_params_per_trial[subject] = subject_params

    return subject_params_per_trial

def plot_sigmas_and_alphas_with_means(subject_params_per_trial):
    # Initialize lists to hold sigma and alpha values for the first and second trials
    sigma_first_trial = []
    sigma_second_trial = []
    alpha_first_trial = []
    alpha_second_trial = []

    # Extract values from the subject_params_per_trial dictionary
    for subject, trials in subject_params_per_trial.items():
        sigma_first_trial.append(trials[1]['sigma'])
        sigma_second_trial.append(trials[2]['sigma'])
        alpha_first_trial.append(trials[1]['alpha'])
        alpha_second_trial.append(trials[2]['alpha'])

    # Plotting Sigma values
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)  # First subplot for sigma
    plt.scatter(sigma_first_trial, sigma_second_trial)
    plt.title('Sigma: Trial 2 vs. Trial 1')
    plt.xlabel('Sigma from Trial 1')
    plt.ylabel('Sigma from Trial 2')

    # Calculate and plot mean lines for sigma
    mean_sigma_first = np.mean(sigma_first_trial)
    mean_sigma_second = np.mean(sigma_second_trial)
    plt.axvline(x=mean_sigma_first, color='r', linestyle='--', label=f'Mean Trial 1: {mean_sigma_first:.2f}')
    plt.axhline(y=mean_sigma_second, color='g', linestyle='--', label=f'Mean Trial 2: {mean_sigma_second:.2f}')
    plt.legend()

    plt.grid(True)
    plt.axis('equal')

    # Plotting Alpha values
    plt.subplot(1, 2, 2)  # Second subplot for alpha
    plt.scatter(alpha_first_trial, alpha_second_trial)
    plt.title('Alpha: Trial 2 vs. Trial 1')
    plt.xlabel('Alpha from Trial 1')
    plt.ylabel('Alpha from Trial 2')

    # Calculate and plot mean lines for alpha
    mean_alpha_first = np.mean(alpha_first_trial)
    mean_alpha_second = np.mean(alpha_second_trial)
    plt.axvline(x=mean_alpha_first, color='r', linestyle='--', label=f'Mean Trial 1: {mean_alpha_first:.2f}')
    plt.axhline(y=mean_alpha_second, color='g', linestyle='--', label=f'Mean Trial 2: {mean_alpha_second:.2f}')
    plt.legend()

    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('sigmas_and_alphas.png')
    plt.show()



def main():
    #q1
    experiment = HamsterExperiment(MY_ID)
    experiment.run_experiment()
    experiment.plot_utility_function()
    x_values = np.linspace(0.01, 1, 100)
    plot_graphs(x_values, U, 'Utility Function', 'x', 'U(x, sig)')
    plot_graphs(x_values, PI, 'Probability Function', 'p', 'PI(p, alph)')

    #q2
    df = pd.read_csv('ex7_q2_data.csv')
    alpha_sig_dict = find_sigma_alpha_with_indifference(df)
    plot_sigmas_and_alphas_with_means(alpha_sig_dict)



if __name__ == "__main__":
    main()