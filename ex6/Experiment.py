
import os
import time
import random
import matplotlib.pyplot as plt
import json

random.seed(42)

def plot_choices(results, q1, q0):
    # Extract the choices from the results
    choices = [choice for choice, reward in results]

    # Compute the moving average with a window of 20
    moving_avg = [sum(choices[i:i + 20]) / 20 for i in range(len(choices) - 19)]

    # Create a scatter plot for the choices
    plt.scatter(range(1, 101), choices, label='User Choices', alpha=0.6)

    # Plot the moving average
    plt.plot(range(20, 101), moving_avg, label='Moving Average (window=20)', color='orange')

    # Add titles and labels
    plt.title('User Choices and Moving Average - q1: {q1:.2f}, q0: {q0:.2f}'.format(q1=q1, q0=q0))
    plt.xlabel('Trial Number')
    plt.ylabel('Choice')
    plt.legend()
    plt.savefig('choices_prob_{prob1}_{prob0}.jpg'.format(prob1=q1, prob0=q0))

    # Show the plot
    plt.show()
0

def run_experiment(q1, q0, trial_num):
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results = []
    for trial in range(100):
        while True:
            choice = input(f"trial {trial}: Choose 1 or 0: ".format(trial=trial))
            if choice in ('1', '0'):
                break
            else:
                print("Invalid input. Please choose 1 or 0.")

        choice = int(choice)
        reward = random.choices([1, 0], weights=[q1, 1 - q1] if choice == 1 else [q0, 1 - q0])[0]

        print(f"Reward: {reward}")
        time.sleep(0.3)
        os.system('cls' if os.name == 'nt' else 'clear')

        results.append((choice, reward))

    # Save the results to a file
    results_file_path = os.path.join(results_dir, f'experiment_results_trial_{trial_num}.json')
    with open(results_file_path, 'w') as f:
        json.dump(results, f)

    print(f"Results saved to {results_file_path}")
    return results

def lria_update(choices, rewards, eta):
    """
    Function to simulate the learning process using the modified LRIA update rule that includes rewards.

    Parameters:
    - choices: list of 0s and 1s representing the subject's choices.
    - rewards: list of received rewards corresponding to the choices.
    - eta: the learning rate parameter.

    Returns:
    - Pt: list of probabilities corresponding to the LRIA update rule.
    """
    Pt = [0.5]  # Start with P1 = 0.5
    for choice, reward in zip(choices, rewards):
        # Modified LRIA update rule: Pt+1 = Pt + eta * reward * (Ct - Pt)
        # Where Ct is the actual choice (0 or 1), reward is the received reward, and Pt is the current probability.
        Pt.append(Pt[-1] + eta * reward * (choice - Pt[-1]))

    return Pt[:-1]  # Return the series of probabilities excluding the last updated value which has no corresponding choice


def compute_loss_for_eta(Pt, Yt):
    """
    Computes the loss for a given eta.

    Parameters:
    - Pt: List of predicted probabilities.
    - Yt: List of actual choices.
    - eta: Learning rate parameter.

    Returns:
    - The loss for the given eta.
    """
    T = len(Pt)
    loss = sum((Pt[i] - Yt[i]) ** 2 for i in range(T)) / T
    return loss


def find_optimal_eta(results, eta_range):
    """
    Finds the eta value that minimizes the loss.

    Parameters:
    - choices: List of actual choices.
    - eta_range: Range of eta values to test.

    Returns:
    - Tuple of (best_eta, lowest_loss).
    """
    lowest_loss = float('inf')
    best_eta = None
    losses = []
    choices = [choice for choice, reward in results]
    rewards = [reward for choice, reward in results]

    for eta in eta_range:
        # Calculate the probabilities using the LRIA update rule
        Pt = lria_update(choices, rewards, eta)
        # Compute the loss for the current eta
        loss = compute_loss_for_eta(Pt, choices)
        losses.append(loss)

        if loss < lowest_loss:
            lowest_loss = loss
            best_eta = eta

    return best_eta, lowest_loss, losses

def plot_losses(eta_range, losses, subject_num):
    plt.plot(eta_range, losses)
    plt.title('Loss vs. Learning Rate')
    plt.xlabel('Learning Rate (eta)')
    plt.ylabel('Loss')
    plt.savefig(f'loss_vs_learning_rate_{subject_num}.jpg'.format(subject_num=subject_num))
    plt.show()

def main():

    probs = [(1/3, 2/3), (random.uniform(0, 1), random.uniform(0, 1))]
    for i in range(1,len(probs)):
        q1, q0 = probs[i]
        results = run_experiment(q1, q0, i+1)
        plot_choices(results, q1, q0)

    eta_range = [i * 0.001 for i in range(0, 1001)]

    file_path = os.path.join('results', 'experiment_results_trial_1.json')
    with open(file_path, 'r') as f:
        experiment_results = json.load(f)


    best_eta, lowest_loss, losses = find_optimal_eta(experiment_results, eta_range)
    plot_losses(eta_range, losses, 1)
    print('experiment 1:')
    print(f"Best eta: {best_eta}, Lowest loss: {lowest_loss}")

    file_path = os.path.join('results', 'experiment_results_trial_2.json')
    with open(file_path, 'r') as f:
        experiment_results = json.load(f)

    best_eta, lowest_loss, losses = find_optimal_eta(experiment_results, eta_range)
    plot_losses(eta_range, losses, 2)
    print('experiment 2:')
    print(f"Best eta: {best_eta}, Lowest loss: {lowest_loss}")




if __name__ == "__main__":
    main()