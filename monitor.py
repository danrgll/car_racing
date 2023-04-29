import seaborn as sns
from imitation_learning.train import read_data, preprocessing
import matplotlib.pyplot as plt
import pandas as pd


def see_action_data_distribution(data):
    """
    plot data distribution of given dimension
    """
    value_dict = {0: 'STRAIGHT', 1: 'LEFT', 2: 'RIGHT', 3: 'ACCELERATE', 4:"BRAKE"}
    string_data = [value_dict[action] for action in data]
    df = pd.DataFrame({"action": string_data})
    # Define the desired order of the categories
    category_order = ['ACCELERATE', 'STRAIGHT', 'LEFT', "RIGHT", "BRAKE"]
    # Plot the countplot using Seaborn's countplot function
    sns.countplot(data=df, x="action", order=category_order).set(title="Action Distribution")
    plt.show()


def plot_learning_curves():
    pass


if __name__=="__main__":
    X_train, y_train, X_valid, y_valid = read_data("data")
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=3)
    see_action_data_distribution(y_train)
