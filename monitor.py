import seaborn as sns
from imitation_learning.train import read_data, preprocessing
import matplotlib.pyplot as plt
import pandas as pd


def monitor_training(tb, train_loss, train_score, epoch, model, val_loss=None, val_score=None):
    tb.add_scalar("Train Loss", train_loss, epoch)
    if val_loss is not None:
        tb.add_scalar("Validation_loss", val_loss, epoch)
        tb.add_scalars("Train_Validation_Loss", {"train loss": train_loss,
                                                "val_loss": val_loss}, epoch)
        tb.add_scalar("Train Accuracy", train_score, epoch)
        tb.add_scalar("Val Accuracy", val_score, epoch)
        tb.add_scalars("Train_Val_accuracy", {"train_acc": train_score, "val_acc": val_score}, epoch)
        counter = 1
        # monitor weights and biases and gradients of them
        for layer in model.model:
            if hasattr(layer, 'weight'):
                tb.add_histogram("layer_weight_histo" + str(counter), layer.weight, epoch)
                tb.add_histogram("layer_weight_grad__histo" + str(counter), layer.weight.grad, epoch)
            if hasattr(layer, 'bias'):
                tb.add_histogram("layer_bias_histo" + str(counter) + "bias", layer.bias, epoch)
                tb.add_histogram("layer_bias_grad_histo" + str(counter) + "bias", layer.bias.grad, epoch)
            counter += 1


def see_action_data_distribution(data):
    """
    plot data distribution of given dimension
    """
    value_dict = {0: 'STRAIGHT', 1: 'LEFT', 2: 'RIGHT', 3: 'ACCELERATE', 4:"BRAKE"}
    string_data = [value_dict[action] for action in data]
    df = pd.DataFrame({"action": string_data})
    # Define the desired order of the categories
    category_order = ['ACCELERATE"', 'STRAIGHT', 'LEFT', "RIGHT", "BRAKE"]
    # Plot the countplot using Seaborn's countplot function
    sns.countplot(data=df, x="action", order=category_order).set(title="Action Distribution")
    plt.show()


if __name__=="__main__":
    X_train, y_train, X_valid, y_valid = read_data("data")
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)
    see_action_data_distribution(y_train)
