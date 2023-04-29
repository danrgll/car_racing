import numpy as np
import gzip
import pickle

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all((a == [-1.0, 0.0, 0.0]).flatten()):
        return LEFT               # LEFT: 1
    elif all((a == [1.0, 0.0, 0.0]).flatten()):
        return RIGHT             # RIGHT: 2
    elif all((a == [0.0, 1.0, 0.0]).flatten()):
        return ACCELERATE        # ACCELERATE: 3
    elif all((a == [0.0, 0.0, 0.2]).flatten()):
        return BRAKE             # BRAKE: 4
    else:       
        return STRAIGHT                                      # STRAIGHT = 0


def id_to_action(action_id, max_speed=0.8):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    a = np.array([0.0, 0.0, 0.0])

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.05])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.05])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])
    

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))


def concatenate(dataset1, dataset2):
    # open the first gzip file for reading
    with gzip.open(dataset1, 'rb') as file1:
        # read the contents of the first file
        dict1 = pickle.load(file1)
    # open the second gzip file for reading
    with gzip.open(dataset2, 'rb') as file2:
        # read the contents of the first file
        dict2 = pickle.load(file2)

    s1 = np.array(dict1["state"])
    a1 = np.array(dict1["action"])
    r1 = np.array(dict1["reward"])
    ns1 = np.array(dict1["next_state"])
    t1 = np.array(dict1["terminal"])

    s2 = np.array(dict2["state"])
    a2 = np.array(dict2["action"])
    r2 = np.array(dict2["reward"])
    ns2 = np.array(dict2["next_state"])
    t2 = np.array(dict2["terminal"])

    # concatenate the content of the two files
    concatenated_states = np.concatenate((s1, s2), axis=0)
    concatenated_next_state = np.concatenate((ns1, ns2), axis=0)
    concatenated_action = np.concatenate((a1, a2), axis=0)
    concatenated_reward = np.concatenate((r1, r2), axis=0)
    concatenated_terminal = np.concatenate((t1, t2), axis=0)
    print(concatenated_states.shape[0])
    samples = {"state": concatenated_states,
               "next_state": concatenated_next_state,
               "action": concatenated_action,
               "reward": concatenated_reward,
               "terminal": concatenated_terminal}

    # write the concatenated data to a new gzip file
    with gzip.open('big_data.pkl.gzip', 'wb') as f:
        pickle.dump(samples, f)