import bisect
import csv
from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch


class KContextSpectrograms(torch.utils.data.Dataset):
    """Preprocesses dataset and returns frame surrounded by K context frames on both sides.
    
    Inherits from PyTorch Dataset class.
    
    Example:
    >>> dev_dataset = KContextSpectrograms(data/dev.npy, k=0)

    Args:
        data_path (str): Path to data file (i.e. "data/dev.npy")
        labels_path (str): Path to labels file (i.e. "data/dev_labels.npy")
        k (int): How many context frames to add to each side of the frame 
    """
    def __init__(self, data_path, labels_path=None, k=0):
        # Load in data (and labels, if available)
        data = np.load(data_path, allow_pickle=True)
        labels = np.load(labels_path, allow_pickle=True) if labels_path else None
        
        # Check the shape, type, or other details
        print(f'\nDetails for {data_path} is below:')
        print(f'\tNumber of utterances: {data.shape}')
      
        # Get the total number of frames in the dataset
        self.total_frames = np.sum([len(utterance) for utterance in data])
        
        # Preprocess the data and track indices of utterances.
        beg_of_utterances = [0]
        for u in range(len(data)):
            # Track the index of the first frame of each utterance
            beg_of_utterances.append(len(data[u]) + beg_of_utterances[u])
            
            # Pad the beginning/end of every utterance with k frames of zeros.
            # We do this so context can exist for the frames near the beg/end of the utterance
            data[u] = torch.FloatTensor(np.pad(data[u], ((k,k),(0,0)), mode='constant', constant_values=0))
            # print(f'\tNumber of frames in utterance {u} is: {data[u].shape}')

        if labels is not None:
            labels = [torch.LongTensor(l) for l in labels]
            
        # Store everything we'll need later as class variables
        self.k = k
        self.labels = labels
        self.data = data
        self.beg_of_utterances = beg_of_utterances
      
        
    def __len__(self):
        """Defines how many frames (observations) are in the dataset
        
        Example:
        >>> dev_dataset = KContextSpectrograms(path_to_dev_file, k=0) # assume 10 frames in dev dataset
        >>> len(dev_dataset)
        10

        Returns:
            int: num observations in dataset
        """
        return self.total_frames
    
    def __getitem__(self, i):
        """Given an index in [0, `__len__()`], returns single observation for that index.
        Dataloader calls this for you; you shouldn't need to call it.

        Args:
            i (int): index of observation to get

        Returns:
            torch.FloatTensor (and if label, torch.LongTensor): single observation shaped (2*k+1*40,) 
                                                                and, if exists, single label shaped (1,)
        """
        # Given index from [0, `__len__()`], find which utterance and frame within that utterance it comes from 
        utterance_id = bisect.bisect_right(self.beg_of_utterances, i) - 1
        frame_id = i - self.beg_of_utterances[utterance_id]
        
        # Get frame with context, shaped (2*k+1, 40)
        d = self.data[utterance_id][frame_id:frame_id+2*self.k+1]
        # Flatten into shape (2*k+1*40,) because linear layer is first layer
        d = d.flatten()

        # Return the data and label if it exists
        if self.labels is None:
            return d
        else:
            l = self.labels[utterance_id][frame_id]
            return d, l

def num_ms(k, stride=10, frame_width=25):
    """Calculate the number of milliseconds (ms) a target frame + context will cover

    Args:
        k (int): The number of context frames on each side of the target
        stride (int, optional): Stride (in ms) used to create mel-spectrogram. Defaults to 10.
        frame_width (int, optional): Width of frame (in ms). Defaults to 25.

    Returns:
        int: Number of ms covered by target frame and its context
    """
    return stride * (2 * k + 1) + frame_width

def read_phonemes(path="data/phonemes.txt"):
    """Reads in text file of phonemes and makes dict to convert to idx

    Args:
        path (str, optional): Path to file. Defaults to "data/phonemes.txt".

    Returns:
        dict: Mapping int index of phoneme to string of phoneme
    """
    idx_to_phoneme = dict()
    with open(path, "r") as f:
        for line in f:
            phoneme, idx = line.split()
            idx_to_phoneme[int(idx)] = phoneme
    return idx_to_phoneme

def export_predictions_to_csv(preds):
    """Export list of predictions to formatted `.csv` file.

    Args:
        preds (list(np.array, np.array, ...)): Should be output of `predict()` 
        filename (str): Name/path of csv file to output
    """
    if not os.path.exists("submissions"):
        os.mkdir("submissions/")
    file_name = f"submissions/submission_{datetime.now()}.csv"
    with open(file_name, 'w', newline = '') as f:
        reader = csv.writer(f)
        reader.writerow(['id', 'dipti_aswath_phoneme_label'])
        reader.writerows(enumerate(preds))
    print(f"Wrote predictions to {file_name}")

def plot_loss(losses, num_batches, num_epochs=10):
    plt.rcParams['savefig.facecolor'] = 'white'

    plt.plot(losses, label=f"Final Train Loss: {losses[-1]}")
    batches = np.arange(0, len(losses)+1, step=num_batches)
    plt.xticks(batches, np.arange(len(batches)))
    plt.ylabel("Loss Value Per Batch")
    plt.xlabel("Epoch #")
    plt.legend()
    plt.title(f"Loss value per batch during training")
    
def plot_val_accuracy(val_accuracies, num_epochs=10):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), [100 * acc for acc in val_accuracies], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()
