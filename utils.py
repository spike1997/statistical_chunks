from abc import ABC

import numpy as np
import warnings
import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_options = 5
num_presses = 1000
chunk_indices = np.array([1, 4, 9, 13, 16, 17])
chucks = np.array([[0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0],
                   [1.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0, 0.0]])


def search_for_pattern(sequence, pattern):
    n = len(pattern)
    possibles = np.where(sequence == pattern[0])[0]

    indices = []
    for p in possibles:
        check = sequence[p:p + n]
        if np.all(check == pattern):
            indices.append(p)
    return indices


def estimating_transition_matrix(sequence):
    transition_matrix_estimation = np.zeros((num_options**2, num_options))

    for i in range(1, num_options + 1):
        for j in range(1, num_options + 1):
            a = np.array([i, j])
            length = len(search_for_pattern(sequence, a))
            for k in range(1, num_options + 1):
                if length == 0:
                    warnings.warn("Poor estimation of transition matrix! Run the code again.")
                    transition_matrix_estimation[(i - 1) * num_options + j - 1, k - 1] = 0
                else:
                    a = np.array([i, j, k])
                    transition_matrix_estimation[(i - 1) * num_options + j - 1, k - 1] = \
                        len(search_for_pattern(sequence, a)) / length
    return transition_matrix_estimation


def generating_transition_matrix():
    possible_p_vectors = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                                   [0.8, 0.2, 0.0, 0.0, 0.0],
                                   [0.8, 0.1, 0.1, 0.0, 0.0],
                                   [0.7, 0.3, 0.0, 0.0, 0.0],
                                   [0.7, 0.2, 0.1, 0.0, 0.0],
                                   [0.6, 0.4, 0.0, 0.0, 0.0],
                                   [0.6, 0.3, 0.1, 0.0, 0.0],
                                   [0.6, 0.2, 0.2, 0.0, 0.0],
                                   [0.5, 0.5, 0.0, 0.0, 0.0],
                                   [0.5, 0.4, 0.1, 0.0, 0.0],
                                   [0.5, 0.3, 0.2, 0.0, 0.0]])

    transition_matrix = np.zeros((num_options ** 2, num_options))

    for i_row in range(num_options ** 2):
        if i_row == 5:
            columns = np.array([0, 2, 3])
            transition_matrix[i_row, columns] = possible_p_vectors[
                np.random.randint(possible_p_vectors.shape[0]), np.random.permutation(len(columns))]
        elif i_row == 7:
            columns = np.array([0, 1, 2, 4])
            transition_matrix[i_row, columns] = possible_p_vectors[
                np.random.randint(possible_p_vectors.shape[0]), np.random.permutation(len(columns))]
        elif i_row == 12:
            columns = np.array([0, 1, 2, 4])
            transition_matrix[i_row, columns] = possible_p_vectors[
                np.random.randint(possible_p_vectors.shape[0]), np.random.permutation(len(columns))]
        elif i_row == 22:
            columns = np.array([0, 1, 2, 4])
            transition_matrix[i_row, columns] = possible_p_vectors[
                np.random.randint(possible_p_vectors.shape[0]), np.random.permutation(len(columns))]
        elif i_row == 23:
            columns = np.array([0, 2, 3, 4])
            transition_matrix[i_row, columns] = possible_p_vectors[
                np.random.randint(possible_p_vectors.shape[0]), np.random.permutation(len(columns))]
        else:
            transition_matrix[i_row] = possible_p_vectors[
                np.random.randint(possible_p_vectors.shape[0]), np.random.permutation(num_options)]

    transition_matrix[chunk_indices] = chucks

    return transition_matrix


def generating_sequence(transition_matrix):

    sequence = np.zeros(num_presses)

    sequence[0] = np.random.randint(low=1, high=num_options + 1)
    sequence[1] = np.random.randint(low=1, high=num_options + 1)

    for i_sample in range(2, num_presses):
        p_vector = transition_matrix[int((sequence[i_sample - 2] - 1) * num_options + sequence[i_sample - 1] - 1)]
        sequence[i_sample] = np.random.choice(np.arange(num_options) + 1, 1, p=p_vector)

    # overlapping chunks! is there any in the sequence?
    chuck_indicator_vector = np.zeros_like(sequence)

    for i_chunk in range(len(chunk_indices)):
        b = np.remainder(chunk_indices[i_chunk]+1, num_options)
        a = (chunk_indices[i_chunk] + 1 - b) / 5 + 1
        c = np.where(chucks[i_chunk] == 1)[0][0] + 1
        chuck = np.array([a, b, c])
        chuck_indicator_vector[search_for_pattern(sequence, chuck)] = 1
        # print(len(search_for_pattern(sequence, chuck)))
    # np.sum(np.dot(chuck_indicator_vector, np.roll(chuck_indicator_vector, -2)))

    return sequence


def generating_input_output(sequence):

    total_step = num_presses - 3
    input_size = 20
    num_classes = 31
    n = 4

    x = np.zeros((input_size, total_step))
    y = np.zeros((num_classes, total_step))

    for i_step in range(total_step):

        visible_numbers = sequence[i_step:i_step + n]
        x[int(visible_numbers[0] - 1), i_step] = 1
        x[int(visible_numbers[1] + 5 - 1), i_step] = 1
        x[int(visible_numbers[2] + 10 - 1), i_step] = 1
        x[int(visible_numbers[3] + 15 - 1), i_step] = 1

        y[int(visible_numbers[0] + 0 - 1), i_step] = 1
        y[int(visible_numbers[0] + 5 - 1), i_step] = 1
        y[int(visible_numbers[1] + 10 - 1), i_step] = 1
        y[int(visible_numbers[1] + 15 - 1), i_step] = 1
        y[int(visible_numbers[2] + 20 - 1), i_step] = 1
        y[int(visible_numbers[3] + 25 - 1), i_step] = 1

        if (visible_numbers[:-1] == np.array([1, 2, 3])).all() or \
                (visible_numbers[:-1] == np.array([3, 4, 5])).all() or \
                (visible_numbers[:-1] == np.array([4, 2, 1])).all() or \
                (visible_numbers[:-1] == np.array([1, 5, 3])).all() or \
                (visible_numbers[:-1] == np.array([2, 5, 4])).all() or \
                (visible_numbers[:-1] == np.array([4, 3, 3])).all():
            y[30, i_step] = 1
    return x, y


class RNN(nn.Module, ABC):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out