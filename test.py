from utils import LSTM
from utils import RNN
import numpy as np
import torch
from utils import generating_input_output
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transition_matrix = np.load('transition_matrix.npy')

num_options = 5

xticks = ['1', '2', '3', '4', '5']
yticks = ['11', '12', '13', '14', '15',
          '21', '22', '23', '24', '25',
          '31', '32', '33', '34', '35',
          '41', '42', '43', '44', '45',
          '51', '52', '53', '54', '55']

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

plt.imshow(transition_matrix, cmap='magma', interpolation='nearest')
plt.xticks(np.arange(num_options),xticks, fontsize=9)
plt.yticks(np.arange(num_options**2), yticks, fontsize=9)
plt.colorbar()

plt.savefig('transition_matrix.pdf')
plt.show()

input_size = 15
hidden_size = 128
num_layers = 1
output_size = 21
sequence_length = 1

model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('model_LSTM.ckpt'))

sequence_test1 = np.array([[1, 2, 3, 4, 5],
                           [4, 3, 3, 4, 5],
                           [1, 2, 3, 4, 5],
                           [4, 2, 1, 5, 3],
                           [4, 2, 1, 2, 3],
                           [2, 5, 4, 3, 3],
                           [2, 5, 4, 2, 1]])

sequence_test2 = np.array([[1, 2, 3, 3, 4, 5],
                           [4, 3, 3, 3, 4, 5],
                           [1, 2, 3, 3, 4, 5],
                           [4, 2, 1, 1, 5, 3],
                           [4, 2, 1, 1, 2, 3],
                           [2, 5, 4, 4, 3, 3],
                           [2, 5, 4, 4, 2, 1]])
# x_test, y_test = generating_input_output(sequence_test)

output_test1 = np.zeros((2, sequence_test1.shape[0]))
output_test2 = np.zeros((3, sequence_test2.shape[0]))

# x_test1, _ = generating_input_output(sequence_test1[0])
# print(x_test1)

# Test the model

model.eval()
with torch.no_grad():

    for i_sequence in range(sequence_test1.shape[0]):

        x_test1, _ = generating_input_output(sequence_test1[i_sequence, :])
        x_test2, _ = generating_input_output(sequence_test2[i_sequence, :])

        for i_step in range(x_test1.shape[1]):
            x_input_test = torch.from_numpy(x_test1[:, i_step].reshape(-1, sequence_length, input_size)).to(device)
            outputs = model(x_input_test.float())

            output_test1[i_step, i_sequence] = outputs[0, -1]

        for i_step in range(x_test2.shape[1]):
            x_input_test = torch.from_numpy(x_test2[:, i_step].reshape(-1, sequence_length, input_size)).to(device)
            outputs = model(x_input_test.float())

            output_test2[i_step, i_sequence] = outputs[0, -1]


test1_mean = np.zeros(3)
test1_mean[np.array([0, 2])] = np.mean(output_test1, axis=1)
test2_mean = np.mean(output_test2, axis=1)

test1_std = np.zeros(3)
test1_std[np.array([0, 2])] = np.std(output_test1, axis=1)
test2_std = np.std(output_test2, axis=1)

n = 3
ind = np.arange(n)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
test1 = ax.bar(ind, test1_mean, width, color=[0.5, 0.5, 1], yerr=test1_std, error_kw=dict(lw=5, capsize=5, capthick=3))
test2 = ax.bar(ind + width, test2_mean, width, color=[1, 0.5, 0.5], yerr=test2_std, error_kw=dict(lw=5, capsize=5, capthick=3))

# add some text for labels, title and axes ticks
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('G1', 'G2', 'G3'))

ax.legend((test1[0], test2[0]), ('overlap', 'non overlap'))

plt.savefig('LSTM.pdf')
plt.show()

