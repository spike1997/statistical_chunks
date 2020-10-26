import numpy as np
import matplotlib.pyplot as plt
from utils import search_for_pattern
from utils import generating_sequence
from utils import generating_input_output
from utils import RNN
import torch
import torch.nn as nn
from utils import generating_transition_matrix


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from utils import estimating_transition_matrix

num_options = 5
num_presses = 1000

# Hyper-parameters
sequence_length = 1
input_size = 15
hidden_size = 128
num_layers = 1
output_size = 21
batch_size = 1
num_epochs = 50
learning_rate = 0.01

model = RNN(input_size, hidden_size, num_layers, output_size).to(device)

# inp = torch.from_numpy(np.zeros((20, 1)).reshape(-1, sequence_length, input_size)).to(device)
# print(inp.shape)
# model(inp.float())
#
# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = num_presses-3

transition_matrix = generating_transition_matrix()

flag = 1
while flag:
    _, dist_chunk = generating_sequence(transition_matrix)
    if len(np.where(dist_chunk==0)[0]) == 0:
        flag = 0
    else:
        transition_matrix = generating_transition_matrix()

# print(dist_chunk)
#
# sequence, _ = generating_sequence(transition_matrix)
# print(type(sequence))
#
# c = search_for_pattern(sequence, [1, 5, 3])
#
# x, y = generating_input_output(sequence)
# #
# print(y[:,c[0]])

for epoch in range(num_epochs):
    sequence, _ = generating_sequence(transition_matrix)
    # transition_matrix_estimation = estimating_transition_matrix(sequence)
    # generating input output of the network
    x, y = generating_input_output(sequence)

    for i_step in range(total_step):

        x_input = torch.from_numpy(x[:, i_step].reshape(-1, sequence_length, input_size)).to(device)
        y_output = torch.from_numpy(y[:, i_step].reshape(-1, output_size)).to(device)

        # Forward pass
        outputs = model(x_input.float())
        loss = criterion(outputs, y_output.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i_step + 1) % total_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i_step + 1, total_step, loss.item()))


# print(outputs.shape)
torch.save(model.state_dict(), 'model.ckpt')
np.save('transition_matrix.npy',transition_matrix)

