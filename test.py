from utils import RNN
import numpy as np
import torch
from utils import generating_input_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 20
hidden_size = 128
num_layers = 2
output_size = 31
sequence_length = 1

model = RNN(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load('model.ckpt'))

sequence_test = np.array([1, 2, 3, 3, 4, 5])
x_test, y_test = generating_input_output(sequence_test)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i_step in range(x_test.shape[1]):
        x_input_test = torch.from_numpy(x_test[:, i_step].reshape(-1, sequence_length, input_size)).to(device)
        outputs = model(x_input_test.float())

        print('speed: {} %'.format(outputs[0, -1]))
