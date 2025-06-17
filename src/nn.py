import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, inputs, stride=1):
        super(ResidualBlock, self).__init__()

        # Shortcut connection
        self.l1 = nn.Linear(inputs, inputs)
        # self.l2 = nn.Linear(inputs, inputs)

        self.aticv = nn.ReLU()

    def forward(self, x):
        y = self.l1(x)
        # y = self.aticv(y)**3
        # y = self.l2(y)
        y = self.aticv(y)**3 + x
        return y


class DoubleSigmoid(nn.Module):
    def __init__(self):
        super(DoubleSigmoid, self).__init__()

        # Shortcut connection
        self.shift = nn.Parameter(torch.rand(1, dtype=torch.float32))

    def forward(self, x):
        s = nn.Sigmoid()
        return self.shift * s(x - 5) + (1 - self.shift) * s(x + 5)


class GaussianCDFActivation(nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        """
        Initialize the Gaussian CDF activation function.

        Parameters:
        - mean: The mean of the Gaussian distribution (default is 0).
        - std: The standard deviation of the Gaussian distribution (default is 1).
        """
        super(GaussianCDFActivation, self).__init__()

    def forward(self, x):
        return (1 + torch.erf(x)) / 2


class FunctionModel(nn.Module):
    def __init__(self, layer_sizes, activation='linear'):
        super(FunctionModel, self).__init__()
        self.layers = nn.ModuleList()
        self.layer_sizes = layer_sizes
        self.activation = activation
        for i in range(len(layer_sizes) - 1):
            # self.layers.append(ResidualBlock(layer_sizes[i]))

            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

            if i < len(layer_sizes) - 2:  # No activation after the last layer
                self.layers.append(nn.Softplus())

        if activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif activation == 'tanh':
            self.layers.append(nn.Tanh())
        elif activation == 'gcdf':
            self.layers.append(GaussianCDFActivation())
        elif activation == 'relu':
            self.layers.append(nn.ReLU())
        elif activation == 'softplus':
            self.layers.append(nn.Softplus())
        elif activation == 'doublesigmoid':
            self.layers.append(DoubleSigmoid())
        elif activation == 'linear':
            pass
        else:
            print('Warning! Activation function is unavailable! Using Linear by default!')

    def forward(self, x: torch.float32):
        for layer in self.layers:
            x = layer(x)
        return x

    def initialize_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Initialize weights and biases to zero
                nn.init.xavier_normal_(layer.weight)


# Function to compute the integral I[v]


def save_model(model: FunctionModel, model_path, config_path):
    with open(config_path, 'w') as file:
        # Write the list of numbers
        file.write("Layer size: " +
                   ', '.join(map(str, model.layer_sizes)) + '\n')
        # Write the string content
        file.write("Activation: " + model.activation + '\n')
    print(f"Model saved to {config_path}")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Load the model


def load_model(model_path, config_path):
    print(f"Configuration loaded from {config_path}")
    with open(config_path, 'r') as file:
        for line in file:
            if line.startswith("Layer size:"):
                # Extract the numbers and convert them to a list of integers
                numbers_part = line[len("Layer size: "):].strip()
                layer_sizes = list(map(int, numbers_part.split(', ')))
            elif line.startswith("Activation:"):
                # Extract the string content
                activation = line[len("Activation: "):].strip()
    model = FunctionModel(layer_sizes, activation)
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    return model


if __name__ == '__main__':
    layers = [2, 4, 4, 2, 1]
    actication = 'sigmoid'
    model_path = 'test.pth'
    config_path = 'config.txt'

    q = FunctionModel(layers, actication)

    save_model(q, model_path, config_path)

    qq = load_model(model_path, config_path)

    a = torch.tensor([1, 1], dtype=torch.float32)
    print(q(a), qq(a))
    q0 = q
    q = qq
    qq = q0
    q.initialize_weights()
    print(q(a), qq(a))
