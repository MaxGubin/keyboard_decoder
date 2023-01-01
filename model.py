from collections.abc import Sequence, Tuple

import torch
from torch import nn

import keyboard_simulator
from tqdm import tqdm


class KeyboardDecoder(nn.Module) :
    """Simple keyboard decoder
    """

    def __init__(self, layer_sizes:Sequence[Tuple(int, int)], output_size:int):
        super(KeyboardDecoder, self).__init__()
        self.fully_connected_layers = [nn.Linear(ms[0],ms[1]) for ms in layer_sizes]
        self.relus = [nn.ReLU()]*len(layer_sizes)
        self.output_layer = nn.Linear(layer_sizes[-1][1], output_size)

    def forward(self, x):
        for linear, relu in zip(self.fully_connected_layers, self.relus):
            x = linear(x)
            x = relu(x)
        return self.output_layer(x)

def ConvertPointsToInput(points: Sequence[keyboard_simulator.CGPoint]) -> torch.Tensor:
    output = torch.tensor([[p.x,p.y] for p in points], dtype=torch.float32)
    return output


def BuildOptimizer(model):
    """Creates an optimizer"""
    return torch.optim.Adam(model.parameters(), lr=0.01)

def BuildModel():
    """Builds a model and initialize parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KeyboardDecoder(layer_sizes=[(2,128), (128, 128), (128, 128)], output_size=keyboard_simulator.NUM_CLASSES).to(device)
    return model

def TrainLoop(model:nn.Module, optimizer:torch.optim.Optimizer, num_steps:int):
    loss_fn = nn.CrossEntropyLoss()
    for n_batch in range(num_steps):
        points, labels = keyboard_simulator.random_batch_sample(128)
        data_points = ConvertPointsToInput(points)
        data_labels = torch.tensor(labels, dtype=torch.long)

        logits = model(data_points)
        loss = loss_fn(logits, data_labels)
        acc = (logits.argmax(1) == data_labels).type(torch.float32).sum().item()/128


        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if n_batch % 10 == 0:
            loss_item = loss.item()
            print(f"Batch {n_batch} Loss {loss_item} Acc {acc}")


def main():
    model = BuildModel()
    optimizer = BuildOptimizer(model)
    state = TrainLoop(model, optimizer, 5000)

if __name__ == '__main__':
    main()