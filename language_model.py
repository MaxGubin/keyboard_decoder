# Based https://towardsdatascience.com/shakespeare-meets-googles-flax-ecbd16f9d648
# and https://www.machinelearningnuggets.com/jax-flax-lstm/

from typing import Any, Tuple, Sequence

import torch
from torch import nn

import pandas as pd
import keyboard_simulator

BATCH_SIZE = 16
MAX_FEATURES = 10000
MAX_LEN = 50


class DecoderLSTM(nn.Module):
    def __init__(self, input_layer_sizes: Sequence[Tuple[int, int]], lstm_hidden_size: int, lstm_num_layers: int, output_size: int):
        super(DecoderLSTM, self).__init__()
        self.fully_connected_layers = [
            nn.Linear(ms[0], ms[1]) for ms in input_layer_sizes]
        self.relus = [nn.ReLU()]*len(input_layer_sizes)
        lstm_input_size = input_layer_sizes[-1][-1]
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        self.output_layer = nn.Linear(lstm_hidden_size, output_size)

    def forward(self, input):
        x = input
        for linear, relu in zip(self.fully_connected_layers, self.relus):
            x = linear(x)
            x = relu(x)
        lstm_out, _ = self.lstm(x)
        return self.output_layer(lstm_out)


def ConvertPointsToInput(points: Sequence[keyboard_simulator.CGPoint]) -> torch.Tensor:
    output = torch.tensor([[p.x, p.y] for p in points], dtype=torch.float32)
    return output


def PadCollider(batch):
    (xx, yy) = zip(*batch)
    xx = [ConvertPointsToInput(x[:MAX_LEN]) for x in xx]
    yy = [torch.tensor(y[:MAX_LEN], dtype=torch.long) for y in yy]
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = nn.utils.rnn.pad_sequence(
        yy, batch_first=True, padding_value=keyboard_simulator.PAD_ID)
    return xx_pad, yy_pad


def CalcAccuracy(logits_flatten: torch.Tensor, y_flatten: torch.Tensor) -> torch.Tensor:
    # get the index of the max probability
    max_preds = logits_flatten.argmax(dim=1, keepdim=True)
    non_pad_elements = (y_flatten != keyboard_simulator.PAD_ID).nonzero()
    correct = max_preds[non_pad_elements].squeeze(
        1).eq(y_flatten[non_pad_elements])
    batch_acc = correct.sum() / y_flatten[non_pad_elements].shape[0]
    return batch_acc


def TrainLSTMModel(train_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderLSTM(((2, 128), (128, 128)), 512, 2,
                        keyboard_simulator.NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=keyboard_simulator.PAD_ID)
    for x, y in train_dataloader:
        logits = model(x)
        # flatten output.
        logits_flatten = logits.view(-1, keyboard_simulator.NUM_CLASSES)
        y_flatten = y.view(-1)
        loss = loss_fn(logits_flatten, y_flatten)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = CalcAccuracy(logits_flatten, y_flatten)
        print("Loss ", loss.item(), " Acc ", accuracy.item())


def PrepareDataset() -> torch.utils.data.DataLoader:
    csv_file = pd.read_csv('./imdb_dataset.csv')
    csv_dataset = keyboard_simulator.KeyboardDataset(csv_file['review'])
    dataloader = torch.utils.data.DataLoader(csv_dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0, collate_fn=PadCollider)
    return dataloader


if __name__ == '__main__':
    training_dataloader = PrepareDataset()
    TrainLSTMModel(training_dataloader)
