# Based https://towardsdatascience.com/shakespeare-meets-googles-flax-ecbd16f9d648
# and https://www.machinelearningnuggets.com/jax-flax-lstm/

import argparse
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
        self.fully_connected_layers = nn.ModuleList([
            nn.Linear(ms[0], ms[1]) for ms in input_layer_sizes])
        self.relus = nn.ModuleList([nn.ReLU()]*len(input_layer_sizes))
        lstm_input_size = input_layer_sizes[-1][-1]
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True) if lstm_num_layers > 0 else None
        self.output_layer = nn.Linear(
            lstm_hidden_size if lstm_num_layers > 0 else lstm_input_size, output_size)

    def forward(self, input):
        x = input
        for linear, relu in zip(self.fully_connected_layers, self.relus):
            x = linear(x)
            x = relu(x)
        lstm_out, _ = self.lstm(x) if self.lstm else (x, None)
        return self.output_layer(lstm_out)


def ConvertPointsToInput(points: Sequence[keyboard_simulator.CGPoint]) -> torch.Tensor:
    """Converts a sequence of points into a tensor"""
    output = torch.tensor([[p.x, p.y] for p in points], dtype=torch.float32)
    return output


def AddTypingNoise(coordinates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simple, not correlated noise, returns noised coordinates and a tensor of positions outside keys"""
    bias = torch.normal(mean=0.0, std=0.05, size=(1, 2))
    typing_noise = torch.normal(mean=0.0, std=0.05, size=coordinates.shape)
    typing_noise += bias
    # A naive estimation of how many points are outside of a key
    # assuming that every key has 2*X_PRECISION 2*Y_PRECISION
    precision_tensor = torch.tensor(
        [[keyboard_simulator.X_PRECISION, keyboard_simulator.Y_PRECISION]], dtype=torch.float32)
    out_of_key = ((typing_noise.abs() > precision_tensor).int().sum(dim=1) > 0).int()
    noised_typing = coordinates+typing_noise
    return noised_typing, out_of_key


def PadCollider(batch):
    (xx, yy) = zip(*batch)
    xx, errors = zip(
        *[AddTypingNoise(ConvertPointsToInput(x[:MAX_LEN])) for x in xx])
    yy = [torch.tensor(y[:MAX_LEN], dtype=torch.long) for y in yy]
    xx_pad = nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy_pad = nn.utils.rnn.pad_sequence(
        yy, batch_first=True, padding_value=keyboard_simulator.PAD_ID)
    errors_pad = nn.utils.rnn.pad_sequence(
        errors, batch_first=True, padding_value=0)
    return xx_pad, yy_pad, errors_pad


def CalcAccuracyAndNoiseLevel(logits_flatten: torch.Tensor, y_flatten: torch.Tensor, missed_keys:torch.Tensor) -> Tuple[torch.Tensor, int]:
    # get the index of the max probability
    max_preds = logits_flatten.argmax(dim=1, keepdim=True)
    non_pad_elements = (y_flatten != keyboard_simulator.PAD_ID).nonzero()
    correct = max_preds[non_pad_elements].squeeze(
        1).eq(y_flatten[non_pad_elements])
    noise_level = missed_keys.view(-1)[non_pad_elements].sum()
    return (correct.sum(), noise_level, y_flatten[non_pad_elements].shape[0])


def CalcPercentageErrors(errs):
    return (errs.sum(), errs.shape[0])


def CreateModel() -> nn.Module:
    """Creates a model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DecoderLSTM(((2, 128), (128, 128)), 512, 2,
                        keyboard_simulator.NUM_CLASSES).to(device)
    return model


def TrainLSTMModel(train_dataloader: torch.utils.data.DataLoader, restore_checkpoint: str, checkpoint_pass: str):
    model = CreateModel()
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(ignore_index=keyboard_simulator.PAD_ID)
    if restore_checkpoint:
        checkpoint = torch.load(restore_checkpoint)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print('Restored_checkpoint from ', restore_checkpoint)
    for step, (x, y, errs) in enumerate(train_dataloader):
        logits = model(x)
        # flatten output.
        logits_flatten = logits.view(-1, keyboard_simulator.NUM_CLASSES)
        y_flatten = y.view(-1)
        loss = loss_fn(logits_flatten, y_flatten)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        (accuracy_sum, noise_sum, accuracy_denom) = CalcAccuracyAndNoiseLevel(logits_flatten, y_flatten, errs)
        if step % 5 == 0:
            accuracy = accuracy_sum / accuracy_denom
            noise = noise_sum / accuracy_denom
            print("Step ", step, " Loss ", loss.item(), " Acc ", accuracy.item(), " Noise ", noise.item())
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()
            }, checkpoint_pass)

    # Save the last checkpoint.
    model_state = model.state_dict()
    torch.save({
        'step': step,
        'model_state': model_state,
        'optimizer_state': optimizer.state_dict()
    }, checkpoint_pass)


def EvalLSTMModel(eval_dataloader: torch.utils.data.DataLoader, checkpoint_pass: str):
    model = CreateModel()
    checkpoint = torch.load(checkpoint_pass)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    accuracy_sum_total = torch.tensor(0.0)
    error_level_total = torch.tensor(0.0)
    accuracy_denom_total = 0
    for x, y, errs in eval_dataloader:
        logits = model(x)
        logits_flatten = logits.view(-1, keyboard_simulator.NUM_CLASSES)
        y_flatten = y.view(-1)
        (accuracy_sum, error_level_sum, accuracy_denom) = CalcAccuracyAndNoiseLevel(logits_flatten, y_flatten, errs)
        accuracy_sum_total += accuracy_sum
        error_level_total += error_level_sum
        accuracy_denom_total += accuracy_denom

    accuracy_sum_total /= accuracy_denom_total
    error_level_total /= accuracy_denom_total
    print("Checked samples ", accuracy_denom_total,
          " accuracy ", accuracy_sum_total.item(),
          " noise level ", error_level_total.item())


def PrepareDataset(dataset_path: str, is_training: bool) -> torch.utils.data.DataLoader:
    data_file = open(dataset_path, 'r')
    input_lines = data_file.readlines()
    csv_dataset = keyboard_simulator.KeyboardDataset(input_lines)
    dataloader = torch.utils.data.DataLoader(csv_dataset, batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=0, collate_fn=PadCollider)
    return dataloader


def ParseArguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--command', choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint_path', default='keyboard_predictor.pt')
    parser.add_argument('--restore_checkpoint',
                        default='keyboard_predictor.pt')
    parser.add_argument('--dataset_path', default='train.txt')
    return parser.parse_args()


def Main():
    arguments = ParseArguments()
    is_training = arguments.command == 'train'
    training_dataloader = PrepareDataset(arguments.dataset_path, is_training)
    if is_training:
        TrainLSTMModel(training_dataloader,
                       arguments.restore_checkpoint, arguments.checkpoint_path)
    else:
        EvalLSTMModel(training_dataloader, arguments.checkpoint_path)


if __name__ == '__main__':
    Main()
