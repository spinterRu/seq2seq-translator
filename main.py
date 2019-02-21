from one_hot_encoder import OneHotEncoder

import train
import argparse
import torch
import test


parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', help='path to the training data file')
parser.add_argument('--model_path', help='path to the directory with models')
args = parser.parse_args()

# Vectorize the data.
input_texts = []
target_texts = []

with open(args.data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
i = 0

for line in lines:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    input_texts.append(input_text.lower())
    target_texts.append('\t' + target_text.lower().replace("ё", "е") + '\n')
    i = i + 1
    if i > 10000:
        break

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_encoder = OneHotEncoder(input_texts, device)

target_encoder = OneHotEncoder(target_texts, device)

train.train_iters(args.model_path, 10000, input_texts, target_texts, input_encoder, target_encoder, device, input_encoder.max_length)

samples = ["hi!", "hello!", "go away!", "really?", "listen", "back off!", "my arm hurts.", "you scare me.", "you will die"]
test.test(samples, input_encoder, args.model_path, device, target_encoder)