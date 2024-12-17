import pandas as pd
import yaml
from data.news_dataset import NewsDataset
from data.test_sample import TestSamples
from model.neucf import NeuCF
from torch.utils.data import DataLoader
from training_task.training_neucf import TrainingNeuCF
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--full_data_file", type=str, required=True)
parser.add_argument("--train_file", type=str, required=True)
parser.add_argument("--val_file", type=str, required=True)
parser.add_argument("--test_file", type=str, required=True)
parser.add_argument("--save_name", type=str, required=True)

args = parser.parse_args()

with open(args.config_file, 'rb') as f:
    config = yaml.safe_load(f)

df = pd.read_csv(args.full_data_file)
train = pd.read_csv(args.train_file)
test = pd.read_csv(args.test_file)
val = pd.read_csv(args.val_file)

train_data = NewsDataset(config, train)
val_data = NewsDataset(config, val)
test_data = TestSamples(config, test, df)

train_loader = DataLoader(train_data,
                          batch_size=config['DATA']['BATCH_SIZE'],
                          collate_fn=train_data.collate_fn)
val_loader = DataLoader(val_data,
                        batch_size=1,
                        collate_fn=val_data.collate_fn)
test_loader = DataLoader(test_data,
                         batch_size=1,
                         collate_fn=test_data.collate_fn)

model = NeuCF(config)
task = TrainingNeuCF(config, model)
task.start(train_loader, val_loader)

output = task.evaluation(test_loader)
with open(args.save_name, 'w') as f:
    json.dump(output, f)