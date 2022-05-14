import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
from typing import Union
import random
from timm.models.resnet import resnet18
from tqdm.auto import tqdm
from collections import defaultdict
from torchvision.transforms import Compose, Normalize
from datetime import datetime
import json

RANDOM_SEED = 0
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEFAULT_TRANSFORMS = Compose([
    # ToPILImage(),
    # ToTensor(),
    Normalize(0.5, 0.5)
])


class MNISTDataset(Dataset):
    def __init__(self, path: Union[Path, str], subset: float = 1., shuffle: bool = False,
                 transforms=DEFAULT_TRANSFORMS, device: torch.device = torch.device('cuda')):
        """
        Initialize an MNIST dataset. The dataset is taken from path and should be in the format of the kaggle MNIST
        data. If subset is a number between (0,1), return a stratified sample of subset percent of the data.
        :param path: Path to the dataset.
        :param subset: Percent of train to keep. Default is 1, keep all.
        :param shuffle: Shuffle the DataFrame or not. mostly useful when selecting a subset to select a
                        different subset every time.
        :param transforms: Transforms for the dataset object.
        :param device: Choose device (gpu/cpu).
        """
        super().__init__()
        df = pd.read_csv(path)
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        if 0 < subset < 1:
            df, _ = train_test_split(df, stratify=df.label, train_size=subset, random_state=RANDOM_SEED)
        elif subset != 1.:
            warnings.warn('ValueError in subset variable: subset <= 0 or subset > 1. Subset reset to 1.')

        self.core_df = df
        self.X = np.repeat(df[df.columns[1:]].values.reshape((-1, 1, 28, 28)).astype('uint8'), 3, 1)
        self.X = torch.tensor(self.X).to(device, dtype=torch.get_default_dtype())
        # self.X = np.stack([X, X, X], axis=-1)
        self.y = torch.tensor(df['label'].values).to(device, dtype=torch.uint8)
        # self.y = df['label'].values
        self.transforms = transforms
        self.device = device

    def __getitem__(self, item):
        return self.transforms(self.X[item]), self.y[item]

    def __len__(self):
        return self.y.shape[0]


def create_dataset(data_mode: str = 'train', subset: float = 1., shuffle: bool = False,
                   transforms=DEFAULT_TRANSFORMS, device: torch.device = torch.device('cuda')) -> MNISTDataset:
    """
    Create an MNIST dataset for the given data_mode (can be 'train' or 'test').
    :param data_mode: Selecting if the data is for train or test. Default is train.
    :param subset: Percent of train to keep. Default is 1, keep all.
    :param shuffle: suffle the DataFrame or not. mostly useful when selecting a subset to select a
                    different subset every time.
    :param transforms: Transforms for the dataset object.
    :param device: Choose device (gpu/cpu).
    :return: The relevant dataset.
    """

    core_path = Path(r'C:\Users\Yuval\Projects\university\thesis\code\datasets\mnist\yann_mnist')
    path = core_path / f'{data_mode}.csv'
    return MNISTDataset(path, subset, shuffle, transforms, device)


def is_jsonable(x) -> bool:
    """
    Check if x is JSON serializable.
    :param x: Object that we want to check if serializable.
    :return: Serializable or not (bool).
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


# TODO: implement
def train():
    pass


# TODO: implement
def evaluate():
    pass


if __name__ == '__main__':
    log_path = Path(rf'logdir\{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}')

    hyper_dict = {
        'epochs': 1,
        'learning_rate': 0.001,
        'batch_size': 128,
        'df_subset': 0.005,
        'criterion': nn.CrossEntropyLoss()
    }

    logs = {
        'train': defaultdict(list),
        'test': defaultdict(list)
    }

    gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpu else "cpu")
    cpu = torch.device("cpu")
    device = cpu
    print(device)

    ds_train = create_dataset(data_mode='train', subset=hyper_dict['df_subset'], device=device)
    ds_test = create_dataset(data_mode='test', device=device)
    dl_train = DataLoader(ds_train, batch_size=hyper_dict['batch_size'], pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=hyper_dict['batch_size'], pin_memory=True)

    res18 = resnet18()
    model = nn.Sequential(
        res18,
        nn.ReLU(),
        nn.Linear(in_features=res18.fc.out_features, out_features=10),
        nn.Softmax(dim=-1)
    )

    model.to(device)

    hyper_dict['optimizer'] = torch.optim.Adam(model.parameters(), lr=hyper_dict['learning_rate'])
    optimizer = hyper_dict['optimizer']
    criterion = hyper_dict['criterion']
    for epoch in range(hyper_dict['epochs']):

        running_loss = 0.0
        running_acc = 0.0
        running_correct = 0
        total_count = 0

        model.train()
        pbar = tqdm(dl_train, unit='batch')
        pbar.set_description(f'Epoch {epoch}')
        i = 0
        for inputs, labels in pbar:
            i += 1
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs.detach(), 1)
            running_correct += (preds == labels).sum().item()
            running_loss += loss.item()
            total_count += labels.size(0)
            running_acc = running_correct / total_count

            pbar.set_postfix(loss=running_loss / i, accuracy=running_acc)
        pbar.close()

        logs['train']['loss'].append(running_loss / len(dl_train))
        logs['train']['accuracy'].append(running_acc)

        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_acc = 0.0
            running_correct = 0
            total_count = 0

            pbar_eval = tqdm(dl_test, unit='batch')
            pbar_eval.set_description('Validation')
            for data, target in pbar_eval:
                images = data
                labels = target
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs.detach(), 1)
                running_correct += (preds == labels).sum().item()
                running_loss += loss.item()
                total_count += labels.size(0)
                running_acc = running_correct / total_count

                pbar_eval.set_postfix(loss=running_loss / i, accuracy=running_acc)
            pbar_eval.close()

        logs['test']['loss'].append(running_loss / len(dl_train))
        logs['test']['accuracy'].append(running_acc)

    print('Finished Training')

    log_path.mkdir(exist_ok=True)
    torch.save(model, log_path / 'model.pt')
    with open(log_path / 'logs.json', 'w') as fp:
        json.dump(logs, fp)
    with open(log_path / 'hyper_dict.json', 'w') as fp:
        json.dump({k: v if is_jsonable(v) else str(v) for k, v in hyper_dict.items()}, fp)
    ds_train.core_df.to_csv(log_path / 'train_df.csv', index=False)

print(f'log:\n{logs}')
    print('Done.')
