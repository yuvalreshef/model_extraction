from collections import defaultdict

import pandas as pd
import torch
from torch import nn
from typing import Union
from pathlib import Path
from mnist_train import MNISTDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int = 100, label_embed_len: int = 100, n_classes: int = 10):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.label_condition_embedding = nn.Sequential(nn.Embedding(n_classes, label_embed_len),
                                                       nn.Linear(label_embed_len, 16))

        self.latent_embedding = nn.Sequential(nn.Linear(latent_dim, 4 * 4 * 512),
                                              nn.LeakyReLU(0.2, inplace=True))

        self.gen = nn.Sequential(nn.ConvTranspose2d(in_channels=513, out_channels=64 * 8, kernel_size=(4, 4),
                                                    stride=(2, 2), padding=(1, 1), bias=False),
                                 nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(in_channels=64 * 8, out_channels=64 * 4, kernel_size=(4, 4),
                                                    stride=(2, 2), padding=(1, 1), bias=False),
                                 nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(in_channels=64 * 4, out_channels=64 * 2, kernel_size=(4, 4),
                                                    stride=(2, 2), padding=(1, 1), bias=False),
                                 nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(in_channels=64 * 2, out_channels=64 * 1, kernel_size=(4, 4),
                                                    stride=(2, 2), padding=(1, 1), bias=False),
                                 nn.BatchNorm2d(64 * 1, momentum=0.1, eps=0.8),
                                 nn.ReLU(True),
                                 nn.ConvTranspose2d(in_channels=64 * 1, out_channels=1, kernel_size=(4, 4),
                                                    stride=(2, 2),
                                                    padding=(1, 1), bias=False),
                                 nn.Tanh())

    def get_latent_dim(self):
        return self.latent_dim

    def forward(self, inputs: torch.Tensor):
        noise_vector, label = inputs
        label_output = self.label_condition_embedding(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent_embedding(noise_vector)
        latent_output = latent_output.view(-1, 512, 4, 4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.gen(concat)
        return image


class ConditionalDiscriminator(nn.Module):
    def __init__(self, label_embed_len: int = 100, n_classes: int = 10):
        super(ConditionalDiscriminator, self).__init__()
        self.condition_embedding = nn.Sequential(nn.Embedding(n_classes, label_embed_len),
                                                 nn.Linear(label_embed_len, 3 * 128 * 128))

        self.disc = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64 * 2, kernel_size=(4, 4), stride=(3, 3), padding=(2, 2),
                      bias=False),
            nn.BatchNorm2d(64 * 2, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64 * 2, out_channels=64 * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(64 * 4, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64 * 4, out_channels=64 * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                      bias=False),
            nn.BatchNorm2d(64 * 8, momentum=0.1, eps=0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(4608, 1),
        )

    def forward(self, images: torch.Tensor, conditions: torch.Tensor):
        conditions_output = self.condition_embedding(conditions).view(-1, 3, 128, 128)
        concat = torch.cat((images, conditions_output), dim=1)
        output = self.model(concat)
        return output


class ModelConditionedGAN(object):
    def __init__(self, oracle_path: Union[Path, str], generator: nn.Module, discriminator: nn.Module,
                 data_path: Union[Path, str], hyper_dict: dict):
        super(ModelConditionedGAN, self).__init__()
        self.oracle: nn.Module = torch.load(oracle_path)
        self.gen: nn.Module = generator
        self.disc: nn.Module = discriminator
        self.trained: bool = False
        self.latent_dim = self.gen.get_latent_dim()
        self.train_ds = MNISTDataset(path=data_path)
        self.train_dl = DataLoader(self.train_ds, batch_size=hyper_dict["batch_size"], pin_memory=True, shuffle=True)
        self.batch_size, self.learning_rate,  = **hyper_dict

    def train(self, real_images: torch.Tensor, labels: torch.Tensor):
        self.gen.train()
        self.disc.train()

        logs = {
            'train': defaultdict(list),
            'test': defaultdict(list)
        }

        for epoch in range(self.hyper_dict['epochs']):
            gen_loss = 0.
            disc_loss = 0.
            total_count = 0

            pbar = tqdm(self.train_dl, unit='batch')
            pbar.set_description(f'Epoch {epoch}')
            i = 0
            for real_images, labels in pbar:
                i += 1
                self.hyper_dict["optimizer"].zero_grad()

                real_target = torch.ones(real_images.size(0), 1, requires_grad=False)
                fake_target = torch.zeros(real_images.size(0), 1, requires_grad=False)

                gen_images = self.gen((real_images, labels))

                fake_judgement = self.disc((gen_images, labels), fake_target)
                real_judgement = self.disc((real_images, labels), real_target)

                self.hyper_dict["G_optimizer"].zero_grad

                loss_D_fake = self.hyper_dict

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
        disc_real_output = self.disc((real_images, labels), )
        disc_fake_output = self.disc((real_images, labels))

    @torch.no_grad()
    def generate(self, conditions: torch.Tensor, generate_count: int = 1):
        self.gen.eval()
        z = torch.randn(generate_count, self.latent_dim)
        return self.gen((z, conditions))
