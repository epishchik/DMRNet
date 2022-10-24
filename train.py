import yaml
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from prettytable import PrettyTable

from model import DMRNet

import warnings
warnings.filterwarnings('ignore')


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


class Trainer:
    def __init__(self):
        self.parse_config()
        self.init_dataset()
        self.init_model()
        self.train()

    def parse_config(self):
        with open('./train.yaml') as stream:
            self.config = yaml.safe_load(stream)
        print('finished parsing')

    def init_dataset(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        self.batch_size = self.config['dataset']['batch_size']
        self.epochs = self.config['epochs']
        self.print_steps = self.config['dataset']['print_steps']
        self.num_workers = self.config['num_workers']

        trainset = datasets.CIFAR10(root='./data',
                                    train=True,
                                    download=True,
                                    transform=transform)

        self.trainloader = DataLoader(trainset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers)

        testset = datasets.CIFAR10(root='./data',
                                   train=False,
                                   download=True,
                                   transform=transform)

        self.testloader = DataLoader(testset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers)

        print('finished dataset initialization')

    def init_model(self):
        h = self.config['dataset']['image_size']['height']
        w = self.config['dataset']['image_size']['width']

        in_channels = self.config['model']['input_channels']
        convs = self.config['model']['convs_per_block']
        h_channels = self.config['model']['hidden_channels']
        outc = self.config['model']['output_channels']

        lr = self.config['optimizer']['learning_rate']
        wd = self.config['optimizer']['weight_decay']

        self.device = torch.device(self.config['device'])

        self.model = DMRNet(sizes=(h, w),
                            in_channels=in_channels,
                            convs=convs,
                            h_channels=h_channels,
                            out_channels=outc).to(self.device)

        self.model.apply(init_weights)

        if 'checkpoint' in self.config.keys():
            self.model.load_state_dict(torch.load(self.config['checkpoint']))

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=wd)

        self.criterion = nn.CrossEntropyLoss()
        self.save_path = self.config['save_path']

        print(self.model)
        self.count_parameters()
        print('finished model initialization')

    def count_parameters(self):
        table = PrettyTable(['modules', 'parameters'])
        total_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f'total trainable parameters: {total_params}')

    def train(self):
        best_val_err = 99999999999

        for epoch in range(self.epochs):
            running_loss = 0.0

            print('training')
            self.model.train()
            for i, data in enumerate(self.trainloader):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % self.print_steps == self.print_steps - 1:
                    val = running_loss / self.print_steps
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {val:.3f}')
                    running_loss = 0.0

            running_loss = 0.0

            print('validation')
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.testloader):
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    running_loss += loss.item()

                    loss_val = running_loss / (i % self.print_steps + 1)
                    if loss_val < best_val_err:
                        best_val_err = loss_val

                    if i % self.print_steps == self.print_steps - 1:
                        val = running_loss / self.print_steps
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {val:.3f}')
                        running_loss = 0.0

            save_name = f'dmrnet_{best_val_err:.3f}'
            torch.save(self.model.state_dict(), self.save_path + save_name)

        print('finished training')


if __name__ == '__main__':
    Trainer()
