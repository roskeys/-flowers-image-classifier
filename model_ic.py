import os

import numpy as np
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, accuracy, name):
    plt.figure()
    plt.title("Training loss")
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.savefig(f"plots/{name}_Loss_plot.png")
    plt.close()

    plt.figure()
    plt.title("Accuracy")
    plt.plot(accuracy)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.savefig(f"plots/{name}_Accuracy.png")
    plt.close()
class Myclassifier1(nn.Module):
    def __init__(self, out_dim, hidden=1024, p=0.5):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1152, hidden),
            nn.Dropout(p=p),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim)
        )
        self.classifier = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)


class Myclassifier2(nn.Module):
    def __init__(self, out_dim, hidden=1024, p=0.5):
        super().__init__()
        self.layer1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6272, hidden),
            nn.Dropout(p=p),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim)
        )
        self.classifier = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x1 = self.layer1_1(x)
        x2 = self.layer1_2(x)
        x3 = self.layer1_3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)

# Define classifier class
class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # Add the first layer, input to a hidden layer
        if isinstance(input_size, list):
            product = 1
            for i in input_size:
                product *= i
            input_size = product
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        x = nn.Flatten()(x)
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)

        x = self.output(x)

        return F.log_softmax(x, dim=1)


# Define validation function
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


# Define NN function
def make_NN(n_hidden, n_epoch, labelsdict, lr, device, model_name, trainloader, validloader, testloader, train_data,
            from_scratch, train_all_parameters, logger):
    name = model_name + ("_all" if train_all_parameters else "") + ("_scr" if from_scratch else "")
    logger.info(name)

    # Import pre-trained NN model
    if model_name == "mymodel1":
        n_in = 1
        n_out = len(labelsdict)
        model = Myclassifier1(out_dim=n_out, hidden=n_hidden[0], p=0.5)
    elif model_name == "mymodel2":
        n_in = 1
        n_out = len(labelsdict)
        model = Myclassifier2(out_dim=n_out, hidden=n_hidden[0], p=0.5)
    else:
        model = getattr(models, model_name)(pretrained=not from_scratch)
        # Freeze parameters that we don't need to re-train
        if not train_all_parameters and not train_all_parameters:
            for param in model.parameters():
                param.requires_grad = False

        # Make classifier
        if model_name == "densenet169":
            n_in = next(model.classifier.modules()).in_features
        else:
            n_in = next(model.fc.modules()).in_features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))

        n_out = len(labelsdict)
        model.classifier = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden)
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    if train_all_parameters or from_scratch or model_name == "mymodel1" or model_name == "mymodel2":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.to(device)
    start = time.time()

    epochs = n_epoch
    steps = 0
    running_loss = 0
    print_every = 40
    val_loss_list = []
    loss_list = []
    accuracy_list = []
    highest_accuracy = 0
    test = False
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, device)

                if accuracy / len(validloader) > highest_accuracy:
                    test = True
                    highest_accuracy = accuracy / len(validloader)

                logger.info(
                    f"Epoch: {e + 1}/{epochs} - Training Loss: {running_loss / print_every:.3f} - validation Loss: {test_loss / len(validloader):.3f} - Validation Accuracy: {accuracy / len(validloader):.3f}")
                loss_list.append(running_loss / print_every)
                val_loss_list.append(test_loss / len(validloader))
                accuracy_list.append(accuracy / len(validloader))
                running_loss = 0

                # Make sure training is back on
                model.train()

        if (e + 1) % 10 == 0:
            plot_loss(loss_list, val_loss_list, accuracy_list, name + "_" + str(e))
        if (e+1) % 50 == 0 or test:
            with torch.no_grad():
                test_loss, test_accuracy = validation(model, testloader, criterion, device)
            logger.info(
                f"Test loss:{test_loss / len(testloader):.3f} Test Accuracy:{test_accuracy / len(testloader):.3f}")
            test = False
    # Add model info
    model.classifier.n_in = n_in
    model.classifier.n_hidden = n_hidden
    model.classifier.n_out = n_out
    model.classifier.labelsdict = labelsdict
    model.classifier.lr = lr
    model.classifier.optimizer_state_dict = optimizer.state_dict
    model.classifier.model_name = model_name
    model.classifier.class_to_idx = train_data.class_to_idx

    plot_loss(loss_list, val_loss_list, accuracy_list, name)
    with torch.no_grad():
        test_loss, test_accuracy = validation(model, testloader, criterion, device)

    logger.info(f"Test loss:{test_loss / len(testloader):.3f} Test Accuracy:{test_accuracy / len(testloader):.3f}")

    logger.info(f'model: {model_name} - hidden layers: {n_hidden} - epochs: {n_epoch} - lr: {lr}')
    logger.info(f"Run time: {(time.time() - start) / 60:.3f} min")
    return model


# Define function to save checkpoint
def save_checkpoint(model, path):
    checkpoint = {'c_input': model.classifier.n_in,
                  'c_hidden': model.classifier.n_hidden,
                  'c_out': model.classifier.n_out,
                  'labelsdict': model.classifier.labelsdict,
                  'c_lr': model.classifier.lr,
                  'state_dict': model.state_dict(),
                  'c_state_dict': model.classifier.state_dict(),
                  'opti_state_dict': model.classifier.optimizer_state_dict,
                  'model_name': model.classifier.model_name,
                  'class_to_idx': model.classifier.class_to_idx
                  }
    torch.save(checkpoint, path)


# Define function to load model
def load_model(path):
    cp = torch.load(path)

    # Import pre-trained NN model 
    model = getattr(models, cp['model_name'])(pretrained=True)

    # Freeze parameters that we don't need to re-train 
    for param in model.parameters():
        param.requires_grad = False

    # Make classifier
    model.classifier = NN_Classifier(input_size=cp['c_input'], output_size=cp['c_out'], \
                                     hidden_layers=cp['c_hidden'])

    # Add model info 
    model.classifier.n_in = cp['c_input']
    model.classifier.n_hidden = cp['c_hidden']
    model.classifier.n_out = cp['c_out']
    model.classifier.labelsdict = cp['labelsdict']
    model.classifier.lr = cp['c_lr']
    model.classifier.optimizer_state_dict = cp['opti_state_dict']
    model.classifier.model_name = cp['model_name']
    model.classifier.class_to_idx = cp['class_to_idx']
    model.load_state_dict(cp['state_dict'])

    return model


def test_model(model, testloader, device='cuda'):
    model.to(device)
    model.eval()
    accuracy = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
