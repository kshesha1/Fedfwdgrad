import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
#from cnnfwd import train_fwdgrad, SimpleCNN
import hydra
import math
import time
from functools import partial
from collections import OrderedDict
import torch.func as fc
import torchvision
import torchvision.transforms as transforms
from functorch import make_functional
from omegaconf import DictConfig
#import flwr as fl
from fwdgrad.loss import functional_xent, xent
from fwdgrad.model import NeuralNet,ConvNet

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)
    
def load_MNIST():
    transform = [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    transform.append(torchvision.transforms.Lambda(lambda x: torch.flatten(x)))
    mnist_train = torchvision.datasets.MNIST(
        "/tmp/data",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(transform),
    )
    mnist_test = torchvision.datasets.MNIST(
        "/tmp/data",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(transform),
    )
    input_size = mnist_train.data.shape[1] * mnist_train.data.shape[2]
    output_size = len(mnist_train.classes)
    train_loader = DataLoader(mnist_train,batch_size=64,shuffle=True,num_workers=8,pin_memory=True,drop_last=False)
    test_loader = DataLoader(mnist_test,batch_size=64,shuffle=True,num_workers=8,pin_memory=True,drop_last=False)
    return train_loader, test_loader, input_size, output_size

def train_fwd(model,train_loader, total_epochs):
    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()
    steps = 0
    t_total = 0.0
    init_lr = 2e-4
    k = 1e-4
    for epoch in tqdm(range(total_epochs)):
        t0 = time.perf_counter()
        for batch in train_loader:
            steps += 1
            images, labels = batch
             # Sample perturbation (tangent) vectors for every parameter of the model
            v_params = tuple([torch.randn_like(p) for p in params])
            f = partial(
                functional_xent,
                model=model,
                names=names,
                buffers=named_buffers,
                x=images.to(DEVICE),
                t=labels.to(DEVICE),
            )
             # Forward AD
            loss, jvp = fc.jvp(f, (tuple(params),), (v_params,))
             # Forward gradient + parmeter update (SGD)
            lr = init_lr * math.e ** (-steps * k)
            for p, v in zip(params, v_params):
                p.sub_(lr * jvp * v)
#            writer.add_scalar("Loss/train_loss", loss, steps)
#            writer.add_scalar("Misc/lr", lr, steps)
        t1 = time.perf_counter()
        t_total += t1 - t0
#        writer.add_scalar("Time/batch_time", t1 - t0, steps)
#        writer.add_scalar("Time/sps", steps / t_total, steps)
        print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
    print(f"Mean time: {t_total / total_epochs:.4f}")

def test_fwd(model, test_loader):
    named_buffers = dict(model.named_buffers())
    named_params = dict(model.named_parameters())
    names = named_params.keys()
    params = named_params.values()
    acc = 0
    for batch in test_loader:
        images, labels = batch
        out = fc.functional_call(model, (named_params, named_buffers), (images.to(DEVICE),))
        pred = F.softmax(out, dim=-1).argmax(dim=-1)
        acc += (pred == labels.to(DEVICE)).sum()
    print(f"Test accuracy: {(acc / len(test_loader.dataset)).item():.4f}")
    return float(acc / len(test_loader.dataset))

def train_back(model, train_loader, total_epochs):
    steps = 0
    t_total = 0.0
    init_lr = 2e-4
    k = 1e-4
    params = tuple(model.parameters())
    for epoch in tqdm(range(total_epochs)):
        t0 = time.perf_counter()
        for batch in train_loader:
            steps += 1
            images, labels = batch
            loss = xent(model, images.to(DEVICE), labels.to(DEVICE))
            loss.backward()
            lr = init_lr * math.e ** (-steps * k)
            for p in params:
                p.data.sub_(lr * p.grad.data)
                p.grad = None
#           writer.add_scalar("Loss/train_loss", loss, steps)
#            writer.add_scalar("Misc/lr", lr, steps)
        t1 = time.perf_counter()
        t_total += t1 - t0
#       writer.add_scalar("Time/batch_time", t1 - t0, steps)
#       writer.add_scalar("Time/sps", steps / t_total, steps)
        print(f"Epoch [{epoch+1}/{total_epochs}], Loss: {loss.item():.4f}, Time (s): {t1 - t0:.4f}")
    print("Mean time:", t_total / total_epochs)

def test_back(model, test_loader):
    acc = 0
    for batch in test_loader:
        images, labels = batch
        out = model(images.to(DEVICE))
        pred = F.softmax(out, dim=-1).argmax(dim=-1)
        acc += (pred == labels.to(DEVICE)).sum()
#   writer.add_scalar("Test/accuracy", acc / len(mnist_test), steps)
    print(f"Test accuracy: {(acc / len(test_loader.dataset)).item():.4f}")
    return float(acc / len(test_loader.dataset))


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


trainloader, testloader = load_data()

# Load model and data (simple CNN, CIFAR-10)
with torch.no_grad():
    net = Net().to(DEVICE)
#net= NeuralNet(input_size=input_size, hidden_sizes = [1024], output_size=output_size)
#net.to(DEVICE)
#net.float()
#net.train()
# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("start fiting")
        with torch.no_grad(): train_fwd(net, trainloader, 10)
        print("end fiting")
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        with torch.no_grad(): accuracy = test_fwd(net, testloader)
        print("Accuracy:", accuracy)
        return 0.5, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address="0.0.0.0:8080",
    client=FlowerClient(),
)