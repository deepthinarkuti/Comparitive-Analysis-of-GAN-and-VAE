from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from torchvision import datasets, transforms
import os
import torch
import torch.utils.data



seed = 1

is_cuda = False

dimensions = 20

Size_of_batch = 128



torch.manual_seed(seed)
if is_cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'workers_count': 1, 'pin_memory': True} if is_cuda else {}
train_set_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=Size_of_batch, shuffle=True, **kwargs)

test_set_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.ToTensor()),
    batch_size=Size_of_batch, shuffle=True, **kwargs)

class VAE(nn.Module):
    def _init_(self):
        super(VAE, self)._init_()
        # ENCODER
        self.fullyConnected1 = nn.Linear(784, 400)
        self.relu = nn.ReLU()
        self.fullyConnected21 = nn.Linear(400, dimensions)  # mu layer
        self.fullyConnected22 = nn.Linear(400, dimensions)  # logvariance layer

        # DECODER
        self.fullyConnected3 = nn.Linear(dimensions, 400)
        self.fullyConnected4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x: Variable) -> (Variable, Variable):
        h1 = self.relu(self.fullyConnected1(x))  # type: Variable
        return self.fullyConnected21(h1), self.fullyConnected22(h1)

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)

        else:
            return mu

    def decoder(self, z: Variable) -> Variable:
        h3 = self.relu(self.fullyConnected3(z))
        return self.sigmoid(self.fullyConnected4(h3))

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

model = VAE()
if is_cuda:
    model.cuda()

def loss_function(recon_x, x, mu, logvar) -> Variable:
    BinaryCrossEntropy = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    KLDivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLDivergence /= Size_of_batch * 784
    return BinaryCrossEntropy + KLDivergence

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for indx_batch, (data, _) in enumerate(train_set_loader):
        data = Variable(data)
        if is_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        reconstruct_batch, mu, logvar = model(data)
        loss = loss_function(reconstruct_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if indx_batch % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, indx_batch * len(data), len(train_set_loader.dataset),
                100. * indx_batch / len(train_set_loader),
                loss.data / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_set_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_set_loader):
        if is_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(Size_of_batch, 1, 28, 28)[:n]])
          save_image(comparison.data.cpu(),
                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_set_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def main():
    os.chdir("./")

    EPOCHS = 16
    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)
        sample = Variable(torch.randn(64, dimensions))
        if is_cuda:
            sample = sample.cuda()
        sample = model.decoder(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')

main()
