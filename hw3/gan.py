from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        modules = []
        # We found a good tutorial on the net, which helped us
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        insize = in_size[1]

        modules.append(
            nn.Conv2d(in_size[0], insize, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(insize))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.Conv2d(insize, insize * 2,
                                 kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(insize * 2))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.Conv2d(insize * 2, insize * 4,
                                 kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(insize * 4))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.Conv2d(insize * 4, insize * 8,
                                 kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(insize * 8))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        modules.append(nn.Conv2d(insize * 8, 1, kernel_size=4,
                                 stride=1, padding=0, bias=False))

        self.discriminator = nn.Sequential(*modules)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        device = next(self.parameters()).device
        y = self.discriminator(x).view(-1, 1).to(device)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======

        insize = 64
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(z_dim, insize * 8, kernel_size=4,
                               stride=1, padding=0, bias=False),
            nn.BatchNorm2d(insize * 8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(insize * 8, insize * 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(insize * 4),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(insize * 4, insize * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(insize * 2),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(insize * 2, insize, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(insize),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(insize, out_channels,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        # sample noise to generate fake images.
        z = torch.randn(n, self.z_dim, device=device)
        if with_grad is True:
            samples = self.forward(z)
        else:
            samples = self.forward(z).data
        samples = samples.to(device)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        device = next(self.parameters()).device
        z_unflatten = z.view(-1, self.z_dim, 1, 1).to(device)
        x = self.generator(z_unflatten)
        x = x.to(device)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    # Distribution of labels
    distance = torch.distributions.uniform.Uniform(
        data_label - label_noise / 2, data_label + label_noise / 2)
    y_real_data_fuzzy = distance.sample(y_data.shape).to(device)

    # Flip 1 to 0 etc.
    data_label = abs(1 - data_label)
    distance = torch.distributions.uniform.Uniform(
        data_label - label_noise / 2, data_label + label_noise / 2)
    y_generated_fake_fuzzy = distance.sample(y_generated.shape).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    loss_data_from_real = loss_function(y_data, y_real_data_fuzzy)
    loss_generated_from_fake = loss_function(
        y_generated, y_generated_fake_fuzzy)
    # ========================
    return loss_data_from_real + loss_generated_from_fake


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    size_of_y_generated = y_generated.size()[0]
    loss_function = nn.BCEWithLogitsLoss()
    loss = loss_function(y_generated, torch.full(
        (size_of_y_generated,), data_label).to(device))
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dimension = x_data.shape[0]
    dsc_optimizer.zero_grad()
    y_data = dsc_model.forward(x_data).view(-1)
    generated_x = gen_model.sample(dimension, with_grad=False)
    generated_y = dsc_model.forward(generated_x).view(-1)
    dsc_loss = dsc_loss_fn(y_data, generated_y)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    generated_x = gen_model.sample(dimension, with_grad=True)
    generated_y = dsc_model.forward(generated_x).view(-1)
    gen_loss = gen_loss_fn(generated_y)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f'{checkpoint_file}-dsc_losses-{dsc_losses[-1]}-gen_losses-{gen_losses[-1]}.pt'

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    if dsc_losses[-1] <= 0.8 and gen_losses[-1] <= 0.8:
        gen_model.save(checkpoint_file)
    # ========================

    return saved
