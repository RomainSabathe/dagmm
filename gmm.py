"""Implements a GMM model."""

import torch
import numpy as np
from torch import nn


class GMM(nn.Module):
    """Implements a Gaussian Mixture Model."""
    def __init__(self, num_mixtures, dimension_embedding):
        """Creates a Gaussian Mixture Model.

        Args:
            num_mixtures (int): the number of mixtures the model should have.
            dimension_embedding (int): the number of dimension of the embedding
                space (can also be thought as the input dimension of the model)
        """
        super().__init__()
        self.num_mixtures = num_mixtures
        self.dimension_embedding = dimension_embedding

        mixtures = [Mixture(dimension_embedding) for _ in range(num_mixtures)]
        self.mixtures = nn.ModuleList(mixtures)

    def forward(self, inputs):
        out = None
        for mixture in self.mixtures:
            to_add = mixture(inputs, with_log=False)
            if out is None:
                out = to_add
            else:
                out += to_add

        return -torch.log(out)

    def _update_mixtures_parameters(self, samples, mixtures_affiliations):
        """
        Args:
            samples (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
            mixtures_affiliations (Variable of shape [batch_size, num_mixtures])
                the probability of affiliation of each sample to each mixture.
                Typically the output of the estimation network.
        """
        if not self.training:
            # This function should not be used when we are in eval mode.
            return

        for i, mixture in enumerate(self.mixtures):
            affiliations = mixtures_affiliations[:, i]
            mixture._update_parameters(samples, affiliations)



class Mixture(nn.Module):
    def __init__(self, dimension_embedding):
        super().__init__()
        self.dimension_embedding = dimension_embedding

        self.Phi = np.random.random([1])
        self.Phi = torch.from_numpy(self.Phi).float()
        self.Phi = nn.Parameter(self.Phi, requires_grad=False)

        # Mu is the center/mean of the mixtures.
        self.mu = 2.*np.random.random([dimension_embedding]) - 0.5
        self.mu = torch.from_numpy(self.mu).float()
        self.mu = nn.Parameter(self.mu, requires_grad=False)

        # Sigma encodes the shape of the gaussian 'bubble' of a given mixture.
        self.Sigma = np.eye(dimension_embedding, dimension_embedding)
        self.Sigma = torch.from_numpy(self.Sigma).float()
        self.Sigma = nn.Parameter(self.Sigma, requires_grad=False)

        # We'll use this to augment the diagonal of Sigma and make sure it is
        # inversible.
        self.eps_Sigma = torch.FloatTensor(
                        np.diag([1.e-8 for _ in range(dimension_embedding)]))


    def forward(self, samples, with_log=True):
        """Samples has shape [batch_size, dimension_embedding]"""
        # TODO: cache the matrix inverse and determinant?
        # TODO: so ugly and probably inefficient: do we have to create those
        #       new variables and conversions from numpy?
        batch_size, _ = samples.shape
        out_values = []
        inv_sigma = torch.inverse(self.Sigma)
        det_sigma = np.linalg.det(self.Sigma.data.cpu().numpy())
        det_sigma = torch.from_numpy(det_sigma.reshape([1])).float()
        det_sigma = torch.autograd.Variable(det_sigma)
        for sample in samples:
            diff = (sample - self.mu).view(-1, 1)
            #det_sigma = torch.from_numpy(det_sigma).float()

            out = -0.5 * torch.mm(torch.mm(diff.view(1, -1), inv_sigma), diff)
            out = (self.Phi * torch.exp(out)) / (torch.sqrt(2. * np.pi * det_sigma))
            if with_log:
                out = -torch.log(out)
            out_values.append(float(out.data.cpu().numpy()))

        out = torch.autograd.Variable(torch.FloatTensor(out_values))
        return out

    def _update_parameters(self, samples, affiliations):
        """
        Args:
            samples (Variable of shape [batch_size, dimension_embedding]):
                typically the input of the estimation network. The points
                in the embedding space.
            mixtures_affiliations (Variable of shape [batch_size])
                the probability of affiliation of each sample to each mixture.
                Typically the output of the estimation network.
        """
        if not self.training:
            # This function should not be used when we are in eval mode.
            return

        batch_size, _ = samples.shape

        # Updating phi.
        phi = torch.mean(affiliations)
        self.Phi.data = phi.data

        # Updating mu.
        num = 0.
        for i in range(batch_size):
            z_i = samples[i, :]
            gamma_i = affiliations[i]
            num += gamma_i * z_i
        denom = torch.sum(affiliations)
        self.mu.data = (num / denom).data

        # Updating Sigma.
        mu = self.mu
        num = None
        for i in range(batch_size):
            z_i = samples[i, :]
            gamma_i = affiliations[i]
            diff = (z_i - mu).view(-1, 1)
            to_add = gamma_i * torch.mm(diff, diff.view(1, -1))
            if num is None:
                num = to_add
            else:
                num += to_add

        denom = torch.sum(affiliations)
        self.Sigma.data = (num / denom).data + self.eps_Sigma
