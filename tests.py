import torch
import numpy as np

from gmm import Mixture, GMM
from model import DAGMMArrhythmia


def test_dagmm():
    net = DAGMMArrhythmia()

    input = np.random.random([5, 274])
    input = torch.autograd.Variable(torch.from_numpy(input).float())

    out = net(input)
    print(out)


def convert_to_var(input):
    out = torch.from_numpy(input).float()
    out = torch.autograd.Variable(out)
    return out


def test_update_mixture():
    batch_size = 5
    dimension_embedding = 7
    mix = Mixture(dimension_embedding)
    latent_vectors = np.random.random([batch_size, dimension_embedding])
    affiliations = np.random.random([batch_size])
    latent_vectors = convert_to_var(latent_vectors)
    affiliations = convert_to_var(affiliations)

    for param in mix.parameters():
        print(param)

    mix.train()
    mix._update_mixture_parameters(latent_vectors, affiliations)

    for param in mix.parameters():
        print(param)


def test_forward_mixture():
    batch_size = 5
    dimension_embedding = 7

    mix = Mixture(dimension_embedding)
    latent_vectors = np.random.random([batch_size, dimension_embedding])
    latent_vectors = convert_to_var(latent_vectors)

    mix.train()
    out = mix(latent_vectors)
    print(out)


def test_update_gmm():
    batch_size = 5
    dimension_embedding = 7
    num_mixtures = 2

    gmm = GMM(num_mixtures, dimension_embedding)

    latent_vectors = np.random.random([batch_size, dimension_embedding])
    latent_vectors = convert_to_var(latent_vectors)

    affiliations = np.random.random([batch_size, num_mixtures])
    affiliations = convert_to_var(affiliations)

    for param in gmm.parameters():
        print(param)

    gmm.train()
    gmm._update_mixtures_parameters(latent_vectors, affiliations)

    for param in gmm.parameters():
        print(param)


if __name__ == '__main__':
    #test_update_mixture()
    #test_forward_mixture()
    #test_update_gmm()
    test_dagmm()
