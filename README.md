# VAE
A Pytorch implementation of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) on the MNIST dataset.

# Introduction
The paper introduces techniques for performing efficient learning in directed probabilistic models with continuous latent variables. The EM algorithm is not applicable in the general case because of intractable marginal distributions. VAE gets around this problem by optimization using stochastic gradient methods and fitting an approximate posterior inference model (the recognition model) to the intractable posterior.

