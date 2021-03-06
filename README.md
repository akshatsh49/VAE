
# VAE
A Pytorch implementation of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) on the MNIST dataset.

# Introduction
The paper introduces techniques for performing efficient learning in directed probabilistic models with continuous latent variables. The EM algorithm is not applicable in the general case because of intractable marginal distributions. VAE gets around this problem by optimization using stochastic gradient methods and fitting an approximate posterior inference model (the recognition model) to the intractable posterior.

# Results 
## Reconstructions

<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src="https://github.com/akshatsh49/VAE/blob/main/reconst_folder/Data.png" width=1000" />
<td><img src="https://github.com/akshatsh49/VAE/blob/main/reconst_folder/Reconstructions_11.png" width=1000" />
<td><img src="https://github.com/akshatsh49/VAE/blob/main/reconst_folder/Reconstructions_201.png" width=1000" />
</tr>
</table>

## Samples from gaussian prior

<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src="https://github.com/akshatsh49/VAE/blob/main/sample_folder/3.png" width=1000" />
<td><img src="https://github.com/akshatsh49/VAE/blob/main/sample_folder/101.png" width=1000" />
  <td><img src="https://github.com/akshatsh49/VAE/blob/main/sample_folder/201.png" width=1000" />
</tr>
</table>

## Estimated Lower Bound Objective (ELBO)
<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src="https://github.com/akshatsh49/VAE/blob/main/loss/Training_loss.png" width=1000" />
<td><img src="https://github.com/akshatsh49/VAE/blob/main/loss/Validation_loss.png" width=1000" />
</tr>
</table>

## Handpicked Linear Space Interpolation

<table align='center'>
<tr align='center'>
</tr>
<tr>
<td><img src="https://github.com/akshatsh49/VAE/blob/main/space_interpolations/201%20(1).png" width=1000" />
<td><img src="https://github.com/akshatsh49/VAE/blob/main/space_interpolations/201%20(2).png" width=1000" />
</tr>
</table>

## Model architecture overview
The encoder net is composed of 2 fully connected layers which parametrize the posterior distribution by returning the mean and log-variance of the recognition model.
The decoder net is composed of 2 fully connected layers which model the conditional distribution p(x|z). 
The model class contains an encoder and decoder object and returns the sufficient statistics for computation of the ELBO.

