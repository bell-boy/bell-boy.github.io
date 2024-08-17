---
title: Epileptic Uncertainty Framework technical report
layout: post
---

I recently spent a couple months working on a research project. The goal was to submit to NeurIPS' new High School track. I'd never done something like this, and I learned a lot about doing ML research by trying.

I'll split this post into two parts. This part will be more technical, describing the results. I really don't feel like writing another research paper about this, so this blog post will have to do. 

The next part will be about the journey, describing small tricks and meta-skills I learned about doing ML research. I think that post will be helpful if you're like I was, young and excited to try research. 

# Introduction

The theme of NIPS' HS track was something like "ML for social good", asking that we write a paper applying ML to some socital problem. A couple of friends of mine had recently been working on a seizure prediction project, so we decided to try adapting that.

The main problem that seizure prediction is trying to solve is that around 20-40% of people with epilepsy have *refactory* epilepsy. Meaning, drugs don't work well to treat them. There are many options for treatment, but seizure prediction aims to improve the quality of life of the afflicted, by at least telling them in advance that a seizure will occur. This way, they can receive preemptive treatment or get somewhere safe to have the seizure. 

The main way seizures are predicted is under what I'll call the *Standard Framework* (SF). Under the SF, seizure prediction takes electroencephalogram (EEG) data and uses machine learning to predict if a seizure will happen in a window of time specified by two numbers. The *Seizure Prediction Horizon* (SPH) and the *Seizure Occurance Period* (SOP). 

When we make a prediction, we're saying that within SOP minutes, but not before SPH minutes, a seizure will occur. These two numbers are hyper parameters, and are fixed before training. This effictivly makes seizure prediction under the SF a classification problem. 

[image here]

Research has suggested that there is no optimal SPH/SOP for all people.[^1] Instead, these numbers vary from paitent to paitent. 

In this work, I propose the Epileptic Uncertainty Framework (EUF). In the EUF we keep the SOP, but redefine it as the Maximum Prediction Horizon (MPH). Additonally, we predict a *distribution* over possibile times that the seizure will occur. Making seizure prediction under the EUF a classification *and* regression problem.

I acheive similar classification results to prior work, and very calibrated regression distributions, as seen by brier score. We by training the model to predict a distribution, we go from being able to answer the question "What is the probablity I will have a seizure in this specific time window" to being able to answer "What is the probablity I'll have a seizure within x minutes", where x is controlled by the paitent.

# Methodology

## Dataset

I use the CHBMIT seizure dataset.[^2] The dataset consists of >300 hours of EEG data over 23 patients, with labeled seizure start and end times. 

The EEG data has 23 channels and is broken up into multiple files that are around 1 hour long. In between each of the files are gaps that are on average around ~3 seconds long.

I draw samples that are ten minutes long. To extend the use of the dataset, I draw *overlapping* samples. When a sample crosses over a file gap, I just fill it with zeros. I run a filtering process on the whole dataset to filter out samples that are more than 3% zeros. 

The seizure signals are very weak, so I multiply them by $ 10^3 $ and run them through the Short-Time Fourier Transform (STFT). STFT turns the raw signals into spectrograms, which gives the model time and frequency information. In theory, the model can implement it's own STFT, but I find that you need an extra OOM of parameters to do the job of the STFT.

The end result of that is a 23 channel image. I resize all the images into a 256 by 256 image and pass it to the network.

The end result is a tensor of shape $ (23, 256, 256) $. Which is passed to the model.

I only train on a single patient due to not wanting to work on this project for another week.

## Epileptic Uncertainty Framework

Under EUF, the model is represented as a function $ f $ which maps *data* to *distribution parameters*. What $ f $ is approximating is the conditional distribution $ p(t | \mathbf{X}) $. In english, "the probablity density of a seizure happening in exactly $ t $ seconds from now given the EEG data"

We assume that the conditional distribution is from a family of distributions $ \mathcal{F} $. Each distribution in this family is uniquely defined by a parameter vector $\mathbf{\theta}$. So our model will take in the EEG data $ \mathbf{X} $ and return the parameter vector $ \mathbf{\theta} $ specifying the distribution which we'll use to approximate the conditional distribution. 

For this task, we assume that epliepsy is a *possion process* and model the distribution over the next seizure time as an *exponential distribution*. The parameter vector is just a single value $\lambda$ and the distribution is 

$$ p'(t) = \lambda \exp(-\lambda t)$$

However, to make training easier, we use the MPH to bound how far the model has to predict. So the model actually outputs the vector

$$ 
\begin{bmatrix}
  \omega & \mathbf{\theta}
\end{bmatrix} 
$$

$\mathbf{\theta}$ is still the same parameter vector as before, but $\omega$ is the probablity that a seizure happpens in the MPH at all. Phrased mathematically 

$$ p(\Omega) = \omega $$

Where $\Omega$ is the event where a seizure occurs within the MPH.

Now, instead of approximating the conditional, we now approximate 

$$ p(t | \mathbf{X}, \Omega) $$

In english, "the probability density that a seizure happens in $ t $ seconds given the data and that a seizure occurs within the MPH."

So the probablity density at $ t $ is 

$$ p(t | \mathbf{X}, \Omega)p(\Omega) $$

## Model

I use a model very similar in archecture to AlexNet.I adjusted in the nubmer channels that had to pass through the model, and some filter sizing. I also make the model *Multi-Head*. Meaning for each parameter in the output vector, a seperate set of MLP layers is set to extract different information from the shared layers without interfering with each other.

[image]

I find that single head training doesn't work very well. The model has to struggle to maintain performane on all of the tasks. The classification head is trained seperately from the others.

## Training

To train the model, I use the Maximum Likelihood Objective. We're given a dataset $ \mathcal{D} = \{(\mathbf{X}_i, t_i)\}_N $ consisting of $N$ sameples which we assume are independent of each other, but identically distributed. 

The likelihood is equal to 

$$ \prod_{i=1}^N p'(t | \mathbf{X}, \Omega)p'(\Omega) $$

To make maximizing easier, we move to log space, and, as is tradition, make everything negaitve and switch to minimizing. For the exponential distribution, this means that the optimization criterion $\mathcal{L}$ is 

$$ \mathcal{L}(\mathcal{D}) = \sum_{i=1}^N \lambda t_i - \ln \lambda - \ln \omega $$

Out of curiosity, I tried a different distribution. The Normal distribution. There isn't really a good argument to use this distribution though, I just found interesting results when training. The Normal criterion is 

$$ \mathcal{L}_{Normal}(\mathcal{D}) = \sum_{i=1}^N \frac{(\mu - t_i)^2}{\sigma^2} + \frac{\ln 2 \pi \sigma^2}{2} - \ln \omega $$

A major problem when using multiple heads is *Catastrophic Forgetting*. The performance of one head goes down as the other head goes up. To minimize this, I use the regularization provided in the paper Learning Without Forgetting.[^3]

It's an additional term added to the loss that prevents the predicted distribution of the classification head from it's original state, while still being flexible enough to learn the new task. The loss is simply 

$$ D_\textrm{KL}(\omega_{new} || \omega_{old}) $$

In english, "The KL divergence between the distributions predicted by the new and previous model"

Note that is distribution is the *classification distribution* denoted by $ \omega $. It's a Boltzmann distribution over 2 states, a seizure will happen within the MPH, or not.

This loss works very well to prevent Catastrophic Forgetting, without any hyperparameter tuning!

[^1]: Bandarabadi, Mojtaba, Jalil Rasekhi, César A. Teixeira, Mohammad R. Karami, and António Dourado. “On the Proper Selection of Preictal Period for Seizure Prediction.” Epilepsy & Behavior 46 (May 1, 2015): 158–66. https://doi.org/10.1016/j.yebeh.2015.03.010.

[^2]: Ali Shoeb. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, September 2009.