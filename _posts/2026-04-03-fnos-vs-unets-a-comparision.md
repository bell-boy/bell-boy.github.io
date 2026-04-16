---
layout: post
title: "fourier neural operators vs unets: a comparison [wip]"
usemathjax: true
featured: true
featured_order: 2
---

_tl;dr: Fourier Neural Operators (FNOs) and U-Nets are two classes of Image to Image Networks. I explore their performance on a semantic segmentation task. Code can be viewed [here](https://colab.research.google.com/drive/1TLfvFH6_nG3WNUh7UZB0CUE9E_kuKYnb?usp=sharing)_

# introduction

_many thanks to brooks bryant and aditya madhan for reading over earlier versions of this article and giving me ideas for experiments_

UNets are a class of neural networks that are widely used for _image to image_ tasks, tasks where the input is an image and the output is also an image. 
A great example of an image to image task is _semantic segmentation_. 
Given an image $I \in \mathbb{R}^{H \times W \times C}$, the goal of semantic segmentation is to assign a class label to each pixel in the image, resulting in an output image $O \in \mathbb{Z}^{H \times W}$ where each pixel value corresponds to a class label.

![semantic segmentation]({{ site.baseurl }}/assets/images/fno-unet/semantic-seg.png)

Even with the rise of transformer-based architectures, UNets still remain a popular choice for image to image tasks, and have been used in a wide range of applications, from medical image analysis to satellite imagery.

UNets acheive their success by using a unique architecture that consists of an encoder and a decoder.
Leveraging the inductive bias of convolutional layers, the encoder captures local features and patterns in the input image, while the decoder reconstructs the output image from these features.
The architecture also includes skip connections that allow the network to retain high-resolution information from the encoder, which is crucial for tasks like semantic segmentation where fine details are important.

![unet architecture]({{ site.baseurl }}/assets/images/fno-unet/u-net-architecture.png)

Fourier Neural Operators (FNOs) are a newer class of neural networks that have been gaining attention in the field of scientific machine learning.

They are _operators_, which is the fancy mathematical way of indicating that FNOs are a special type of function. They don't operate on vectors like normal neural networks. Instead, they operate on _functions_.

The training dataset of an FNO consists of pairs of _functions_ $(f(\mathbf{x}), g(\mathbf{x}))$. An FNO is an operator that maps the input function $f: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}$ to the output function $g: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out'}}$.

In other words, The FNO $\mathcal{M}$ is trained so that for every $f$ in the dataset, $(\mathcal{M}f)(\mathbf{x})$ is a close to $g(\mathbf{x})$ as possible.

This is particularly useful in applications such as solving partial differential equations (PDEs), where the input and output are functions that describe physical phenomena.
For example, in fluid dynamics the input function could represent the initial conditions of a fluid flow, mapping $(x, y)$ cordinates to a scaler vorticity, while the output function could represent the velocity field at a later time.

![semantic segmentation]({{ site.baseurl }}/assets/images/fno-unet/vorticity.png)

The issue is that functions are typically represented in an _infinite-dimensional_ basis, which makes it difficult to apply traditional neural network architectures that are designed for finite-dimensional inputs and outputs.

As we'll see later, FNOs get around this issue by leveraging the Fourier transform to represent functions in the _frequency domain_, a finite-dimensional basis that (given certain conditions are met) can fully represent almost any function.

Things get interesting when we realize that images can be thought of as functions that map pixel coordinates in $\mathbb{R}^2$ to to color values in $\mathbb{R}^3$, which means that FNOs can be applied to image to image tasks as well.

This raises the question: how do FNOs compare to U-Nets on image to image tasks like semantic segmentation?

In this post, I try to explore this question by comparing FNO and U-Net performance on the [Cityscapes](https://huggingface.co/datasets/tanganke/cityscapes) semantic segmentation dataset. We'll go over FNOs, the dataset we're using, and run some small experiements to compare the two models on this task. I've written this post assuming that the reader has some familiarity with Convolutional Neural Networks and U-Nets, but is new to FNOs.

You can recreate the experiments in this post by following along with the accompanying [notebook](https://colab.research.google.com/drive/1TLfvFH6_nG3WNUh7UZB0CUE9E_kuKYnb?usp=sharing).

# FNOs

The _output_ of an FNO layer takes the form

\\[v_{t + 1}(x) = \sigma(W_tv_t(x)  + (\mathcal{F}_{w_t}v_t)(x))\\]

It's important to realize that $v_{t + 1}$ is not the hidden layer, it is the _output_ of the  hidden layer. FNOs operate directly on functions and their _hidden states_ are also functions.

$v_{t + 1}$ maps from the original input function $f$'s input space $\mathbb{R}^{in}$ to an intermediate output space $\mathbb{R}^{hidden}$.
This means that $v_{t + 1}(x)$ is just a vector in $\mathbb{R}^{hidden}$

So the above equation is telling you that given both an input to the original function $x$, the previous function $v_t$, and the weights of the current layer $W_t$ and $w_t$, this is how you compute the output $v_{t + 1}(x)$ in the hidden space. 

Now let's break the equation down.

$\mathcal{F}_{w_t}$ is the heart of the FNO, a parameterized convolution in the frequency domain, but what does that actually mean?

Given two functions $f$ and $g$, their convolution is defined as:
\\[(f * g)(x) = \int f(y)g(x-y)dy\\]

In modern deep learning, we typically use discrete convolutions[^1], which are defined as:
\\[(f * g)[n] = \sum_{m} f[m]g[n-m]\\]

Discrete convolution has many desirable properties, such as translation eqivariance and the ability to capture local patterns in the input function, 
but it's not invariant to the resolution of the input and output functions. Additionally, it can be computationally expensive to capture long-range dependencies in the input function using discrete convolution, as it requires using large kernels or deep architectures.

To get around these issues, let's first start by noting that all functions[^2]
can be represented a finite weighted sum of sinusoids oscillating at different frequencies, which is the basis of the Fourier transform[^3].

This representation is known as the frequency domain representation of the function, and it allows us to represent functions in a way that is independent of the discretization used for the input and output functions. 
It additionally has the nice property that convolution in the spatial domain corresponds to pointwise multiplication in the frequency domain, which allows us to perform convolution efficiently.

We'll define our weights $w_t$ to be the frequency domian parameterization of some function. So $w_t : R^{in} \rightarrow R^{hidden}$ is stored as the coeffecients of the sinusoids. We can define $(\mathcal{F}_{w_t}v_t)(x) = \int v_t(x)w_t(x - y)dy$ and compute it as follows[^4]:
1. Take the Fourier transform of the input function $v_t$ to get its representation in the frequency domain.
2. Multiply the Fourier coefficients of $v_t$ with the Fourier coefficients of $w_t$.
3. Take the inverse Fourier transform of the result to get the output function in the spatial domain.

For efficiency reasons, we keep only the lowest $k$ Fourier modes, which allows us to capture long-range dependencies in the input function without using large kernels or deep architectures.

The trade off is that each layer loses the ability to independantly capture high-frequency information in the input function, which can be important for tasks like semantic segmentation where fine details are important.

![filters]({{ site.baseurl }}/assets/images/fno-unet/filters.png)


The outer part of the equation is a _residual connection_ commonly used in neural networks. 

$\sigma: \mathbb{R}^{hidden} \rightarrow \mathbb{R}^{hidden}$ is a non-linearity and $W: R^{hidden \times hidden}$ is a matrix that operates on hidden vectors. We add the linearly transformed output of the previous layer to the output of the FNO. For FNOs this also has the added benefit of preserving higher freqeuncy components that would've been removed by the convolution layer.

# dataset

We'll specifically be working with the 8-class version of the Cityscapes dataset, which includes the following classes:
- Flat: Essentially the ground, including roads, sidewalks, and parking lots.
- Construction: Slightly misleading name, any man-made structure that doesn't fit into the other categories, such as buildings,
bridges, fences, tunnels, etc.
- Nature: Vegitation and grassy areas not mant to be driven on.
- Vehicle: Cars, trucks, buses, motorcycles, bicycles, etc.
- Sky: Open sky.
- Object: Traffic signs, traffic lights, and poles.
- Human: People, including pedestrians and cyclists.
- Void: This class is used for pixels that do not belong to any of the other classes, such as the background or areas and objects that are not relevant to the task.

The goal is to take a $128 \times 256$ image from the dataset and predict a $128 \times 256$ segmentation mask, where each pixel in the mask corresponds to one of the 8 classes.

![segmentation-example]({{ site.baseurl }}/assets/images/fno-unet/segmentation_example.png)

During training, I apply extensive augmentations (flips, shears, rotations, translations, etc) to the data to improve generalization performance.

# empirical comparison

The original plan for the comparison was to train both models on the dataset for 10 minutes and compare their performance, but this ignores a crucial bias towards the UNet -- decades of research and engineering that've gone into optimizing the forward and backward passes of convolutional neural networks.

FNOs on the other hand aren't just unoptimized, but are acutally _handicapped_ by their use of complex weights. Modern GPUs have Tensor Cores which dramatically speed up matrix multiplications, but only for real valued matricies. The FNOs can't take advantage of that.

The result is that a UNet can be much larger than an FNO and still take more steps in 10 minutes.

So new plan: We'll train both models for 28 epochs and compare their performance.

I've hand tuned the hyperparameters for each model. Ideally I would've done a more systematic hyperparameter search, but I only had so much time (and gpu credits!) for this project :p.

I spent _much_ more time tuning the parameters for the FNO than I did for the UNet.

I had trained an FNO first and spent an afternoon tuning it to improve performance until I'd hit a validation accuracy of about 55%, and then the very first UNet I trained after that brutally mogged it, getting to 62% after the first 100 steps and finishing at ~82% in 10 minutes.

This was the last time I changed the UNet hyperparamters.

After a long weekend, I had tuned the FNO to the point where it could get ~67% in 35 minutes. 
Changes included:
1. Dramatically reducing the number of fourier modes per layer, FNOs can compose frequencies to produce higher frequency filters.
2. Dramatically increasing hidden dimension size. Causes huge increases in computational cost, but also causes huge corresponding gains in performance.
4. Slightly changing the architecture: Instead of adding the convolution output to the previous state then performing a ReLU followed by a BatchNorm, I perform the ReLU on the convolution output followed by addition to the previous state and a BatchNorm. This was motivated by noticing that the original form probably had a harder time expressing the identity. 

![metrics]({{ site.baseurl }}/assets/images/fno-unet/fno-vs-unet-metrics.png)
![example]({{ site.baseurl }}/assets/images/fno-unet/fno-vs-unet-sample.png)

There is much more to do! The next step for really comparing performance is using [nightly torch code to squeeze out extra performance on complex ops](https://github.com/pytorch/pytorch/issues/125718) and trying out more architectural tweaks, but that's outside the scope of this post.

I think this is moderate evidence that for normal image to image tasks in Computer Vision, a UNet will probably outperfrom an FNO for a fixed amount of data or compute.

# why does unet mog so hard?

I don't know! In this section I do a couple of ablations to try to get to the answer.

## are spatial convolutions just better?

Let's define a Fully Convolutional Network (FCN), it's identical to the FNO but with one key difference. We replace $\mathcal{F}_{w_t}$ with a normal CNN convolution.

Below are the results of training both architectures on 14 epochs of the data, with minimal tuning.

![metrics]({{ site.baseurl }}/assets/images/fno-unet/fcn.png)
![example]({{ site.baseurl }}/assets/images/fno-unet/fcn-img.png)

They perform almost identically! Why?

Hypotheses:
1. The issue is depth. The resolution downsampling allows us to train a much deeper model given our limited memory. Downsample and upsample inside of a much deeper FNO and we should be able to close the gap.
2. Depth alone isn't the problem. The multi-scale architecture allows UNets to learn broad features across the whole image _and_ local features as well. Downsampling alone won't change anything, spatial convolution is just better.

## unets sans skip

To test this, let's create a new architecture: UNet Sans Skip (USS).
It's a UNet, but we don't copy over the outputs from one side of the U to the other. Information soley flows through the bottleneck dimension at the bottom of the architecture.






<hr>

[^1]: Well really deep learning uses cross-correlation, which is defined as: \\[(f \star g)[n] = \sum_{m} f[m]g[n+m]\\] But the two operations are very similar, and the difference is not important for our purposes.
[^2]: More precisely, if we assume that the function is periodic and satisfies certain regularity conditions, then it can be represented as a discrete weighted sum of sinusoids.
[^3]: Here's a [link]() to learn more, also check out this cool [3b1b video](https://www.youtube.com/watch?v=spUNpyF58BY) for a visual introduction
[^4]: This is a bit of a simplification; the function $w_t$ doesn't map to the same space as $v_t$, it maps to the space of square matricies operating on the same space as $v_t$. Also things are a bit more complex since our function inputs are vector valued. Check out the original [paper](https://arxiv.org/pdf/2010.08895) for more details.
