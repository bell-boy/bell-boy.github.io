---
layout: post
title: "fourier neural operators vs u-nets: a comparison"
usemathjax: true
featured: true
featured_order: 2
---

_tl;dr: Fourier Neural Operators (FNOs) and U-Nets are two classes of Image to Image Networks. I explore their performance on a semantic segmentation task. Code can be viewed [here](https://colab.research.google.com/drive/1TLfvFH6_nG3WNUh7UZB0CUE9E_kuKYnb?usp=sharing)_

# introduction

U-Nets are a class of neural networks that are widely used for _image to image_ tasks, tasks where the input is an image and the output is also an image. 
A great example of an image to image task is _semantic segmentation_. 
Given an image $I \in \mathbb{R}^{H \times W \times C}$, the goal of semantic segmentation is to assign a class label to each pixel in the image, resulting in an output image $O \in \mathbb{R}^{H \times W}$ where each pixel value corresponds to a class label.

[example image here]

Even with the rise of transformer-based architectures, U-Nets still remain a popular choice for image to image tasks, and have been used in a wide range of applications, from medical image analysis to satellite imagery.

U-Nets acheive their success by using a unique architecture that consists of an encoder and a decoder.
Leveraging the inductive bias of convolutional layers, the encoder captures local features and patterns in the input image, while the decoder reconstructs the output image from these features.
The architecture also includes skip connections that allow the network to retain high-resolution information from the encoder, which is crucial for tasks like semantic segmentation where fine details are important.

[architecture image here]

Fourier Neural Operators (FNOs) are a newer class of neural networks that have been gaining attention in the field of scientific machine learning.
Instead of operating on images, FNOs map functions to functions. 
Given an input function $f: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}$, the goal of an FNO is to learn a mapping from the input function to the output function $g: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out'}}$.
This is particularly useful in applications such as solving partial differential equations (PDEs), where the input and output are functions that describe physical phenomena.
For example, in fluid dynamics, the input function could represent the initial conditions of a fluid flow, and the output function could represent the velocity field at a later time.

[example image here]

A big issue with this paradigm is that it's not immediately clear how to learn a mapping from infinite-dimensional function spaces, which is where the Fourier transform comes in.
FNOs leverage the Fourier transform to represent operations on functions in the _frequency domain_ as a sum of sine and cosine functions, which in theory allows them to actually operate on infinite-dimensional function spaces regardless of the discretization used for the input and output functions.

Mathematically, that looks like this: $$v_{t + 1}(x) = \sigma(Wv_t(x)  + (\mathcal{F}_{w_t}v_t)(x))$$

Where $v_t$ is the input function at layer $t$, $W$ is a learnable weight matrix, $\sigma$ is a non-linear activation function, and $\mathcal{F}_{w_t}$ is a parameterized convolution.

This works because we can do convolution in the frequency domain _very_ easily, and the Fourier transform allows us to represent functions in a way that is independent of the discretization used for the input and output functions.

In theory, this means that FNOs should be invariant to the resolution of the input and output functions, which is a desirable property for many applications.
The transform to fourier space also allows FNOs to cheaply capture long-range dependencies in the input function, which can be difficult for traditional convolutional neural networks to capture without using very large kernels or deep architectures.

[illustrating image]

Things get interesting when we realize that images can be thought of as functions that map pixel coordinates to pixel values, which means that FNOs can be applied to image to image tasks as well.

This raises the question: how do FNOs compare to U-Nets on image to image tasks like semantic segmentation?

In this post, I try to explore this question by comparing FNO and U-Net performance on the [Cityscapes](https://huggingface.co/datasets/tanganke/cityscapes) segmentation dataset. We'll go through the dataset, the model achitectures, and a couple experiements to see how the two models perform on this task.

You can recreate the experiments in this post by following along with the accompanying [notebook](https://colab.research.google.com/drive/1TLfvFH6_nG3WNUh7UZB0CUE9E_kuKYnb?usp=sharing).

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

[interactive example]
