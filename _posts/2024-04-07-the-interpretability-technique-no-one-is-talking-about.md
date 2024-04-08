---
title: the interpretablity technique no one is talking about
cnt: "0x0C"
layout: post
tags: AI
usemathjax: true
---
*Yes, I am aware I'm three weeks behind on posts. It's been a busy couple of weeks*

[This](https://arxiv.org/pdf/1806.07538.pdf) is the coolest paper I've seen in a while, yet I've never heard of the technique. It's not mentioned on blog posts about AI interpretability/AI safety, and I've found very few papers trying to build on the technique, but it seems like a very promising approach.

It's an intrinsic method, meaning that it achieves interpretability goals by making changes to model architecture, not by interpreting weights or activations.

non linear-linearity
====================

The paper starts by outlining what it *really* means to have an interpretation of a model.

1.  Explicitness: How easy is it to understand the interpretations of the model
2.  Faithfulness: How faithful are the interpretations of the model to what's actually going on inside of it
3.  Stability: How consistent are explanations for neighboring examples[^1]

We actually already have a model which has explanations that satisfy all these criteria, and is still widely used to this day! The humble **Linear Model**. It's coefficient vector \\(\\pmb{\\theta}\\) tells you *exactly* how each input contributed to the output of the model and is *identical* for each input.

But for most classification and regression tasks, the linear model is shit.

The problem is, most functions we care about aren't linear. Now they might be *locally* linear, the same way that zooming in really close to a graph kind of looks like a straight line. But the formula \\(f(\\boldsymbol{x}) = \\boldsymbol{\\theta}^T\\boldsymbol{x}\\) can only ever *globally* be a straight line (or an \\(N\\) dimensional analogue), even as the function may do wildly non-linear things.

However, what if we just train a model to compute \\(\\pmb{\\theta}\\) for each input? The authors of the paper propose using a Deep Neural Network they call \\(\\pmb{\\theta}(\\pmb{x})\\) to be the coefficients of the linear model for a given input.

Intuitively, this feels like trying to approximate the derivative of some underlying function instead of the actual function.[^2]

However, there's still one big problem. The dimensionality of some problems is mind boggling. Even if we knew how much each pixel contributed to the overall output, how much would we care?

refined representations
=======================

Our model is theoretically more interpretable, but it tells us how much each *pixel* contributes to the final answer. Frankly, these kind of pixel level shallow attribution techniques have been shown to be [not that robust](https://arxiv.org/pdf/1906.07983.pdf) or [all that useful](https://www.alignmentforum.org/s/a6ne2ve5uturEEQK7/p/f8nd9F7dL9SxueLFA). Yet the authors of the paper go one step further. They replace \\(\\pmb{x}\\) with the latent space of a *sparse auto encoder*. The auto encoder part ensures that all the data of the original is present in compressed form, while sparsity is meant to ensure the features we find are meaningful.

So now our 'roided up linear function has two new parts. A compression function that turns high dimensional data into low dimensional interpretable concept data that we can then use as inputs to our linear function (we call the compressor/encoder of the autoencoder \\(h(x)\\)[^3]. It now looks like this

\\\[f(x) = \\theta(x)^Th(x)\\\]

 But wait, there's more![^4]

locality
========

We need to ensure our functions are locally interpretable. In layman's terms, we need to ensure that for similar inputs, our functions generate similar interpretations.

The authors achieve this by a trying to ensure that a function is *locally difference bounded* (see definition 3.2 in the paper for the full definition of this). Roughly meaning that for two close concepts, the distance between outputs is always less than the distance between inputs (multiplied by some arbitrary, *locally-consistent* constant factor).

To enforce this the authors define a regularization term 

\\\[\\mathcal{L}_\\theta(f(x)) = \\lVert \\nabla\_x f(x) - \\theta(x)^TJ\_x^h(x) \\rVert\\\]

 Which we try to minimize when training.

I will admit. The vector calculus they used to derive this function went over my head a bit, however, it makes intuitive sense. Our hyper powered linear function is kind of like the derivative of our function with respect to our input vector \\(h(x)\\). Using the chain rule, they *indirectly* measure the similarity of this function to the derivative of the function with respect to the inputs. Kind of a bullshit explanation, but I'll have to comeback and wrestle with the math more later.

so... does it work?
===================

Yes! Kind of.

The problem with using auto encoded features is that we have to interpret these features. Which is tricky. The authors look for dataset examples that maximally activate one feature (called prototypes) and then try to figure out the commonality between them.

![](https://39669.cdn.cke-cs.com/rQvD3VnunXZu34m86e5f/images/9dc62b55dfd214c7ad9e7db70e0f141be6f502fdff68e5ce.png)

For this six concept model trained to classify digits on MNIST, these prototypes are most representative of each of the generated concepts. At least for MNIST, this looks kind like something! Concept 1 seems to be straightness, and Concept 2 looks like roundness, but what's Concept 3? Sevenness? So it this might be the limiting factor of SENNs, but still, I expected it to get a lot more attention than it currently gets.

Less interestingly to me (but still very interesting), the authors of the paper propose 2 quantitative ways to evaluate interpretability methods.

Measure one evaluates methods by how important they say different features are to the model. They remove a feature and compare the drop in probability of the answer to the interp. method's prediction of that feature's importance.

Measurement two estimates the Lipschitz constant around a particular input. A lower Lipschitz constant means that explanations are more tightly bounded by their inputs. Thereby becoming more stable.[^5]

By the authors' own metrics, SENN beats out most methods for achieving interpretability.

next steps
==========

I would've expected people to have already been all over this. You wanted concept vectors? Here is a vector full of concepts! You want to understand *why* your model made a certain choice? Just take a peek at the coefficient vector dude. You want benchmarks and evaluations for interpretability methods? The paper literally provides quantitative ways to evaluate methods on two of the major criteria it posed. Yet there seems to be vary little attention around this paper.

So either A, I'm missing some Huge Obvious Flaw. Or B, there might need to be more work into seeing how competitive SENNs can be against plain jane NNs. How might one build a transformer out of SENNs? How did the authors perform classification with a SENN? If they used a Softmax, does the Softmax make interpreting things weird? How would trying to stack multiple SENNs work? How would this hold up against the other techniques Casper et. al used for [detecting trojans](https://arxiv.org/abs/2302.10894)?

However, I'm skeptical on the idea of playing with SENNs for a couple months. I think I'll get more value out of just reading more.

I'm interested in your thoughts!

[^1]:From what I understand, the motivation for this criterion is so that similar points have similar interpretations. However, think of a function like the sigmoid. There, some points that are close together just have different drastically different outputs. I'm skeptical on the value of this one.
    
[^2]:It isn't obvious to me why this would be true. But I plan to implement the architecture on a lower dimensional problem and find out how valid this line of thinking is.
    
[^3]:h(x) doesn't *have* to be an auto-encoder. The authors also suggest expert-generated features. However, from [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) style intuitions, I doubt that expert-generated features are the way to go for long term, scalable performance. 
    
[^4]:Quite a bit more. But I choose cover the parts I find to be the most interesting. I recommend reading the full paper.
    
[^5]:This measurement actively peaks my interest. Could a high Lipschitz constant for a *model* indicate a vulnerability to adversarial attacks?