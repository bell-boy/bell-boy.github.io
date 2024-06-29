---
title: three old dead statisticians
cnt: "0x0A"
layout: post
tags: AI productivity
usemathjax: true
---

Hi! This post took (a lot) longer to write than I expected. It's going to be more technical than usual, but the (two) readers of my blog should be smart enough to keep up. 

Disclaimer: While I'm confident that what I understand is true, I haven't studied statistics, linear algebra, or multivariable calculus. I've learned enough to understand the gist of most the papers I read, but I may lose finer details/nuance. I'll pepper in footnotes to the sources I draw this information from.  

# continual learning
Imagine we want a neural network to learn how to recognize digits in images. However we have two *distributions* of images. A green distributions where the images are all green, and a blue distribution where the images are all blue. The primary method for learning a task like this is to create a dataset of many different letter and image drawings and label them with the appropriate label. But what if this wasn't possible? What if we had two different datasets for the two different distribution, and we trained the network on each task sequentially? We'd probably experience a phenomenon known as **catastrophic forgetting** (researchers are so dramatic!) where the model almost immediately starts forgetting how to do previous tasks in order to learn a new one. 

Most solutions to mitigating this problem do so by retaining a small sample of data from old tasks and using those in the new training set. This way, the model is penalized for starting to forget old tasks. But for many reasons (privacy, storge limitations, etc.) this approach cannot always be employed. What if we can't re-train on old task data? How do we maintain model performance?

## defining the problem
To work toward an answer, lets start by defining the specific problem of continual learning. 

This is hard, as there are really *three* continual learning problems [^1], but this blog is already late so I choose to focus on the *domain-incremental* continual learning problem. We have multiple distributions of inputs, which we want to map to the same set of outputs. Formally 

$$P(X_1) \neq P(X_2)$$

However, 

$$\{Y_1\} = \{Y_2\}$$

## bludgeoned by bayes.  
One method for trying to solve this problem is Elastic Weight Contribution[^2]. EWC is based of the intuition that the brain learns new things by building new neural pathways and then making them resistant to change. So if we want to do something similar, we'd need to determine what weights are most important to the task at hand, and make them more resistant to change by the learning process of future tasks.  

To find these "important neurons" the paper describing EWC starts by considering training from a probabilistic perspective, where the goal is to maximize \\(P(\theta \| \mathcal{D})\\) (in english: the probablity of the model parameters given the dataset), where \\(\theta\\) represents the parameters of our model, and \\(\mathcal{D}\\) represents the dataset. **Bayes' Rule** tells us that

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D}|\theta)P(\theta)}{P(\mathcal{D})}$$

Okay, now this is where the first big trick of EWC comes in. We split the dataset into two parts, the data of the task we're currently training on \\(\mathcal{D}_B\\) and the data of the task we trained on previously \\(\mathcal{D}_A\\). So the equation above can be re-written as 

$$P(\theta | \mathcal{D}) = \frac{P(\mathcal{D}_B|\theta)P(\theta|\mathcal{D}_A)}{P(\mathcal{D}_B)}$$

When we're dealing maximization problems involving the multiplication of very small probabilities its a common aim to instead maximize the *natural log* of everything instead. Since the location of the max of the logarithm is the location of the max of the actual probablity, and the log simplifies calculations considerably. 

$$ \ln P(\theta | \mathcal{D}) = \ln P(\mathcal{D}_B | \theta) + \ln P(\theta | \mathcal{D}_A) - \ln (\mathcal{D}_B) $$

The first term \\(\ln P(\mathcal{D}_B\|\theta)\\)  is actually just the negative **loss** (\\(\mathcal{L}(\theta)\\)) of our model. Which is easily calculable and commonly used in training. The paper doesn't mention the final term \\(-\ln (\mathcal{D}_B)\\), I assume this is because that this probablity is usually fixed and not depend on the parameters. So all of the information about what weights are important for the previous problem are hidden within the term \\(\ln P(\theta \| \mathcal{D}_A)\\). However, finding the quantity directly might be intractable. Here is the second big contribution of the paper, they model the posterior probablity \\(\ln P(\theta \| \mathcal{D}_A)\\) using the *Laplace Approximation*[^3]

## losing to laplace

If you found the previous section difficult to follow, well, buckle up. 

Without getting too technical, Laplace's approximation allows us to approximate the area under functions that have most of their mass concentrated around a small region of their domain. In other words, we kind find the area under functions that look like this little hump.[^4] 

![](/assets/posts/0x0A/Gaussian Dist.png)

We approximate the function using the normal distribution (AKA the bell curve) which also looks like a hump. The normal distribution has two parameters, **mean** and **variance**. Lets assume we know the location of the maximum that we're trying to approximate. Let's call that value \\(x_0\\). We find the parameters of the normal distribution lurking sneakily within the function we're trying to approximate. 

Let's call the function we're trying to integrate \\(g(x)\\). Let's also define \\(h(x) = \ln g(x)\\). Then[^5] 

$$\int g(x)dx = \int \exp[h(x)]dx$$

Now let's do a 2-term Taylor series approximation of \\(h(x)\\) around \\(x_0\\), the max value we found earlier.

$$\int g(x)dx \approx \int \exp[h(x_0) + h'(x_0)(x-x_0)+\frac{1}{2}h''(x_0)(x-x_0)^2]dx$$

But we know that \\(x_0\\) is the max of the function, so \\(h'(x_0) = 0\\), and \\(h(x_0)\\) is just a constant value. So let's rewrite this integral with these simplifying observations.

$$\int g(x)dx \approx g(x_0) \int \exp[\frac{1}{2}h''(x_0)(x-x_0)^2] dx$$

If you've taken any level of statistics, or are just a general math nerd, the right hand expression should be setting off alarm bells in your head! It's *so* close to the equation for the normal distribution (\\(\frac{1}{\sqrt{2\pi\sigma^2}} \exp[\frac{1}{2} \cdot \frac{(x-\mu)^2}{\sigma^2}]\\), where \\(\mu\\) is the mean and \\(\sigma^2\\) is the variance)! In fact, with a little bit of algebraic manipulation, we can form the non-constant part of the normal distribution with the left hand side of our integral!

$$\int g(x)dx \approx g(x_0) \int \exp[-\frac{1}{2} \cdot \frac{(x-x_0)^2}{-h''(x_0)^{-1}}] dx$$

\\(x_0\\) is now our mean and \\(-h\'\'(x_0)^{-1}\\) is our variance. Let's let \\(\varphi(\mu,\sigma^2)\\) be the pdf of the standard normal distribution. Then we can replace the entire right side with 

$$ \int g(x)dx \approx g(x_0) \sqrt{\frac{2\pi}{-h''(x_0)^{-1}}} \int \varphi(\mu,\sigma^2) dx $$

(For our purposes, this is all we need to get cooking. However, we're so close to a key result of the Laplace approximation that I might as well continue)

In most problems involving the Laplace approximation, we're trying to integrate over the entire domain of the function, which for the normal pdf is always defined to be one. So the integral of \\(g(x_0)\\) over \\(\mathbb{R}\\) is approximately

$$g(x_0) \sqrt{\frac{2\pi}{-h''(x_0)^{-1}}}$$

Where \\(h(x_0) = \ln g(x_0)\\).

Got that? Good, because we're just getting started. 

## the hard part. 

To pull this off, we have to take everything we just did, and kick it up to higher dimensions. I do not know enough multivariable calculus to derive these results myself[^6], but I will draw parallels to the singular dimensional version.

Given some \\(N\\) dimensional function with most of it's mass concentrated around it's domain, Laplace says we can approximate the function as 

$$ g(\boldsymbol{x}) \approx g(\boldsymbol{x_0}) \exp[-\frac{1}{2} \cdot (\boldsymbol{x}-\boldsymbol{x_0})^\intercal \boldsymbol{S}^{-1}(\boldsymbol{x}-\boldsymbol{x_0})] $$

Where \\(\boldsymbol{x}_0\\), (\\(\boldsymbol{x}_0 \in \mathbb{R}^N\\)), is the location of the maximum of \\(g(\boldsymbol{x})\\), and \\(\boldsymbol{S}^{-1}\\), (\\(\boldsymbol{S}^{-1} \in \mathbb{R}^{N \times N}\\)), is \\(-\nabla_x\nabla_x\ln g(\boldsymbol{x})\\) at \\({\boldsymbol{x}=\boldsymbol{x}_0}\\) (the Hessian, AKA the multivariate second derivative, of \\(\ln g(\boldsymbol{x})\\) at \\(\boldsymbol{x}_0\\)).

How does this relate to the posterior probablity \\(\ln P(\theta \| \mathcal{D}_A)\\) again? Well this probablity is actually equal to 

$$\ln P(\theta | \mathcal{D}_A) = \ln P(\theta, \mathcal{D}_A) - \ln P(\mathcal{D}_A)$$

Remember the probablity of the data \\(\mathcal{D}_A\\) is fixed and *cannot* be affected by us. When doing optimization and trying to maximize the posterior (or equivalently, minimize the negative posterior) , we can safely ignore it.

Now onto maximizing the other part. We actually do not know what \\(P(\theta, \mathcal{D}_A)\\) is! This is where the Laplace's approximation comes in handy. Going back to the problem of continual learning, when we're training task \\(B\\) we already assume that we've got the best parameters for the task \\(A\\), meaning that our current parameters \\(\theta^*_A\\) are already maximized for the task at hand. So we have our \\(\boldsymbol{x}_0\\), but if you've been following along we need the function \\(P(\theta, \mathcal{D}_A)\\) to calculate \\(\boldsymbol{S}^{-1}\\), but we need \\(\boldsymbol{S}^{-1}\\) to approximate the function! 

## let's go fishing

This was by far the hardest part to understand (and I still don't fully understand it!) and the biggest reason that this blog post was so late, so take what I say here with a grain of salt. 

The value of \\(\boldsymbol{S}^{-1}\\) is actually known as the [observed fisher information](https://en.wikipedia.org/wiki/Observed_information), and is a sample version of the [Fisher information](https://en.wikipedia.org/wiki/Fisher_information). What these two measures tell us is how important each pair of parameters is for the data. Since we can't actually calculate the observed information, we calculate the Fisher information and use that as our observed information. 

The paper also only uses the diagonal of the Fisher information matrix. I have no idea why! This seems to assume that the parameters are independent of each other, but I see no reason why this would be true. Changes to one parameter absolutely might affect another, and the non-diagonal Fisher information matrix reflects this. But this blog post is already too damn late, so I can investigate more next week. 

The Fisher information matrix is defined as follows

$$\boldsymbol{F} = E[\nabla\mathcal{L}(\theta)^\intercal\nabla\mathcal{L}(\theta)]$$

Where \\(\mathcal{L}\\) is the loss function over dataset \\(\mathcal{D}_A\\)

While we can't calculate the *observed* information, the *Fisher* information is simple to calculate. So we use the Fisher information as a substitute   

## reeling it in

So now the problem boils down to maximizing \\(P(\theta | \mathcal{D}_A)\\) which we approximate as 

$$ P(\boldsymbol{\theta^*_A} | \mathcal{D}_A) \exp[-\frac{1}{2} \cdot (\boldsymbol{\theta}-\boldsymbol{\theta^*_A})^\intercal \boldsymbol{F}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})] $$

\\(P(\boldsymbol{\theta^*_A} \| \mathcal{D}_A)\\) doesn't actually matter, since it's a constant, and we typically take the log of things we want to maximize in ML to make computations easier. So our goal is to maximize 

$$\ln \exp[-\frac{1}{2} \cdot (\boldsymbol{\theta}-\boldsymbol{\theta^*_A})^\intercal \boldsymbol{F}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})]$$

$$-\frac{1}{2} \cdot (\boldsymbol{\theta}-\boldsymbol{\theta^*_A})^\intercal \boldsymbol{F}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})$$

But in ML, problems are usually framed in terms of minimization. So our goal is to *minimize* this function of \\(\boldsymbol{\theta}\\)

$$\frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})^\intercal \boldsymbol{F}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})$$

Then we can add it back to the original equation to get the EWC-Loss function.[^7] 

$$\mathcal{L}_B(\boldsymbol{\theta}) + \frac{\lambda}{2}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})^\intercal \boldsymbol{F}(\boldsymbol{\theta}-\boldsymbol{\theta^*_A})$$

Where \\(\lambda\\) is a constant we choose that says how much we should weight retaining old tasks over training on new ones. 

## reflections

I learned *a lot* from struggling with this paper, and I now know little I really know. But it's so crazy that some research advisors expect their grad students to read a paper per day, when it took me a week and a half to read *one*! Keep in mind that the EWC paper was just a citation in another paper which I was trying to finish in a day, I haven't even starting digging into that one!

Maybe grad students with an undergrad + early grad school education can read a paper per day, but as I am right now I struggle with getting past a paper per *week*. I definitely need to slow down, and add a book on statistics to my reading list!

# productivity check in

Last week I said...

>Weekly goals:
	1. Finish 2 Chapters of [Understanding Deep Learning](https://udlbook.github.io/udlbook/)
	2. Redesign UI for engineering project
	3. Don't fail tests lol.
	4. Write a blog post about continual learning
	5. Read two chapters of [TSCM](https://www.amazon.com/Scout-Mindset-Perils-Defensive-Thinking/dp/0735217556)

I actually did pretty okay this week! I finished a chapter and a half of UDL. I didn't start on the UI, but I did start on the Input system for my engineering project. I learned a ton from that too. I did manage to pass my tests this week! And, you're just finished reading(/skimming) the blog post about continual learning. I need to find more time to read in general, but I'm not too upset with how much I read right now.

I contribute this rise in productivity to A - more manageable goals, and B - trying my best to stick to the [pomodoro technique](https://todoist.com/productivity-methods/pomodoro-technique). I've been tracking my pomodoros (25 minute work sessions) and I average out to about 6 per day. That's roughly three hours of focused work per day, which is kind of sad. Yeah, I don't count time spent doing school busy work, but three hours still doesn't seem like enough. I want to work my way up to five hours someday soon.

I won't bother posting any new goals right now, since this blog is coming so late. There's two days left before 0x0B, I'll let you know how it goes. 




[^1]: [A paper defining all three sub-problems](https://arxiv.org/abs/1810.12488) 
[^2]: [(Kirkpatrick et al., 2017)](https://arxiv.org/abs/1612.00796)
[^3]: https://bookdown.org/rdpeng/advstatcomp/laplace-approximation.html
[^4]: Formally, the function \\(g(x)\\) must exist in \\(\mathfrak{L}^2\\), meaning that \\(\int_\mathbb{R} g(x)^2dx < \infty\\)   
[^5]: \\(\exp[x] = e^x\\)
[^6]: I draw everything from [Wikipedia](https://en.wikipedia.org/wiki/Laplace%27s_approximation)
[^7]: This isn't the *exact* function described in the paper, because I'm still not fully on board with the idea of a diagonal Fisher information matrix. However, if \\(\boldsymbol{F}\\) is diagonal the equation I have is identical to the one in the paper.