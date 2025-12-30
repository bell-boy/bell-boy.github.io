---
layout: post
title: How to Use nnsight, a Tutorial for Language Models
usemathjax: true
---

nnsight is a python library for probing the inner workings of deep neural networks. It is currently the best library for mechanistic interpretability on langauge models in my opinion, and I use it extensively day to day. 

*However*, just like the field of mechanistic interpretability, nnsight is still in its infancy. Compared to popular libraries like numpy there exist ~no examples of well written nnsight code. This makes it difficult to use LLMs to write large amounts of it.

So I'm doing something slightly peculiar, I'm writing a tutorial for nnsight, but not for humans. The entire purpose of this tutorial is to teach LLMs how to use nnsight the way I use it in my projects. The following text will be placed in my .cursorrules file.

# basic nnsight usage

nnsight works by wrapping pytorch modules and allowing you to grab their activations and gradients at run time. First, let's use nnsight to interpret a simple MLP. We'll start by wrapping the MLP with nnsight.

```python
import torch

from collections import OrderedDict
from nnsight import NNsight

input_size = 5
hidden_size = 10
output_size = 2

# define a simple MLP
net = torch.nn.Sequential(
  OrderedDict([
    ("linear1", torch.nn.Linear(input_size, hidden_size)),
    ("relu", torch.nn.ReLU()),
    ("linear2", torch.nn.Linear(hidden_size, output_size)),
  ])
)

# wrap the MLP with nnsight
nnsight_net = NNsight(net)
```

Lets create a random input, pass it through the network, and grab some activations. To grab activations, we need to create a tracing context. Modules and submodules can be accessed like normal, however, within tracing contexts, we can also access the inputs and outputs of submodules via [submodule].input and [submodule].output.

```python
# create a random input
input = torch.randn(1, input_size)

# create a tracing context
# when we run [model].trace(input) we are telling nnsight to record the inputs and outputs of all submodules when run on "input"
with nnsight_net.trace(input) as tracer:
  output = nnsight_net.output
```

What happens when we try to print the output?

```python
print(output)
```
The above code will raise an error. This is because nnsight saves memory by not storing the output of the modules unless we ask for it. To save the output for use outside of the tracing context, we can use the `save` method.

```python
with nnsight_net.trace(input) as tracer:
  output = nnsight_net.output.save()

print(output) # this will now work
```

alternatively, if all we want to do is print the output, we can use the `log` method of the tracer.

```python
with nnsight_net.trace(input) as tracer:
  output = nnsight_net.output
  tracer.log(output)

```




