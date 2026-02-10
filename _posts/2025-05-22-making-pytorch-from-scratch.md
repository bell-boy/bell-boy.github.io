---
layout: post
title: making pytorch from scratch in c++
usemathjax: true
---

_tl;dr: I built a simple version of PyTorch from scratch in C++. It supports most standard tensor operations, automatic differentiation, and Transformer infrence on CPU. The code is open source and available on [GitHub](https://github.com/bell-boy/deeptiny/tree/main). I go into the details of how I built it, why I made certain design choices, and what I learned from the process. [blog post in progress]_

I've wanted to create a project like this for a while now, and when we were told that we would have freedom to build anything computer architecture related for our final project, I knew it was time. 

The main motivation for doing something like this was noticing that the best engineers at big labs (Google, OAI, Anthropic) also tend to be the best researchers. So while building theoretical knowledge seemed straightforward, building engineering skill seemed less clear cut. I thought this project was the perfect opportunity for me to get my feet wet!


I'm going to assume that you have some familiarty with the notion of gradient descent, c++, and pytorch, but I'll try to keep the rest fairly accessible.

The first half will focus on the design, and the second half on low level optimizations to make things go fast.

# theory

At the core of any modern deep learning framework is the concept of automatic differentiation. This allows us to compute gradients efficiently, which is essential for training neural networks. 
The most common method for automatic differentiation is reverse-mode differentiation, which is what PyTorch uses.
Reverse mode differentiation works by building a computational graph of the operations performed on tensors, and then traversing this graph in reverse to compute gradients.

For example, let's take the code

```python
x = [1, 2, 3]
y = [4, 5, 6]
z = <x, y>  # dot product
z.backward()  # compute gradients
```
Reverse mode stards by building a computational graph that looks like this:

![comp_graph_1]({{site.baseurl}}/assets/images/making-pytorch/comp_graph_1.png)

You'll note that we have two types of nodes: data nodes (the circles) and operation nodes (the squares). The data nodes represent the tensors, while the operation nodes represent the operations performed on those tensors. 

When we call `z.backward()`, we do two things
  1. Define the gradient of the output node (in this case, `z`) to be 1. This is because we want to compute the gradient of `z` with respect to itself, which is always 1.
  2. Traverse the graph in reverse, applying the chain rule to compute the gradients of the input nodes (in this case, `x` and `y`).

Let's try a more complex example:

```python
x = [1, 2, 3]
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

y = A @ x  # matrix-vector multiplication
z = <y, x> # dot product
z.backward()  # compute gradients
```
The compuational graph now looks like so:

![comp_graph_2]({{site.baseurl}}/assets/images/making-pytorch/comp_graph_2.png)

You'll notice that x now has two edges coming out of it, one to the matrix-vector multiplication node and one to the dot product node.

Reverse mode differentiation handles this nicely, the true gradient of x is the sum of the gradients from both paths. So when we traverse the graph in reverse, we compute the gradient from the dot product node and add it to the gradient from the matrix-vector multiplication node.

# design

I was heavily inspired by the design of PyTorch, specifically, [this blog](https://blog.ezyang.com/2019/05/pytorch-internals/) on pytorch's internals by Edward Yang.
I didn't want to just copy PyTorch, so I went through the process of designing my own framework from scratch after reading this, but I did borrow some ideas from PyTorch's design.

The core of the library is the `Tensor` class, which is small enough that I can paste it below:

```cpp
class Tensor {
 protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
  std::shared_ptr<AutogradMeta> autograd_meta_;

  Tensor(std::shared_ptr<TensorImpl> tensor_impl,
         std::shared_ptr<AutogradMeta> autograd_meta);

 public:
  // Create a a contingous tensor with uninitialized data
  Tensor(Shape shape, DType dtype, Device device, bool requires_grad);

  View operator()(std::vector<Slice> slices);
  const View operator()(std::vector<Slice> slices) const;
  Shape shape() const;
  DType dtype() const;
  Device device() const;
  Tensor Clone() const;
  Tensor Squeeze(const std::vector<uint64_t>& dims);
  bool requires_grad() const;
  std::optional<Tensor> grad() const;
  void Backward(bool keep_graph = false);

  /**
   * Creates a tensor on the CPU with the expectation that the bytes are laid
   * out in row-major order
   */
  static Tensor FromBuffer(std::span<const std::byte> bytes, Shape shape,
                           DType dtype = DType::Float32,
                           Device device = Device::CPU,
                           bool requires_grad = false);
  void operator+=(const Tensor& other);
  void operator-=(const Tensor& other);
  void operator*=(const Tensor& other);
  void operator/=(const Tensor& other);

  friend struct utils::TensorAccessor;
};
```

We'll explain all of this in due time, but for now, let's focus on the first two members: `tensor_impl_` and `autograd_meta_`.

The `tensor_impl_` is a pointer to the actual data of the tensor, as well as some metadata about the tensor (shape, dtype, device).

In contrast the `autograd_meta_` is a pointer to the metadata that we need for automatic differentiation. This includes things like the gradient of the tensor, the operation that created the tensor, and the tensors that were used to create the tensor.

The key takeaway here is that the data and the metadata are completely separate, and interact in only a few key places. We'll take advantage of this seperation later.

## tensor impl
The `TensorImpl` class is a bit more complex, but the main idea is that it's an interface that tell us how to interact with the data of the tensor.

The hardware has no notion of "vectors" or "matrices", it only knows about lists of bytes. So the `TensorImpl` class is responsible for two things
  1. Managing a chunk of memory that holds our bytes
  2. Telling us how to interpret those bytes as a mathematical object.

To handle the first responsibility, each `TensorImpl` owns an instance of a `Storage` object. The `Storage` abstract base class is responsible for defining an API for a managed chunk of memory on the device (CPU, GPU, etc) we're computing on. 

It also handles _versioning_, a fancy term for keeping track of modifications to it's underlying data. In short, everytime the underlying data is modified, we increment a version number. 
The utility of versioning is hard to see now, but will really shine once we start talking about the backward pass.

To understand how `TensorImpl` handles the second responsiblity, we have to talk about strided representations.

### strided represntations
The crux of the second responsiblity is this -- how do I assign each entry of a multi-dimensional array to a location in memory?

Strided representations solve this problem by specifiying one scalar and two lists of numbers: the _offset_, _shape_, and _stride_.

As a running example, let's say we want to represent the `(3, 2)` Tensor `x = [[1,2,3],[4,5,6]]`. 
Now there are a total of six indices we can access -- from `(0, 0)` to `(2, 1)`.

The _shape_ is a concise way to specify this set of valid indexes (or, as we'll see later, a set of valid _slices_). Since the shape is (3, 2), in the first slot we can place any number from 0 to 2 inclusive, while in the second slot we can only place any number from 0 to 1.

The combination of _stride_ + _offset_ is how we map these indices to memory locations. In a majority of cases, `x` has stride `(2, 1)` and an offset of `0`. The best way to show you what this means is to walk through an example

Say we want to find `x[2, 1]`. We'd take each index value and multiply it by it's corresponding stride, sum it up, and add the offset. 

In this case we'd get `location = 2 * 2 + 1 * 1 + 0 = 5`. That'd tell us where to look within the memory buffer managed by the `Storage` object. 

Below is a concise visual mapping from tensor to memory location

![]({{site.baseurl}}/assets/images/making-pytorch/mem_map.png)

Strides are more flexible than this simple example however, they enable _slices_, -- creating a smaller tensor from a larger once

Say we want to treat the second column of `x` as a vector `v=[2, 5]`. Instead of creating a new `Storage` object, `v` and `x` can share one.

What's `v`'s shape, stride and offset? It's clear that the shape is `(2)`, but the stride is now `(2)` (since our elements are now 2 apart from each other in the buffer). Since the first element is now `2` our offset is also `2`

![]({{site.baseurl}}/assets/images/making-pytorch/mem_map2.png)

Strides are incredibly useful, they prevent massive amounts of wasteful copying throughout our programs.
At the same time, we'll soon see that strides are also a _massive_ headache when it comes to the backward pass.

So in summary, the `TensorImpl` class is little more than just a `Storage` object + shape + stride + offset. That's all we need to go off to the races.

## autograd meta
Let's go back to our complicated example from earlier. 

![comp_graph_2]({{site.baseurl}}/assets/images/making-pytorch/comp_graph_2.png)

Just like there are two types of nodes, there are really two types of autograd metadata: `AutogradMeta` and `Function`

`AutogradMeta`'s most important member is it's `std::shared_ptr<Tensor> grad_`.
This is where, on completion of the backward pass, gradients will live. `AutogradMeta` has an `updateGrad` method used to add to the gradient tensor.

Making our grad objects `Tensors` themselves cleanly gives us the ability to do _higher order differientiation_ if we so desire.

`Function` is an abstract base class that each operation implements, it's a recipe to get from upstream gradients to downstream gradients. It has two key properties:
  1. `std::vector<std::strong_ptr<AutogradMeta>> parents_`: The upstream inputs used to create this node.
  2. `void operator()(const Tensor& grad)`: The backward function used to compute parent gradents. It calls `updateGrad` on each of it's parent
  

In addition each gradient has a `Context` object, which can store `Tensors` for later use.
This object automatically checks versioning mentioned earlier, so even if the tensor is modified after it's saved, gradients won't be incorrect silently.

The last key player is the `Engine` object. It's responsible for traversing the graph and calling backward on tensors that have recieved all their downstream gradients and are ready to update their parents. 

Alright, let's walk through this example.
1. `z`'s `AutogradMeta` calls the `operator()(const Tensor &a)` with on it's `InnerProductBackward` `Function` instance. This updates the gradients for both `x` and `y`.
2. `y` is now ready to call it's backward function -- `MatrixVectorBackward` and update the gradients for both `A` and `x`.
3. Both `A` and `x` are completely done, but they don't have `Function`s attached to their `AutogradMeta`'s -- in other words, they're _leaf nodes_. We're now done!

# optimizing

While the above is a good primer to how the library is laid out, I've intentionally ommited a lot of boring details about how code is structured internally & externally (kernel dispatch, public apis, utilites, testing, etc.). 
I think a better way to get these ideas is to walk through the process of optimizng various parts of it for faster inference.

I'll be implmenting inference and LORA finetuning on hugging-face's [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).

All of the code described here can be found in the `demo/transfomer-demo` folder within the git repo.

## inference
At a high level, inference is just tokenization -> embedding -> model -> sample -> de-tokenization.
My library is primarly focused on the middle three steps, so I'll use the excellent [mlc-ai/tokenizers-cpp](https://github.com/mlc-ai/tokenizers-cpp#) library to handle the outer steps for me.

To start, I very quickly wrote a refrence implementation with none of the common optimizations of modern models. That means no KV-Cache, no fused ops, etc.
None of the fancy wheels of modern models except for 1. 
Since SmolLM2-135M-Instruct is a Llama style model with Grouped Query Attention (GQA) (an architectural attention optimization) I was forced to implement this from the get go.

Now that that's done, let's go through the optimizations I made one by one.

### decode

Modern transformer inference can be broken up into two phases:
1. Prefill, where we compute key and value vectors for each token in the sequence (For \\(n\\) tokens takes \\(O(n^2\\) time)
2. Decode, where we take the cached key and value vectors for each token and use them compute the attention pattern for the latest token in \\(O(n)\\) time.

Currently without a KV cache I'm stuck constantly doing prefill for every token -- tanking my performance.

Surprisingly though, most of our inference costs were coming not from attention, but instead from the _Reduce_ operation used in multiple places.
