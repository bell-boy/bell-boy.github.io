---
layout: post
title: making pytorch from scratch in c++
usemathjax: true
featured: true
featured_order: 1
---

_tl;dr: I built a simple version of PyTorch from scratch in C++. It supports most standard tensor operations, automatic differentiation, and can run a 135 Million paramter LLM on my M2 MacBook Air. The code is open source and available on [GitHub](https://github.com/bell-boy/deeptiny/tree/main). I go into the details of how I built it, why I made certain design choices, and what I learned from the process._

I've wanted to create a project like this for a while now, and when we were told that we would have freedom to build anything computer architecture related for our final project, I knew it was time. 

The main motivation for doing something like this was noticing that the best engineers at big labs (Google, OAI, Anthropic) also tend to be the best researchers. So while building theoretical knowledge seemed straightforward, building engineering skill seemed less clear cut. I thought this project was the perfect opportunity for me to get my feet wet!


I'm going to assume that you have some familiarity with the notion of gradient descent, c++, and pytorch, but I'll try to keep the rest fairly accessible. My goal here is to let someone with a basic understanding of the above concepts understand how to build a simple version of PyTorch from scratch, and to give them the tools to do so if they want to.

The first half will focus on the design, and the second half will walk through an actual module.

# theory

At the core of any modern deep learning framework is the concept of _automatic differentiation_. 
This allows us to worry about defining our models and loss functions, and not about the tedious and error prone process of computing gradients by hand.
The most common method for automatic differentiation is reverse-mode differentiation, which is what most modern deep learning libraries use.
Reverse mode differentiation works by building a computational graph of the operations performed on tensors, and then traversing this graph in reverse to compute gradients.

For example, let's take the code

```python
x = [1, 2, 3]
y = [4, 5, 6]
z = <x, y>  # dot product
z.backward()  # compute gradients
```
Reverse mode starts by building a computational graph that looks like this:

![comp_graph_1]({{site.baseurl}}/assets/images/making-pytorch/comp_graph_1.png)

You'll note that we have two types of nodes: data nodes (the circles) and operation nodes (the squares). The data nodes represent the tensors, while the operation nodes represent the operations performed on those tensors. 

`z.backward()` promises that on its completion, the gradient of every _leaf_ data node in the graph with respect to `z` will be computed and stored in the `grad` attribute of the corresponding data node.

A leaf data node is a data node that has no operation nodes coming into it. In this case, `x` and `y` are leaf data nodes, while `z` is not (since it has an operation node coming into it).

The process of computing the gradients is roughly
  1. Define the gradient of the objective node (in this case, `z`) to be 1. This makes sense, since the gradient of `z` with respect to itself is 1.
  2. Traverse the graph in reverse, applying the chain rule to compute the gradients of the input nodes (in this case, `x` and `y`), discarding the gradients of any non-leaf nodes (in this case, `z`).

Let's try a more complex example:

```python
x = [1, 2, 3]
A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

y = A @ x  # matrix-vector multiplication
z = <y, x> # dot product
z.backward()  # compute gradients
```
The computational graph now looks like so:

![comp_graph_2]({{site.baseurl}}/assets/images/making-pytorch/comp_graph_2.png)

Now some of z's parents have parents of their own, so we have to apply the chain rule multiple times to get the gradients for `x` and `A`.

At the start of the backward pass, the gradient of each data node in the graph with respect to `z` is set to 0. 
As we walk backward through the graph, we _update_ the gradients of the parent nodes by applying the chain rule and adding the result to the existing gradient. This is important, since a node can have multiple children, and we need to sum the gradients from each child to get the correct gradient for the parent. 

Once a node has received gradients from all of its children, we can apply the chain rule to compute the gradients for its parents. This process continues until we reach the leaf nodes, at which point we're done.

In this example, a valid ordering of edges to traverse would be
1. `z` to `y`
2. `z` to `x`
3. `y` to `A`
4. `y` to `x`

Note that 2 and 3 could be swapped in this example, but 1 has to come before both of them, since we need to compute the gradient of `y` with respect to `z` before we can compute the gradient of `A` and `x` with respect to `z`.

# design

I was heavily inspired by the design of pytorch, specifically, [this blog](https://blog.ezyang.com/2019/05/pytorch-internals/) on pytorch's internals by Edward Yang.
I didn't want to just copy pytorch, so I went through the process of designing my own framework from scratch after reading this, but I did borrow some ideas from pytorch's design.

The core of the library is the `Tensor` class, which is small enough that I can paste it below:

```cpp
class Tensor {
 protected:
  std::shared_ptr<TensorImpl> tensor_impl_;
  std::shared_ptr<AutogradMeta> autograd_meta_;

  Tensor(std::shared_ptr<TensorImpl> tensor_impl,
         std::shared_ptr<AutogradMeta> autograd_meta);

 public:
  // Create a a contiguous tensor with uninitialized data
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

The key takeaway here is that the data and the metadata are completely separate, and interact in only a few key places. We'll take advantage of this separation later.

## tensor impl
The `TensorImpl` class is a bit more complex, but the main idea is that it's an interface that tells us how to interact with the data of the tensor.

The hardware has no notion of "vectors" or "matrices", it only knows about lists of bytes. So the `TensorImpl` class is responsible for two things
  1. Managing a chunk of memory that holds our bytes
  2. Telling us how to interpret those bytes as a mathematical object.

To handle the first responsibility, each `TensorImpl` owns an instance of a `Storage` object. The `Storage` abstract base class is responsible for defining an API for a managed chunk of memory on the device (CPU, GPU, etc) we're computing on. 

`Storage` also handles _versioning_, a fancy term for keeping track of modifications to its underlying data. In short, every time the underlying data is modified, we increment a version number. Anyone who has access to the `TensorImpl` object can check the version number to see if the data has been modified since they last checked. 
The utility of versioning is hard to see now, but will really shine once we start talking about the backward pass.

To understand how `TensorImpl` handles the second responsibility, we have to talk about strided representations.

### strided representations
The crux of the second responsibility is this -- how do I assign each entry of a multi-dimensional array to a location in memory?

Strided representations solve this problem by specifying one scalar and two lists of numbers: the _offset_, _shape_, and _stride_.

As a running example, let's say we want to represent the `(3, 2)` Tensor `x = [[1,2,3],[4,5,6]]`. 
Now there are a total of six indices we can access -- from `(0, 0)` to `(2, 1)`.

The _shape_ is a concise way to specify this set of valid indexes (or, as we'll see later, a set of valid _slices_). Since the shape is (3, 2), in the first slot we can place any number from 0 to 2 inclusive, while in the second slot we can only place any number from 0 to 1.

The combination of _stride_ + _offset_ is how we map these indices to memory locations. In a majority of cases, `x` has stride `(2, 1)` and an offset of `0`. The best way to show you what this means is to walk through an example

Say we want to find `x[2, 1]`. We'd take each index value and multiply it by its corresponding stride, sum it up, and add the offset. 

In this case we'd get `location = 2 * 2 + 1 * 1 + 0 = 5`. That'd tell us where to look within the memory buffer managed by the `Storage` object. 

Below is a concise visual mapping from tensor to memory location

![]({{site.baseurl}}/assets/images/making-pytorch/mem_map.png)

Strides are more flexible than this simple example however, they enable _slices_, -- creating a smaller tensor from a larger one

Say we want to treat the second column of `x` as a vector `v=[2, 5]`. Instead of creating a new `Storage` object, `v` and `x` can share one.

What's `v`'s shape, stride and offset? It's clear that the shape is `(2)`, but the stride is now `(2)` (since our elements are now 2 apart from each other in the buffer). Since the first element is now `2` our offset is also `2`

![]({{site.baseurl}}/assets/images/making-pytorch/mem_map2.png)

Strides are an incredible double edged sword. 
They allow us to cheaply perform a huge variety of operations (slicing, transposing, etc) without having to copy data, but they also make it much harder to write efficient kernels for our operations, since we have to account for the fact that neighboring elements in the tensor might not be neighboring in memory.

So in summary, the `TensorImpl` class is little more than just a `Storage` object + shape + stride + offset. That's all we need to go off to the races.

## autograd meta
Let's go back to our complicated example from earlier. 

![comp_graph_2]({{site.baseurl}}/assets/images/making-pytorch/comp_graph_2.png)

Just like there are two types of nodes, there are really two types of autograd metadata: `AutogradMeta` (corresponding to the data nodes) and `Function` (corresponding to the operation nodes).

`AutogradMeta`'s most important member is its `std::shared_ptr<Tensor> grad_`.
This is where, on completion of the backward pass, gradients will live. `AutogradMeta` has an `updateGrad` method used to add to the gradient tensor.

Making our grad objects `Tensors` themselves cleanly gives us the ability to do _higher order differentiation_ if we so desire.

`Function` is an abstract base class that each operation implements. It has two key members:
  1. `std::vector<std::strong_ptr<AutogradMeta>> parents_`: The upstream inputs used to create this node.
  2. `void operator()(const Tensor& grad)`: The backward function used to compute parent gradients. 
  Given some grad tensor for the output of the operation, this function computes the gradients for each parent and updates their `grad_` tensors via the `updateGrad` method.


In addition each gradient has a `Context` object, which can store `Tensors` for later use.
The primary job of using the `Context` object over just storing the parents as members of the `Function` is automatic _version checking_. 

An illustrative example of why this is important is the following:

```python
y = 2
x = 3
z = x * y
y += 1
z.backward()
```

In this example, the gradient of `z` with respect to `x` is the value of `y` at the time of `z`'s creation -- 3, 
but if we naively stored `y` as a member of the `Function` that created `z`, we'd compute the gradient as 4 instead, since `y` was modified after `z` was created.

`Context` objects solve this problem by allowing us to store the value of `y` at the time of `z`'s creation, and then checking that the version of `y` at the time of the backward pass is the same as the version of `y` at the time of `z`'s creation. If they're different, we know that `y` was modified after `z` was created, and we can throw an error instead of being silently wrong.

The last key player is the `Engine` object. It's responsible for traversing the graph and calling backward on tensors that have received all their downstream gradients and are ready to update their parents. 

Alright, let's walk through this example.
1. `z`'s `AutogradMeta` calls the `operator()(const Tensor &a)` on its `InnerProductBackward` `Function` instance. This updates the gradients for both `x` and `y`.
2. `y` is now ready to call its backward function -- `MatrixVectorBackward` and update the gradients for both `A` and `x`.
3. Both `A` and `x` are completely done, but they don't have `Function`s attached to their `AutogradMeta`'s -- they're leaf nodes. We're now done!

# a concrete example: batched matrix matrix multiplication
Now that we've gone through the theory and design, let's walk through a concrete example of how to implement a simple function -- batched matrix multiplication.
The bulk of function complexity is mostly shared between functions, so once you understand this example, the implementation for other functions mostly falls into place.

Batched matmul takes two tensors of rank at least three. 
To keep things simple, we'll assume the input is exactly rank three. 
For instance, let tensor `A` have shape $(B, n, k)$, while tensor `B` has shape $(B, k, m)$.
The dimension sizes $B$ and $k$ must match but $m$ and $n$ are free to vary.
Batched matrix multiplication will produce a tensor of size $(B, n, m)$ by multiplying each of the $B$ pairs of $(n, k)$ and $(k, m)$ matrices together. 
We may optionally transpose the `A` or `B` tensors before multiplying them.

We'll see many functions on our journey, but let's start from the actual function users will see. 
Here's the full code of the function.
Pay attention to three things: input shape checks, batch broadcasting, and where backward metadata gets attached to the output tensor.

```c++
Tensor BatchedMatMul(const Tensor& a, const Tensor& b, bool transpose_a,
                     bool transpose_b) {
  utils::CompatabilityCheck({a, b});
  if (a.shape().size() < 3 || b.shape().size() < 3) {
    throw std::runtime_error("BatchedMatMul requires rank >= 3 inputs");
  }

  const auto& a_shape = a.shape();
  const auto& b_shape = b.shape();
  const uint64_t a_rows = a_shape[a_shape.size() - 2];
  const uint64_t a_cols = a_shape[a_shape.size() - 1];
  const uint64_t b_rows = b_shape[b_shape.size() - 2];
  const uint64_t b_cols = b_shape[b_shape.size() - 1];

  const uint64_t lhs_cols = transpose_a ? a_rows : a_cols;
  const uint64_t rhs_rows = transpose_b ? b_cols : b_rows;
  if (lhs_cols != rhs_rows) {
    throw std::runtime_error(
        "BatchedMatMul requires matching inner dimensions");
  }

  Shape a_batch(a_shape.begin(), a_shape.end() - 2);
  Shape b_batch(b_shape.begin(), b_shape.end() - 2);
  auto out_batch = BroadcastLeadingShape(a_batch, b_batch);
  if (!out_batch) {
    std::stringstream err;
    err << "Cannot broadcast batch dimensions from shapes "
        << FormatShape(a_shape) << " and " << FormatShape(b_shape) << ".";
    throw std::runtime_error(err.str());
  }

  Shape a_target = *out_batch;
  a_target.push_back(a_rows);
  a_target.push_back(a_cols);
  Shape b_target = *out_batch;
  b_target.push_back(b_rows);
  b_target.push_back(b_cols);

  auto a_broadcast = utils::BroadcastToShape(a, a_target);
  auto b_broadcast = utils::BroadcastToShape(b, b_target);
  if (!a_broadcast || !b_broadcast) {
    std::stringstream err;
    err << "Failed to broadcast BatchedMatMul inputs to shapes "
        << FormatShape(a_target) << " and " << FormatShape(b_target);
    throw std::runtime_error(err.str());
  }

  auto out_impl = dispatch::batched_matmul::OutOfPlace(*a_broadcast,
                                                       *b_broadcast, transpose_a,
                                                       transpose_b);

  auto backward = std::make_shared<BatchedMatMulBackward>(
      *a_broadcast, *b_broadcast, transpose_a, transpose_b);
  auto out_meta = std::make_shared<AutogradMeta>(backward);
  return utils::TensorAccessor::MakeTensor(out_impl, out_meta);
}
```

I promise this is simpler than it looks!

Firstly, we can skim over the code until we hit that `dispatch::batched_matmul::OutOfPlace` call.
All of that code is to ensure that the tensors `A` and `B` can actually be multiplied together, and making sure we inform the user if we can't. 

The `dispatch::batched_matmul::OutOfPlace` call creates the `TensorImpl` of the output. If we didn't care about gradients, we could stop here!   

We then construct a `BatchedMatMulBackward` for the tensor and construct an `AutogradMeta` from it.

Finally we bind these both together to return the tensor we care about! 

You'll note that aside from the checks at the start, the `BatchedMatMul` function never actually implemented the batched matmul.
It used other functions to implement the two halves, and then glued those halves together!
This is a consistent theme throughout deeptiny.

## dispatching
In theory, Tensors can exist on various devices, and may be one of various data types. 
Each of these device, data type pairs may need a unique implementation of the method!

In deeptiny, figuring out how to route an operation to a device is the responsibility of the dispatch library! 
You can think of it as an internal collection of switch statements that route to a device dispatcher, which then routes to a data type dispatcher. 
Thankfully, since deeptiny only has one pair, dispatching is very simple. 
The dispatch code is below.
Pay attention to how this section builds the output shape once, then routes by device to the concrete kernel.

```c++
namespace deeptiny::dispatch::batched_matmul {

std::shared_ptr<TensorImpl> OutOfPlace(const std::shared_ptr<TensorImpl>& a,
                                       const std::shared_ptr<TensorImpl>& b,
                                       bool transpose_a, bool transpose_b) {
  const auto& a_shape = a->shape();
  const auto& b_shape = b->shape();
  const uint64_t a_rows = a_shape[a_shape.size() - 2];
  const uint64_t a_cols = a_shape[a_shape.size() - 1];
  const uint64_t b_rows = b_shape[b_shape.size() - 2];
  const uint64_t b_cols = b_shape[b_shape.size() - 1];
  const uint64_t lhs_rows = transpose_a ? a_cols : a_rows;
  const uint64_t rhs_cols = transpose_b ? b_rows : b_cols;

  Shape out_shape(a_shape.begin(), a_shape.end() - 2);
  out_shape.push_back(lhs_rows);
  out_shape.push_back(rhs_cols);

  auto out = std::make_shared<TensorImpl>(out_shape, a->dtype(), a->device());
  switch (out->device()) {
    case Device::CPU:
      cpu::BatchedMatMul(a, b, out, transpose_a, transpose_b);
      return out;
    default: {
      std::stringstream err;
      err << "Operation does not support " << out->device().ToString();
      throw std::runtime_error(err.str());
    }
  }
}
```

Again, simpler than it looks!

Focus your attention on the `cpu::BatchedMatMul` call, everything before that is just a setup for the output `TensorImpl` we pass in. 
This is the CPU specific "dispatcher".

The "dispatcher" itself is currently acting as a kernel for float32 matmuls, and is pretty confusing to understand at first glance due to the strange consequences of strided representations. 
We won't go deeper than this.

`out_impl` now holds a `TensorImpl` representing the result of the batched matmul.

## the backward pass

Most of the heavy lifting for wiring the backward is done in the `BatchedMatmulBackward` class.
Let's take a look!
Pay attention to what gets saved in `context()` during the forward pass, and how each parent's gradient is computed in `operator()`.

```c++
class BatchedMatMulBackward : public Function {
 public:
  enum struct ContextObjects : uint64_t {
    SAVED_A = 0,
    SAVED_B = 1,
  };

  BatchedMatMulBackward(const Tensor& saved_a, const Tensor& saved_b,
                        bool transpose_a, bool transpose_b)
      : Function({utils::TensorAccessor::GetAutogradMeta(saved_a),
                  utils::TensorAccessor::GetAutogradMeta(saved_b)}),
        transpose_a_(transpose_a),
        transpose_b_(transpose_b) {
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_A), saved_a);
    context().Set(static_cast<uint64_t>(ContextObjects::SAVED_B), saved_b);
  }

  void operator()(const Tensor& grad) override {
    Tensor saved_a =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_A));
    Tensor saved_b =
        context().Get(static_cast<uint64_t>(ContextObjects::SAVED_B));

    const auto& parents = getParents();
    assert(parents.size() == 2 &&
           "BatchedMatMulBackward must have exactly 2 parents");
    assert(parents[0] && "BatchedMatMulBackward first parent must not be null");
    assert(parents[1] &&
           "BatchedMatMulBackward second parent must not be null");

    if (!transpose_a_) {
      parents[0]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(grad, saved_b, false,
                                               !transpose_b_));
    } else {
      parents[0]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(saved_b, grad, transpose_b_,
                                               true));
    }

    if (!transpose_b_) {
      parents[1]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(saved_a, grad, !transpose_a_,
                                               false));
    } else {
      parents[1]->updateGrad(
          dispatch::batched_matmul::OutOfPlace(grad, saved_a, true,
                                               transpose_a_));
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
};
```

There's two methods here: the constructor and the `operator()` function. 
The constructor simply stores the tensors we'll need for the backward pass, and registers them as parents.
The `operator()` is more subtle.

It uses the stored tensors from the forward pass to update the gradients in the backward pass. 
In this case, the backward pass of a batched matmul involves another batched matmul. 

Despite what I said earlier about higher order differentiation, I found it too much trouble and decided to only have first order support.

Since we don't care about registering higher order gradients, we can call the dispatch functions directly here.
`TensorImpl`s implicitly cast to Tensors, so we can pass it to the `updateGrad` method without any problems.

That's it! We've now finished wiring up the `BatchedMatMul` for both the forward and backward passes. This will just work irrespective of any future functions we add!

This is the process of adding functions. Mostly a lot of boring wiring crap that's nonetheless important for making sure everything works together.

# reflections

I've left out a lot of the process of building deeptiny here.
Notably everything about how we actually traverse the graph and call the backward functions has been left out.
But also many other smaller things, like optimizations I did or tricky functions I had to implement.
That's because it's all pretty boring.

Most of deeptiny is the same boilerplate over and over again. Wiring functions from the kernels to the backward pass. After the second function I implemented, I was curious if I could do this any faster.

After getting a testing harness in and enough code in the codebase so that my style was firmly established, I started experimenting with coding agents.
The results were dramatic.
Even reading every line of code by hand, I was averaging about 4+ PRs per day. 
It was addictive!
The bottleneck became how much code I could review per day.
There was almost never a moment I didn't have two codex instances spinning. 

In a strange way though, coding agents stretched me much more than coding by hand did.
By shifting the easier parts to agents, all the tasks left for me to do were hard!
It felt like I never got a break, by the time I had parsed one PR, the agents had delivered another!

It wasn't all sunshine and rainbows: sometimes the agents wrote horribly bloated code, it wasn't uncommon for a review session to end with a PR 200 to 300 lines smaller than it came in.
Despite this, I felt for the first time the dramatic change going on in SWE. 
It's exhilarating!

Deeptiny is missing a few features, notably GPU support, but I very much enjoyed the process of designing and bringing this to life.

Thank you for reading!
