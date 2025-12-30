---
layout: post
title: making pytorch from scratch
usemathjax: true
---

I've wanted to create a project like this for a while now, and when we were told that we would have freedom to build anything computer architecture related for our final project, I knew it was time. 

The main motivation for doing something like this was noticing that the best engineers at big labs (Google, OAI, Anthropic) also tend to be the best researchers. So while building theoretical knowledge seemed straightforward, building engineering skill seemed less clear cut. I thought this project was the perfect opportunity for me to get my feet wet!

# Design

## Anatomy of a Tensor

In gooch, a tensor has two main parts:

1. Its representation -- i.e. how it's laid out in memory
2. Its gradient metadata -- information used during the gradient calculation.

### Representation

At their core, tensors are lists of numbers, organized in a multi-dimensional way. So natually, at the heart of the tensor sits the data buffer. 

*figure for the data buffer*

This one dimesional C-style list is created at the tensor's birth, alongside a pointer to another pointer for the gradients of this tensor [^1].

Naturally, there are more than just 1-dimensional vectors out there, and while yes, all tensors can be considered isomorphic to some vector, I'd like to enjoy programming.

To acheive this, we give every tensor three pieces of metadata, shape, stide, and offset.

*figure for shape and stride and offset*

The shape defines *how* you can index a tensor. It's pretty straight forward. A 3x2 matrix has shape \\( (3, 2) \\), and the maximum index it'll accept on each slot is \\( (2, 1) \\). In general, a tensor with shape \\( (N_1, \ldots, N_r) \\) is called a rank r tensor, and in index \\( i \in [1, r] \\) it accepts indcies in the range of \\( [0, N_i) \\).

The strides and offset define what data you get given a list of indices. For example, A standard \\( (3, 2) \\) tensor would have strides \\( (2, 1) \\) and offset \\( 0 \\), and to access the data at index \\( (2, 1) \\), I would find ``data[2 * 2 + 1 * 1 + 0]``. In general, given a tensor with stride \\(  (S_1, \ldots, S_r) \\) and offset \\( O \\), the index \\( (I_1, \ldots, I_r) \\) points to location \\[ O + \sum_{i = 1}^r S_iI_i \\] in the data buffer. 

Strides are very flexible, and let us do a variety of cool things. 

A core tensor operation is *partial indexing*. If you fully index a tensor, you get back a scalar. But if you *partially* index a tensor, what should be returned is another (lower rank) tensor. 

For example, lets say I have a batch of 28x28 images. I can store this in a rank 3 tensor \\( T \\) of size \\( (B, 28, 28) \\). Assuming this tensor is contiguous (I'll define this later), it'll have strides \\( (28 * 28, 28, 1) \\). What if I want the third image of the batch? Well I can simply index it via ``T(2)``. 

This returns a *view* \\( V \\) of shape \\( (28, 28) \\) with strides \\( (28, 1) \\). A key thing to note is that \\( V \\) and \\( T \\) point to the *exact same data*, they just tell you how to access it differently. And since ``V(0, 0)`` is not ``T(0, 0, 0)`` (which points to ``data[0]``) but ``T(2, 0, 0)`` (which points to ``data[28 * 28 * 2]``) the offset for \\( V \\) is \\( 28 * 28 * 2 \\). 

Now what if I wanted not just one image, but every *other* image. This is where *slices* come in. 

A slice is essentially a struct that contains a defines a range of tensors to index.

*screenshot of slice class*

I can get every other image by writing ``T(gooch::Slice(0, -1, 2))``[^2][^3]. This will create a *non-contiguous* tensor -- a tensor whose strides don't follow the dense access pattern \\( (N_2\cdots N_r, N_3\cdots N_r, \ldots, N_r, 1 )  \\) -- with shape \\( (\frac{B}{2}, 28, 28) \\) and strides \\( (28 * 28 * 2, 28, 1) \\).

We can also get the first column by typing ``T(gooch::Slice::all(), 0, gooch::Slice::all())`` (``gooch::Slice::all()`` is short hand for ``gooch::Slice(0, -1, 1)``). 

As you can see strides make partial indexing very powerful (we can even reverse an index[^4]), but they are a very *sharp* double edged sword. In exchange for this incredible freedom to manipulate tensors, we basically loose all gaureentees about data locality.

To illustrate this point, lets look at another operation strides make easy, transposition. Say I want to take my image tensor \\( T \\), and completely reverse the order of the indices. So my tensor will now have shape \\( (28, 28, B) \\). To acheive this, we just update the strides in the same way, the new strides are \\( (1, 28, 28 * 28) \\). This is the worst possible outcome. Iterating over that last index first will undoubtly cause more cache misses than otherwise nesscary. 

With only one tensor the problem is trival to solve -- always iterate over the index with the smallest stride first. But what happens when we want to elementwise multiply two tensors? What if the indices of their smallest strides don't line up? We'll have to figure out how to deal with this problem later.

### Gradient Metadata

This is where I made my first blunder. The way pytorch was organized pre-2017 (fact check needed) was in two parts ATen (short for "A tensor library") handled all tensor operations, while pytorch wrapped ATen and handled gradients. 

At the time of reading, this made absolutley no sense to me. Why should a the data tensor be seprerate from the gradient tensor? They're deeply interconnected objects.

Alongside gradient buffer the pointer to a pointer mentioned earlier, each tensor created via some operation also has a ``grad_fn_``. This function takes in another tensor consisting of the gradient of the current tensor.

Originally when called on some tensor T, ``grad_fn_`` adds the contents of the passed tensor to T's gradient buffer, then calculates the gradients of the tensors involved in T's creation and calls the ``grad_fn_`` of those tensors.

This is very very bad. To see why, here's an example of a network that takes \\(  N \\) operations to compute it's output, but \\(  2^{N} \\) operations to compute it's gradients.


[^1]: Why a pointer to a pointer? I don't want to initalize the grad buffer immediately at birth, instead, I want to delay creating this buffer until it's really needed -- during the backward pass. The pointer to a pointer ensures that all *Views* of the original tensor will point to the same data when the time comes.

[^2]: To make life easy for myself, I slice *inclusively*. So the last index is included in the slice.

[^3]: Negative indices can be converted into normal ones by adding the size of that slot to them. So -1 here is actually index \\( B - 1 \\)

[^4]: ``gooch::Slice(0, -1, -1)``