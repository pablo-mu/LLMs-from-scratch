# Chapter 3: Coding Attention Mechanisms

In the [03-attention-mechanisms.ipynb](03-coding-attention-mechanisms/03-attention-mechanisms.ipynb) notebook, following the book, I implemented the multi-head attention mechanism and the transformer model from scratch using PyTorch. 

In this chapter, I learned that:

* Attention mechanisms transform input elements into enhanced context vector representations that incorporate information about all inputs.

* A self-attention mechanism computes the context vector representation as a
weighted sum over the inputs.

* In a simplified attention mechanism, the attention weights are computed via
dot products.

* Matrix multiplications, while not strictly required, help us implement computations
more efficiently and compactly by replacing nested for loops.

* In self-attention mechanisms used in LLMs, also called scaled-dot product
attention, we include trainable weight matrices to compute intermediate transformations
of the inputs: queries, values, and keys.

* When working with LLMs that read and generate text from left to right, we add
a causal attention mask to prevent the LLM from accessing future tokens.

* In addition to causal attention masks to zero-out attention weights, we can add
a dropout mask to reduce overfitting in LLMs.

* The attention modules in transformer-based LLMs involve multiple instances of
causal attention, which is called multi-head attention.

* We can create a multi-head attention module by stacking multiple instances of
causal attention modules.

* A more efficient way of creating multi-head attention modules involves batched
matrix multiplications.

## Bonus Material

Additionally, I added the Raschka's notebooks of the chapter 3 for future learning purposes:

* [mha-implementations.ipynb](03-coding-attention-mechanisms/mha-implementations.ipynb) - Implements and compares different implementations of multi-head attention. 

* [understanding-buffers.ipynb](03-coding-attention-mechanisms/understanding-buffers.ipynb) - Explains the concept of buffers in PyTorch.
