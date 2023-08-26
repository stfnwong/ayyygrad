# Some simple Autograd

This is a quick run through automatic differentiation based on [this](https://vmartin.fr/understanding-automatic-differentiation-in-30-lines-of-python.html) blogpost.

It turns out that the understanding of automatic differentiation that one can get from an example like this is quite limited. For one, it presumes that the computation graph is a tree, when in practice its more like a DAG.

### If this sucks so much, then how does everyone else do it?

So its possible (if one is mathematically inclined enough) to prove the chain rule for ordered derivatives by induction. Apparently the proof appears in _Maximizing long-term gas industry profits in two minutes in lotus using neural network methods, 1989_ but I've not seen this myself.

I probably don't need to fully grasp the proof to do an implementation, but that leaves me wondering how is it done in other implementations?


#### Micrograd
The `backward()` method on a `Value` performs a topological sort of all nodes, then walks the sorted array in reverse and calls `backward()` on each node.


#### Tinygrad
Also does a reverse topological sort. 
