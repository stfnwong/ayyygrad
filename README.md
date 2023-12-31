# Some simple Autograd

This is a quick run through automatic differentiation based on [this](https://vmartin.fr/understanding-automatic-differentiation-in-30-lines-of-python.html) blogpost.

It turns out that the understanding of automatic differentiation that one can get from an example like this is quite limited. For one, it presumes that the computation graph is a tree, when in practice its more like a DAG.

### If this sucks so much, then how does everyone else do it?

So its possible (if one is mathematically inclined enough) to prove the chain rule for ordered derivatives by induction. Apparently the proof appears in _Maximizing long-term gas industry profits in two minutes in lotus using neural network methods, 1989_ but I've not seen this myself. 


## Setup
I am using `poetry` for setting up the Python environment even though I hate it (I hate it less than I hate every other alternative for managing environments in Python). To setup

```
poetry install -vvv
```

Take the `-vvv` off to see less text. Then

```
source $(poetry env info -p)/bin/activate
```

when you are sick of the environment do 

```
deactivate
```

to turn it off.


## Notebooks
There are some notebooks in the `notebooks` folder. These are all in `percent` script format. If like me you are working locally and can't be bothered with security then you can start the notebook by 

- Activating the python env with `source $(poetry env info -p)/bin/activate` or some equivalent.
- Running `jupyter notebook --ip="0.0.0.0" --NotebookApp.token="" --NotebookApp.password=""`
- May not be wise to do this over a network - depends on the network.

