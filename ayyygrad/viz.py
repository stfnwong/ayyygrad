# Visualize a graph
# This is basically stolen from Karpathy via the micrograd video - all credit to him


from graphviz import Digraph
from ayyygrad.value import Value


def trace(root: Value):
    nodes = set()
    edges = set()

    def topo_sort(v: Value):
        if v not in nodes:
            nodes.add(v)
            for e in v.parents:
                edges.add((e, v))
                topo_sort(e)

    topo_sort(root)

    return nodes, edges


def draw(root: Value):
    dot = Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"},
    )
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))

        # Make a rectangle for each of the nodes
        dot.node(name=uid, label=f"{n.label} | data: {n.data:.4f}", shape="record")

        if n.op:
            dot.node(name=uid + n.op, label=n.op)
            dot.edge(uid+n.op, uid)

    # Connect edges
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
