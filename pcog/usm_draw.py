import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque

def generate_graph(usm):
    current = usm.root
    G = nx.DiGraph()
    labels = {}
    idx = 0
    queue = deque([(idx, current)])
    while len(queue) != 0:
        pidx, current = queue.popleft()
        for key, child in current.children.items():
            idx+=1
            G.add_node(idx, tree_node=child)
            G.add_edge(pidx, idx, label=key)
            labels[idx] = key
            queue.append((idx, child))
    return G, labels

def draw_usm(usm, show=True, save_path="graph.png"):
    G, labels = generate_graph(usm)
    draw_graph(G, labels, show, save_path)

def draw_graph(G, labels, show, save_path):
    nx.draw(G, pos=graphviz_layout(G, prog='dot'), labels=labels, arrows=True)
    if show:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()