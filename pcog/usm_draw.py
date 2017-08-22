import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import deque

def _get_color(node):
    if node.is_fringe:
        return 'r'
    else:
        return 'b'

def generate_graph(usm):
    current = usm.get_root()
    G = nx.DiGraph()
    labels = {}
    idx = 0
    queue = deque([(idx, current)])
    colors = [_get_color(current)]
    while len(queue) != 0:
        pidx, current = queue.popleft()
        for key, child in current.children.items():
            idx+=1
            G.add_node(idx, tree_node=child)
            colors.append(_get_color(child))
            G.add_edge(pidx, idx, label=key)
            labels[idx] = key
            queue.append((idx, child))
    return G, labels, colors

def draw_usm(usm, show=True, save_path="graph.png"):
    G, labels, colors = generate_graph(usm)
    draw_graph(G, labels, 
               show=show, 
               save_path=save_path, 
               colors=colors)

def update_usm_drawing(usm):
    G, labels, colors = generate_graph(usm)
    draw_graph(G, labels, show=True, save_path=None, close_plot=False)


def draw_graph(G, labels, colors, show, save_path, close_plot=True):
    nx.draw(G, pos=graphviz_layout(G, prog='dot'), labels=labels, arrows=True, node_color=colors)
    if show:
        plt.show()
    else:
        plt.savefig(save_path)
    if close_plot:
        plt.close()