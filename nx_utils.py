import networkx as nx
import random

from typing import Callable

def arbitrary_weight(low: int, high: int) -> Callable[[int, int], int]:
    return lambda _a, _b: random.randint(low, high)

def to_output_file(g: nx.classes.graph.Graph, decide_weight: Callable[[int, int], int], fname: str) -> None:
    with open(fname, 'w') as f:
        f.write(f'{g.number_of_nodes()} {g.number_of_edges()}\n')

        for edge in g.edges:
            f.write(f'{edge[0]} {edge[1]} {decide_weight(edge[0], edge[1])}\n')


if __name__ == '__main__':
    OUTDIR = 'testfiles'

    # circulant_n1000 = nx.circulant_graph(1000, [1, 2])
    # to_output_file(circulant_n1000, arbitrary_weight(1, 500), f'{OUTDIR}/circulant_n1000.txt')

    # circulant_n50000 = nx.circulant_graph(50000, [1, 2])
    # to_output_file(circulant_n50000, arbitrary_weight(1, 500), f'{OUTDIR}/circulant_n50000.txt')

    # circulant_n250000 = nx.circulant_graph(250000, [1, 2])
    # to_output_file(circulant_n250000, arbitrary_weight(1, 500), f'{OUTDIR}/circulant_n250000.txt')
