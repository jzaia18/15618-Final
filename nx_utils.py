import networkx as nx
import random

from typing import Any, Callable

def arbitrary_weight(low: int, high: int) -> Callable[[int, int], int]:
    return lambda _a, _b: random.randint(low, high)

def to_output_file(g: nx.classes.graph.Graph,
                   decide_weight: Callable[[Any, Any], int],
                   fname: str,
                   nodename_to_idx: Callable[[Any], int]= lambda x: int(x)) -> None:
    with open(fname, 'w') as f:
        f.write(f'{g.number_of_nodes()} {g.number_of_edges()}\n')

        for edge in g.edges:
            # Convert edge names to index
            u = nodename_to_idx(edge[0])
            v = nodename_to_idx(edge[1])
            f.write(f'{u} {v} {decide_weight(edge[0], edge[1])}\n')


if __name__ == '__main__':
    OUTDIR = 'testfiles'

    ## Generate Circulant Graphs

    # circulant_n10000 = nx.circulant_graph(10000, [1, 2])
    # to_output_file(circulant_n10000, arbitrary_weight(1, 500), f'{OUTDIR}/circulant_n10000.txt')

    # circulant_n50000 = nx.circulant_graph(50000, [1, 2])
    # to_output_file(circulant_n50000, arbitrary_weight(1, 500), f'{OUTDIR}/circulant_n50000.txt')

    # circulant_n250000 = nx.circulant_graph(250000, [1, 2])
    # to_output_file(circulant_n250000, arbitrary_weight(1, 500), f'{OUTDIR}/circulant_n250000.txt')

    ## Generate Hypercube Graphs

    # def hypercube_idx(node: list[int]) -> int:
    #     return sum(node[-i-1]* 2**i for i in range(len(node)))

    # hypercube_n5120 = nx.hypercube_graph(10)
    # to_output_file(hypercube_n5120, arbitrary_weight(1, 500), f'{OUTDIR}/hypercube_n5120.txt', nodename_to_idx=hypercube_idx)

    # hypercube_n32768 = nx.hypercube_graph(15)
    # to_output_file(hypercube_n32768, arbitrary_weight(1, 500), f'{OUTDIR}/hypercube_n32768.txt', nodename_to_idx=hypercube_idx)

    # hypercube_262144 = nx.hypercube_graph(18)
    # to_output_file(hypercube_262144, arbitrary_weight(1, 500), f'{OUTDIR}/hypercube_n262144.txt', nodename_to_idx=hypercube_idx)
