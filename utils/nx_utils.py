import networkx as nx
import random

from typing import Any, Callable

def arbitrary_weight(low: int, high: int, seed: int=0) -> Callable[[int, int], int]:
    random.seed(seed)
    return lambda _a, _b: random.randint(low, high)

def to_output_file(g: nx.classes.graph.Graph,
                   decide_weight: Callable[[Any, Any], int],
                   fname: str,
                   binary: bool=False,
                   nodename_to_idx: Callable[[Any], int]= lambda x: int(x)) -> None:

    if binary:
        to_bin = lambda num: int(num).to_bytes(length=4, byteorder='little')
        with open(fname, 'wb') as f:
            f.write(to_bin(g.number_of_nodes()))
            f.write(to_bin(g.number_of_edges()))

            for edge in g.edges:
                # Convert edge names to index
                u = nodename_to_idx(edge[0])
                v = nodename_to_idx(edge[1])
                f.write(to_bin(u))
                f.write(to_bin(v))
                f.write(to_bin(decide_weight(edge[0], edge[1])))
    else:
        with open(fname, 'w') as f:
            f.write(f'{g.number_of_nodes()} {g.number_of_edges()}\n')

            for edge in g.edges:
                # Convert edge names to index
                u = nodename_to_idx(edge[0])
                v = nodename_to_idx(edge[1])
                f.write(f'{u} {v} {decide_weight(edge[0], edge[1])}\n')


if __name__ == '__main__':
    OUTDIR = '../testfiles'

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

    ## Generate Connected Caveman Graphs

    # caveman_1000 = nx.connected_caveman_graph(100, 10)
    # to_output_file(caveman_1000, arbitrary_weight(1, 500), f'{OUTDIR}/conn_caveman_n1000.txt')

    # caveman_62500 = nx.connected_caveman_graph(2000, 25)
    # to_output_file(caveman_62500, arbitrary_weight(1, 500), f'{OUTDIR}/conn_caveman_n62500.txt')

    # caveman_200000 = nx.connected_caveman_graph(5000, 40)
    # to_output_file(caveman_200000, arbitrary_weight(1, 500), f'{OUTDIR}/conn_caveman_n200000.txt')

    # caveman_360000 = nx.connected_caveman_graph(6000, 60)
    # to_output_file(caveman_360000, arbitrary_weight(1, 500), f'{OUTDIR}/conn_caveman_n360000.txt')

    ## Generated (Disconnected) Caveman Graphs

    # caveman_200000 = nx.caveman_graph(5000, 40)
    # to_output_file(caveman_200000, arbitrary_weight(1, 500), f'{OUTDIR}/caveman_n200000.txt')

    # caveman_360000 = nx.caveman_graph(6000, 60)
    # to_output_file(caveman_360000, arbitrary_weight(1, 500), f'{OUTDIR}/caveman_n360000.txt')
