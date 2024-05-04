## Tester for running various parallel boruvkas implementations

import re

from typing import Any, Callable

import networkx as nx

def get_outputs(_stdout: bytes) -> None:
    stdout = _stdout.decode('utf-8')

    match = re.search(r'Initialization time \(sec\): ([\d\.]*)', stdout)
    init_time = float(match.group(1))

    compute_times = re.findall(r'Computation time \(sec\): ([\d\.]*)', stdout)
    compute_times = [float(x) for x in compute_times]

    weights = re.findall(r'Total weight: (\d*)', stdout)
    weights = [int(x) for x in weights]

    return {
        'init_time': init_time,
        'compute_times': compute_times,
        'weights': weights,
    }

if __name__ == '__main__':
    import argparse
    import os
    import subprocess

    import nx_utils

    parser = argparse.ArgumentParser(prog='mstbench',
                                     description='Benchmark different parallel MST implementations')
    parser.add_argument('-r', '--reps',
                        default=5,
                        help='the number of times to repeat each experiment',
                        type=int)
    parser.add_argument('-s', '--seed',
                        default=0,
                        help='the seed value to use for generating random graphs',
                        type=int)
    parser.add_argument('--min-weight',
                        default=1,
                        help='the minimum edge weight in random graphs',
                        type=int)
    parser.add_argument('--max-weight',
                        default=1000,
                        help='the maximum edge weight in random graphs',
                        type=int)

    args = parser.parse_args()

    binmode = True # probably want to use this for almost every case, but could make an argument

    # Locations to store on-the-fly generated graphs
    TEMPFILE_DIR = '/tmp/mst_test/'
    if binmode:
        TEMPFILE_NAME = 'test.bin'
    else:
        TEMPFILE_NAME = 'test.txt'

    os.makedirs(TEMPFILE_DIR, exist_ok=True)
    TEMPFILE_PATH = os.path.join(TEMPFILE_DIR, TEMPFILE_NAME)

    def create_arb_weight_test(g_fxn: Callable[[Any], nx.classes.graph.Graph],
                               g_args: list[Any],
                               nodename_to_idx: Callable[[Any], int]= lambda x: int(x)) -> Callable[[], None]:
        def inner():
            g = g_fxn(*g_args)
            nx_utils.to_output_file(g,
                                    nx_utils.arbitrary_weight(args.min_weight, args.max_weight, args.seed),
                                    TEMPFILE_PATH,
                                    binary=binmode,
                                    nodename_to_idx=nodename_to_idx)

        return inner


    impls = {
        'Sequential': '../sequential/boruvkas',
        # 'CPU (parlaylib) parallel': '../parlay/boruvkas',
        'GPU (CUDA) parallel': '../cuda/boruvkas',
    }

    tests = {
        '2-degree Circulant n=50000':
            create_arb_weight_test(nx.circulant_graph,
                                   (50000, [1, 2]),
            ),

        '2-degree Circulant n=250000':
            create_arb_weight_test(nx.circulant_graph,
                                   (250000, [1, 2]),
            ),

        '5-degree Circulant n=500000':
            create_arb_weight_test(nx.circulant_graph,
                                   (500000, [1, 2, 3, 4, 5]),
            ),

        'Hypercube d=15, n=32768':
            create_arb_weight_test(nx.hypercube_graph,
                                   (15,),
                                   lambda node: sum(node[-i-1]* 2**i for i in range(len(node))),
            ),

        'Hypercube d=18, n=262144':
            create_arb_weight_test(nx.hypercube_graph,
                                   (18,),
                                   lambda node: sum(node[-i-1]* 2**i for i in range(len(node))),
            ),

        'Caveman Graph, 5000 groups of size k=40, n=200000':
            create_arb_weight_test(nx.caveman_graph,
                                   (5000, 40),
            ),

        'Caveman Graph, 7500 groups of size k=70, n=525000':
            create_arb_weight_test(nx.caveman_graph,
                                   (7500, 70),
            ),

        'Connected Caveman Graph, 5000 groups of size k=40, n=200000':
            create_arb_weight_test(nx.connected_caveman_graph,
                                   (5000, 40),
            ),

        'Connected Caveman Graph, 7500 groups of size k=70, n=525000':
            create_arb_weight_test(nx.connected_caveman_graph,
                                   (7500, 70),
            ),
    }


    for (test_name, test_gen) in tests.items():
        print(f'Generating graph for test "{test_name}"...')
        # Generate test file
        test_gen()

        for (impl, exe) in impls.items():
            print(f'  Running {impl} impl on test "{test_name}"...')

            if binmode:
                file_flag = '-bf'
            else:
                file_flag = '-f'
            result = subprocess.run([exe, file_flag, TEMPFILE_PATH, '-r', str(args.reps)],
                                    capture_output=True)
            x = get_outputs(result.stdout)

            print('   ', x)
            print()
        print('\n')
