## Tester for running various parallel boruvkas implementations

import re

from typing import Any, Callable

import networkx as nx

def get_outputs(_stdout: bytes) -> None:
    stdout = _stdout.decode('utf-8')

    match = re.search(r'Initialization time \(sec\): ([\d\.]*)', stdout)
    init_time = float(match.group(1))

    match = re.search(r'File read time \(sec\): ([\d\.]*)', stdout)
    read_time = float(match.group(1))

    compute_times = re.findall(r'Computation time \(sec\): ([\d\.]*)', stdout)
    compute_times = [float(x) for x in compute_times]

    weights = re.findall(r'Total weight: (\d*)', stdout)
    weights = [int(x) for x in weights]

    return {
        'init_time': init_time,
        'read_time': read_time,
        'compute_times': compute_times,
        'weights': weights,
    }

def print_stats(all_metrics: dict[Any, Any], baseline: str) -> None:
    for impl in all_metrics:
        if impl == baseline:
            continue

        print(f'Performance of {impl}:')
        all_tests = all_metrics[impl]
        comp_speedups = []
        tot_speedups = []
        tot_no_read_speedups = []
        for (test, metrics) in all_tests.items():
            print(f'  {test} ({len(metrics["compute_times"])} runs):')

            if 'weight' not in metrics or metrics['weight'] != metrics['weight']:
                print('Inconsistent result on this test')
                continue

            compute_time = metrics['avg_compute_time']
            init_time = metrics['init_time']
            read_time = metrics['read_time']
            tot_no_read_time = compute_time + init_time
            tot_time = tot_no_read_time + read_time

            comp_speedup = all_metrics[baseline][test]['avg_compute_time'] / metrics['avg_compute_time']
            tot_no_read_speedup = (all_metrics[baseline][test]['avg_compute_time'] + all_metrics[baseline][test]['init_time']) / tot_no_read_time
            tot_speedup = (all_metrics[baseline][test]['avg_compute_time'] + all_metrics[baseline][test]['init_time'] + all_metrics[baseline][test]['read_time']) / tot_time

            comp_speedups.append(comp_speedup)
            tot_no_read_speedups.append(tot_no_read_speedup)
            tot_speedups.append(tot_speedup)

            print(f'    Compute time = {compute_time:0.4f}s,  Init time = {init_time:0.4f}s,  Read time = {read_time:0.4f}s,  Total time = {tot_time:0.4f}s, Total time (without read) = {tot_no_read_time:0.4f}s')
            print(f'    Compute speedup={comp_speedup:0.2f}x, Total speedup={tot_speedup:0.2f}x, Total (without read) speedup={tot_no_read_speedup:0.2f}x')
            print()


        print(f'Average computation time speedup of {impl}: {sum(comp_speedups)/len(comp_speedups):0.2f}')
        print(f'Average total time speedup of {impl}: {sum(tot_speedups)/len(tot_speedups):0.2f}')
        print(f'Average total time (without read) speedup of {impl}: {sum(tot_no_read_speedups)/len(tot_no_read_speedups):0.2f}')
        print()

if __name__ == '__main__':
    import argparse
    import os
    import subprocess

    import nx_utils

    parser = argparse.ArgumentParser(prog='mstbench',
                                     description='Benchmark different parallel MST implementations')
    parser.add_argument('--baseline-reps',
                        default=3,
                        help='the number of times to repeat each experiment for the sequential baseline',
                        type=int)
    parser.add_argument('--parallel-reps',
                        default=10,
                        help='the number of times to repeat each experiment for the parallel implementations',
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

    # Which impl is the one being benchmarked against
    BASELINE = 'Sequential'

    impls = {
        BASELINE: '../sequential/boruvkas',
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

        # '5-degree Circulant n=500000':
        #     create_arb_weight_test(nx.circulant_graph,
        #                            (500000, [1, 2, 3, 4, 5]),
        #     ),

        # 'Hypercube d=15, n=32768':
        #     create_arb_weight_test(nx.hypercube_graph,
        #                            (15,),
        #                            lambda node: sum(node[-i-1]* 2**i for i in range(len(node))),
        #     ),

        # 'Hypercube d=18, n=262144':
        #     create_arb_weight_test(nx.hypercube_graph,
        #                            (18,),
        #                            lambda node: sum(node[-i-1]* 2**i for i in range(len(node))),
        #     ),

        # 'Caveman Graph, 5000 groups of size k=40, n=200000':
        #     create_arb_weight_test(nx.caveman_graph,
        #                            (5000, 40),
        #     ),

        # 'Caveman Graph, 7500 groups of size k=70, n=525000':
        #     create_arb_weight_test(nx.caveman_graph,
        #                            (7500, 70),
        #     ),

        # 'Connected Caveman Graph, 5000 groups of size k=40, n=200000':
        #     create_arb_weight_test(nx.connected_caveman_graph,
        #                            (5000, 40),
        #     ),

        # 'Connected Caveman Graph, 7500 groups of size k=70, n=525000':
        #     create_arb_weight_test(nx.connected_caveman_graph,
        #                            (7500, 70),
        #     ),

        'Binomial Graph, p=8e-5 n=350000':
            create_arb_weight_test(nx.fast_gnp_random_graph,
                                   (350000, 8e-5),
            ),

        # 'Binomial Graph, p=5e-4 n=350000':
        #     create_arb_weight_test(nx.fast_gnp_random_graph,
        #                            (350000, 5e-4),
        #     ),
    }

    all_metrics = {
        impl: {} for impl in impls.keys()
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

            if impl == BASELINE:
                nreps = str(args.baseline_reps)
            else:
                nreps = str(args.parallel_reps)

            proc_output = subprocess.run([exe, file_flag, TEMPFILE_PATH, '-r', nreps],
                                         capture_output=True)
            metrics = get_outputs(proc_output.stdout)

            metrics['avg_compute_time'] = sum(metrics['compute_times'])/len(metrics['compute_times'])

            if min(metrics['weights']) == max(metrics['weights']):
                metrics['weight'] = min(metrics['weights'])
                del metrics['weights']
            else:
                print(f'!!! Error on {impl}: inconsistent outputs')

            all_metrics[impl][test_name] = metrics

            print('   ', metrics)
            print()
        print()

    print_stats(all_metrics, BASELINE)
