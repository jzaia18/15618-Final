import argparse
import random

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='GraphGen',
                                     description='Generate graphs for benchmarking')
    parser.add_argument('nvertices', type=int)
    parser.add_argument('-o', '--outfile', default='graph.txt')
    parser.add_argument('-d', '--density', default=0.5, type=float)
    parser.add_argument('--min-weight', default=1, type=int)
    parser.add_argument('--max_weight', default=100, type=int)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')

    args = parser.parse_args()

    total_edges = int(args.density * args.nvertices * (args.nvertices-1) / 2)

    if not args.quiet:
        print(f'Generating a graph on {args.nvertices} vertices...')
        print(f'  Density: {args.density} ({total_edges} edges)')
        print(f'  Edge weights between: [{args.min_weight}, {args.max_weight}]')

    adj_matrix = np.zeros((args.nvertices, args.nvertices), dtype=int)

    for _ in range(total_edges):
        # Generate a random edge
        new_spot = False

        # keep trying until an unoccupied spot is found
        while not new_spot:
            i = random.randint(0, args.nvertices-2)
            j = random.randint(i+1, args.nvertices-1) # ensure no self-loops
            if adj_matrix[i, j] == 0:
                new_spot = True

        w = random.randint(args.min_weight, args.max_weight)

        # Only bother filling upper triangle for undirected graphs
        adj_matrix[i, j] = w

    if args.verbose:
        print()
        print('Graph adjacency matrix:')
        print(adj_matrix)

    with open(args.outfile, 'w') as f:
        f.write(f'{args.nvertices} {total_edges}\n')
        for i in range(args.nvertices):
            for j in range(i+1, args.nvertices):
                if adj_matrix[i,j] != 0:
                    f.write(f'{i} {j} {adj_matrix[i,j]}\n')

'''
File format:

<nvertices> <nedges>
<v1> <v2> <w>
<v1> <v2> <w>
...

'''
