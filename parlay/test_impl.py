import math
import bisect
from sys import stdin, stdout
from math import gcd, floor, sqrt, log, log2, pi, ceil, prod
from collections import defaultdict as dd
from bisect import bisect_left as bl, bisect_right as br
from heapq import *
from collections import deque, Counter as ctr


inp = lambda: int(stdin.readline())
inpf = lambda: float(stdin.readline())
inps = lambda: stdin.readline().strip()

seq = lambda: list(map(int, stdin.readline().strip().split()))
seqc = lambda: list(stdin.readline().strip())

mul = lambda: map(int, stdin.readline().strip().split())
mulf = lambda: map(float, stdin.readline().strip().split())
muls = lambda: stdin.readline().strip().split()

jn = lambda x, l: x.join(map(str, l))

ceildiv = lambda x, d: -(x // -d)

flush = lambda: stdout.flush()
fastprint = lambda x: stdout.write(str(x))

mod = 998244353
class UnionFind:
    def __init__(self, n_verts: int) -> None:
        self.vertices = [i for i in range(n_verts)]

    def find(self, index: int) -> int:
        start = index

        while self.vertices[index] != index:
            index = self.vertices[index]

        self.vertices[start] = index
        return index

    def union(self, i: int, j: int):
        i = self.find(i)
        j = self.find(j)
        if i == j:
            return False
        self.vertices[i] = j
        return True

    def get_top_level(self, verts):
        return [x for x in verts if self.vertices[x] == x]

def main():
    # n is invalid
    [n, m] = seq()
    edges = [seq() for _ in range(m)]

    # verts = [i for i in range(n)]
    ds = UnionFind(n)
    t = 0
    while edges:

        bestv = [(n, math.inf) for _ in range(n)]
        for [u, v, w] in edges:
            if w < bestv[u][1]:
                bestv[u] =  (v, w)
            if  w < bestv[v][1]:
                bestv[v] =  (u, w)

        for u, (v, w) in enumerate(bestv):
            if v == n:
                continue
            if ds.union(u,v):
                t += w     

        # for u in verts:
        #     (v, w) = bestv[u]
        #     if ds.union(u,v):
        #         t += w
        # verts = ds.get_top_level(verts)

        # filter map
        en = []
        for u, v, w in edges:
            un = ds.find(u)
            vn = ds.find(v)
            if un != vn:
                en.append((un, vn, w))
        edges = en
        # edges.sort()
        # e_prev = (-1, -1, -1)
        # en = []
        # for e in edges:
        #     if e[0] == e_prev[0] and e[1] == e_prev[1]:
        #         continue
        #     en.append(e)
        #     e_prev = e
        # edges = en

        print(len(verts), len(edges))

    print(t)
MULT = False
rg = inp() if MULT else 1

for i in range(rg):
    ret = main()
    if ret is not None:
        if isinstance(ret, tuple) or isinstance(ret, list):
            print(*ret)
        else:
            print(ret)
