import sys

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

        self.vertices[i] = j


class Edge:
    def __init__(self, u: int, v: int, weight: int) -> None:
        self.u = u
        self.v = v
        self.weight = weight

    @classmethod
    def from_line(cls, s: str) -> 'Edge':
        parts = s.split()
        return Edge(*[int(token) for token in parts])

    def __repr__(self):
        return f'({self.u}, {self.v}, {self.weight})'

    __str__ = __repr__


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} <filename>')
        sys.exit(1)

    fname = sys.argv[1]
    verbose = (len(sys.argv) > 2)

    with open(fname, 'r') as f:
        header = f.readline().split()
        nvertices = int(header[0])
        nedges = int(header[1])

        edges = []

        for line in f:
            edges.append(Edge.from_line(line))

    edges.sort(key=lambda e: e.weight)
    uf = UnionFind(nvertices)
    mst = []

    # perform kruskals
    for edge in edges:
        if uf.find(edge.u) != uf.find(edge.v):
            mst.append(edge)
            uf.union(edge.u, edge.v)

    print('Final MST sum:', sum(e.weight for e in mst))
    if verbose:
        print(mst)
