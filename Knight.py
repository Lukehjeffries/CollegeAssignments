from collections import deque

FILES = "abcdefgh"
RANKS = "12345678"

def parse_square(s):
    s = s.strip().lower()
    if len(s) != 2 or s[0] not in FILES or s[1] not in RANKS:
        raise ValueError("Use notation like a1..h8")
    c = FILES.index(s[0])
    r = int(s[1]) - 1
    return (r, c)

def square_str(rc):
    r, c = rc
    return f"{FILES[c]}{r+1}"

def in_bounds(r, c):
    return 0 <= r < 8 and 0 <= c < 8

DELTA = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]

def neighbors(rc):
    r, c = rc
    for dr, dc in DELTA:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc):
            yield (nr, nc)

def min_knight_path(start, target):
    """Return shortest [start,...,target] using BFS, or None."""
    ########################################################
    # >>> STUDENT TODO START: BFS min_knight_path
    # Use a queue, an explored set, and a parent map.
    # Enqueue start, then expand neighbors until you dequeue target.
    # Reconstruct the path by following parents from target back to start and reverse.
    if start == target:
        return [start]

    q = deque()
    q.append(start)
    visited = set([start])
    parent = {start: None}

    while q:
        cur = q.popleft()
        if cur == target:
            # reconstruct path
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        for nb in neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                q.append(nb)
    # Should not happen on empty chessboard but follow spec
    return None
    # <<< STUDENT TODO END
    ########################################################

if __name__ == "__main__":
    try:
        s = input("Start square (e.g., a1): ")
        t = input("Target square (e.g., h8): ")
        start = parse_square(s)
        target = parse_square(t)
    except Exception as e:
        print(f"Input error: {e}"); exit(1)

    if start == target:
        print("Moves: 0")
        print(square_str(start))
    else:
        path = min_knight_path(start, target)
        if path is None:
            print("No path? That would be odd on an empty chessboard.")
        else:
            print(f"Moves: {len(path)-1}")
            print(" -> ".join(square_str(p) for p in path))
