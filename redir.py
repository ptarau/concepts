from typing import Iterable, Tuple, Dict, Set, Optional
import unittest

Edge = Tuple[str, str, str]  # (source, edge_label, target)


def redirect_edges_no_backflow(
    edges: Iterable[Edge],
    ranks: Dict[str, float],
    topn: int,
    *,
    drop_self_loops: bool = True,
) -> Set[Edge]:
    """
    Keep the top-n highest-ranked nodes. For edges incident to dropped nodes:

      • Dropped SOURCE  -> redirect downstream to a kept SUCCESSOR (no backflow).
        Prefer a successor different from the (kept/mapped) target to avoid self-loops.

      • Dropped TARGET  -> redirect upstream to a kept PREDECESSOR (no backflow).
        Prefer a predecessor different from the (kept/mapped) source to avoid self-loops.

      • If no valid neighbor exists, drop the edge.

    Edge labels are preserved; duplicates removed by returning a set.
    """

    edges = list(edges)
    all_nodes = {u for (u, _, v) in edges} | {v for (_, _, v) in edges}

    # Determine kept nodes: sort by rank desc, tie-break by name asc for determinism
    ordered = sorted(all_nodes, key=lambda n: (-ranks.get(n, float("-inf")), n))
    kept = set(ordered[:max(0, topn)])

    # Build adjacency
    preds: Dict[str, Set[str]] = {n: set() for n in all_nodes}
    succs: Dict[str, Set[str]] = {n: set() for n in all_nodes}
    for s, _, t in edges:
        succs[s].add(t)
        preds[t].add(s)

    def pick_best_kept(cands: Set[str], *, exclude: Set[str] = frozenset()) -> Optional[str]:
        """
        Pick highest-ranked kept node from candidates, preferring those not in `exclude`.
        If nothing remains after exclusion, fall back to any kept candidate.
        """
        primary = [n for n in cands if n in kept and n not in exclude]
        if primary:
            return max(primary, key=lambda n: (ranks.get(n, float("-inf")), -len(n), n))
        fallback = [n for n in cands if n in kept]
        if fallback:
            return max(fallback, key=lambda n: (ranks.get(n, float("-inf")), -len(n), n))
        return None

    out: Set[Edge] = set()

    for s, lbl, t in edges:
        # Map source (downstream only if dropped)
        if s in kept:
            ms = s
        else:
            # Avoid creating a self-loop if target already kept
            exclude = {t} if t in kept else set()
            ms = pick_best_kept(succs.get(s, set()), exclude=exclude)
            if ms is None:
                continue  # cannot map dropped source

        # Map target (upstream only if dropped)
        if t in kept:
            mt = t
        else:
            # Avoid creating a self-loop by excluding the already-mapped source
            mt = pick_best_kept(preds.get(t, set()), exclude={ms})
            if mt is None:
                continue  # cannot map dropped target

        if drop_self_loops and ms == mt:
            continue

        out.add((ms, lbl, mt))

    return out


# =========================
#        Unit Tests
# =========================

class TestRedirectNoBackflow(unittest.TestCase):
    def setUp(self):
        # Create 20 nodes a..t with descending ranks: a highest, t lowest
        self.nodes = [chr(ord('a') + i) for i in range(20)]  # a..t
        # a:20, b:19, ..., t:1
        self.ranks = {n: 20 - i for i, n in enumerate(self.nodes)}
        self.topn = 10  # keep a..j
        self.kept = set(self.nodes[:self.topn])

        E = set()

        # (1) Simple chain a->b->...->t (labels e_ab, e_bc, ...)
        for i in range(len(self.nodes) - 1):
            s, t = self.nodes[i], self.nodes[i + 1]
            E.add((s, f"e_{s}{t}", t))

        # (2) Kept -> Low edges to test target-upstream redirection (m, r are low)
        # Provide multiple KEPT predecessors for 'm' and 'r' so choice is deterministic.
        E.update({
            ('c', 'cl_to_m', 'm'),   # target low -> should map upstream to best kept predecessor ≠ source
            ('h', 'h_to_r', 'r'),

            ('d', 'd_to_m', 'm'),    # kept preds for m: {c,d,j}
            ('j', 'j_to_m', 'm'),

            ('e', 'e_to_r', 'r'),    # kept preds for r: {e,h}
            ('h', 'h2_to_r', 'r'),
        })

        # (3) Low -> Kept edges to test source-downstream redirection (no backflow)
        # k (low) has kept successors {b, d}; p (low) has kept successors {f, i}
        E.update({
            ('k', 'k_to_b', 'b'),
            ('k', 'k_to_d', 'd'),
            ('p', 'p_to_i', 'i'),
            ('p', 'p_to_f', 'f'),
        })
        # Note: ranks(b)=19 > d=17; ranks(f)=15 > i=12

        # (4) Low -> Low edges with kept neighbors both sides; both ends should redirect
        # p -> r; kept succ of p: {f, i}; kept preds of r: {e, h} (e has higher rank)
        E.update({
            ('p', 'p_to_r', 'r'),
            ('p', 'p_to_g', 'g'),    # g is kept successor of p via chain
            ('p', 'p_to_i2', 'i'),   # i is kept successor of p via explicit edge
            ('e', 'e_to_r2', 'r'),   # kept predecessor of r
            ('h', 'h_to_r2', 'r'),   # another kept predecessor of r
        })

        # (5) Low -> Low edge with no viable mapping; s has no kept succ beyond t; t has no kept preds beyond s
        E.update({
            ('s', 's_to_t', 't'),
        })

        self.edges = E

    def test_redirection(self):
        out = redirect_edges_no_backflow(self.edges, self.ranks, self.topn, drop_self_loops=True)

        # (A) All endpoints must be in kept set
        used_nodes = {x for e in out for x in (e[0], e[2])}
        self.assertTrue(used_nodes.issubset(self.kept))

        # (B) No self-loops when drop_self_loops=True
        self.assertTrue(all(s != t for s, _, t in out))

        # (C) Specific expected redirections

        # c -> m (target low): kept predecessors of m are {c, d, j}; exclude source c -> pick d
        self.assertIn(('c', 'cl_to_m', 'd'), out)

        # h -> r (target low): kept predecessors of r are {e, h}; exclude source h -> pick e
        self.assertIn(('h', 'h_to_r', 'e'), out)

        # k -> d (source low): kept successors of k are {b, d}; prefer highest rank b
        self.assertIn(('b', 'k_to_d', 'd'), out)

        # p -> i (source low): kept successors {f, i}; pick f; expect (f, 'p_to_i', 'i)
        self.assertIn(('f', 'p_to_i', 'i'), out)
        # p -> f becomes (f, 'p_to_f', 'f') which is a self-loop and should be removed
        self.assertNotIn(('f', 'p_to_f', 'f'), out)

        # p -> r (both low): source maps to best kept succ of p = f; target maps to best kept pred of r = e
        self.assertIn(('f', 'p_to_r', 'e'), out)

        # (D) Edge with no valid redirection should be dropped
        self.assertTrue(all(lbl != 's_to_t' for _, lbl, _ in out))

        # (E) Some original kept->kept chain edges survive unchanged
        self.assertTrue({('a', 'e_ab', 'b'), ('b', 'e_bc', 'c'), ('i', 'e_ij', 'j')}.issubset(out))

        # (F) Size sanity: non-trivial but not exploding
        self.assertGreater(len(out), 10)
        self.assertLess(len(out), len(self.edges))


if __name__ == "__main__":
    unittest.main(verbosity=2)
