"""Replay repetition-ended games with a tiny Go engine (Go rules only) to see what the repeating move is."""
import collections
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.exp6 import PASS

out = pickle.load(open(Path(__file__).with_name('exp6_rep.pkl'), 'rb'))
NB = [[(r+dr)*9+c+dc for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)) if 0 <= r+dr < 9 and 0 <= c+dc < 9] for r in range(9) for c in range(9)]
def group(b, p):
    col, st, seen, libs = b[p], [p], {p}, set()
    while st:
        q = st.pop()
        for n in NB[q]:
            if b[n] == 0: libs.add(n)
            elif b[n] == col and n not in seen: seen.add(n); st.append(n)
    return seen, libs
def replay(moves):
    b = [0]*81; hist = [tuple(b)]; info = []
    for t, m in enumerate(moves):
        if m < 0: break
        col = 1 if t % 2 == 0 else -1
        cap = 0; own_libs_after = None
        if m != PASS:
            assert b[m] == 0, (t, m)
            b[m] = col
            for n in NB[m]:
                if b[n] == -col:
                    g, l = group(b, n)
                    if not l:
                        for q in g: b[q] = 0
                        cap += len(g)
            g, l = group(b, m)
            assert l, ('suicide', t)
            own_libs_after = len(l); own_size = len(g)
        else:
            own_size = 0
        info.append((m, cap, own_libs_after, own_size))
        hist.append(tuple(b))
    return hist, info
def analyse(name):
    r = out[name]
    rep = np.where(r['end'] == 1)[0]
    cyc = collections.Counter(); caps = collections.Counter()
    eg = []
    nonrep_ok = 0
    for gi in rep:
        hist, info = replay(r['moves'][gi])
        T = len(info) - 1
        mover_a = (T % 2 == 0) == r['a_black'][gi]
        last = hist[-1]
        prev = [i for i in range(len(hist)-1) if hist[i] == last]
        assert prev, gi
        k = len(hist) - 1 - prev[-1]   # cycle length in moves (incl. passes)
        passes_in_cycle = sum(1 for m, *_ in info[-k:] if m == PASS)
        m, cap, libs, size = info[-1]
        who = 'A' if mover_a else 'B'
        cyc[(who, k, passes_in_cycle)] += 1
        caps[(who, 'cap' if cap else 'nocap', cap, libs, size)] += 1
        if who == 'A' and len(eg) < 3:
            eg.append((gi, k, [(mm, cc) for mm, cc, *_ in info[-k:]]))
    # sanity: a few non-rep games must have no repeated stone-move board
    for gi in np.where(r['end'] != 1)[0][:200]:
        hist, info = replay(r['moves'][gi])
        bad = any(info[t][0] != PASS and hist[t+1] in set(hist[:t+1]) for t in range(len(info)))
        nonrep_ok += not bad
    print(f'== {name}: {len(rep)} repetition games; non-rep sanity {nonrep_ok}/200')
    print('  (mover, cycle length, passes in cycle):', sorted(cyc.items(), key=lambda x: -x[1])[:8])
    print('  (mover, captured?, n captured, own libs after, own group size):', sorted(caps.items(), key=lambda x: -x[1])[:8])
    for e in eg: print('  example A cycle: game', e[0], 'len', e[1], 'moves (move, captured):', e[2])
for name in ['opus_ppo0', 'opus_ppo1', 'astra_ppo0']:
    analyse(name)
