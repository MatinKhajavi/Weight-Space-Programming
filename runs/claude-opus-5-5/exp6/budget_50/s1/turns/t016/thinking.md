The false atari cases cluster where the chain has stones at exactly four hops, so the search horizon is what's causing them. Splitting on that "far" flag, true ataris are 294 versus 1224 false ones when far, and 3153 true versus 115 false when not far. So flagging far chains as uncertain would keep most of the true ataris while dropping almost all the false ones, and I'm still looking at the far-case numbers.

I'm thinking about how to flag "far" chains, the ones with a stone at distance four. A simple option is to threshold the count of stones within four hops, since the far cases look mostly false when that count is high. I'm also considering a walk-count signal that grows with branching, though it's not a clean size measure.

Now I'm working out what a "far" flag should look like, since walk counts don't separate a spread-out chain from a compact one: a 2x2 block gets a higher count than a five-stone line. A geometric test, like whether the chain reaches past a set radius, might capture the far cases better.

I'm checking whether a walk-based count could flag the far-reaching chains. If I propagate counts without self-loops, an isolated stone stays at zero after the first step, a domino stays at one, and a chain's counts stay positive for as long as walks can keep running. So the signal would depend on how far the chain extends, which is what I need.

I'm considering a different proxy for "far": counting same-colored stones in a small diamond around q, since a large local mass probably means a large chain, and that needs no connectivity computation. From the table, a count of six or more nearby stones is mostly false atari, so that threshold could be the uncertainty flag.

I'm checking whether the false ataris line up with local density: I'll count same-colored stones within Manhattan distance 2 (and maybe 3) of q, ignoring connectivity, and compare the true and false split against that count.

