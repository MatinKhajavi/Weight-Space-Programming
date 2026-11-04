Capture stats from the two versions show big captures (two or more stones) being missed far more often than taken: 785 missed against 244 taken in v6, and 529 missed against 53 taken in v8. Many of those misses probably repeat across consecutive plies of the same chance, but the gap is still large. I'm wondering whether the AX/U1 handling for multi-stone groups is the cause.

I'm working out why big-group captures are missed. The SELF check has no exemption for groups larger than two stones, so when the capture interior gives a zero or one, the net score goes negative and the move gets skipped. I'm weighing whether exempting SELF for big captures would bring back the false self-atari repetitions, and whether a second, independent atari test could catch those false positives instead.

I'm looking at whether a size-based exemption could work: the trouble is that a mass threshold high enough to admit real 2–5 stone captures also pulls in long chain ends and interiors, which would bring back the false self-atari repetitions.

I'm checking whether a size-based exemption could cover real captures of two to five stones without also admitting long chains. A diameter-based reach check might work instead: if a group fits within about four steps of the liberty, the seed search covers all of it, so a lack of nearby seeds really does mean atari.

I'm still stuck on how to tell whether a group reaches beyond distance 3 from the liberty. Counting walks at different hop depths doesn't settle it, since the counts differ either way. A frontier signal would need the exact distance, which isn't directly available. I'm also considering whether richer seed coverage would help the big groups.

I'm checking whether the propagation could run from the liberty side instead, starting from every empty point next to the opponent's stones and spreading through the group. The catch is that excluding p has to happen per move, so this would need a separate pass for each candidate.

I'm considering a per-direction variant: for a capture at p through the stone q next to it, the group is in atari only if p is its sole liberty, so any stone with an empty neighbor other than p would rule that out. That check would have to run separately for each candidate point, though.

I'm now checking whether a stone can still count as having a liberty once we exclude p: a stone not touching p needs just one empty neighbor, while a stone touching p needs two. That suggests seed signals that depend only on how many empty neighbors each stone has, with the p-adjacent cases handled separately.

I'm noticing that stones with any liberty could act as seeds, but the trouble is that the stone next to p always has p as a liberty, so every position would look seeded. I'm considering whether the reach limit alone could separate true atari from false, since stones far from p cannot have p as a liberty.

I'm concluding that the reach limit is the real barrier: in big false-atari groups every liberty sits beyond the propagation's four layers, so denser seeds don't help. I'm now considering the reverse approach, treating big groups as safe by default and looking for a cheaper signal that picks out real big-group atari.

I'm now thinking about the capture from the move's side: if our stone at p takes three or more stones, the capture only counts as real when p is the group's sole liberty. Real big captures are usually compact, within about four steps, so the reach-based check should hold for them, though a long chain would still fall outside it.

I'm checking the chain-end case: walks of length 3 from an I4 end and an I8 end come out the same, so a pure walk count can't tell a short chain from a long one. Any reach-based signal would need to see beyond distance 3, and walks alone won't do that, so I'm considering larger reach through 2x2 blocks as super-nodes, where each hop covers two cells.

I'm considering whether a line-based shortcut could extend the reach: precomputing straight three-stone segments as connectors would let propagation jump two cells at a time, though the count of such segments per color makes this costly.

I'm weighing an accept-the-tradeoff option: allow the exemption for every capture, as in v2, and then catch the repetitions with a KO-style check aimed at the three-move cycle, where a self-atari is followed by a recapture.

I'm checking the recapture pattern from game 32, where the stone at (8,5) ends up with two liberties and so doesn't look like a ko. The cycle really starts with the self-atari, where filling our own second-to-last liberty is justified by a capture of a big group that may not actually be in atari. If that capture is real, the move is fine; if the group has two or more liberties, we are the ones left in atari, so I'm weighing a check on the group's liberty count to tell these apart.

I'm weighing a middle path: exempt the self-capture only under some condition on the move's neighborhood, though I can't yet compute how often real big captures occur versus false ones. A KO-style check won't catch this cycle, since the recapturing stone ends up with two liberties and doesn't look like a ko shape.

I'm weighing whether to skip the self-atari check only when the capture is big and uncertain and would leave our stone with a single liberty. That would avoid the disaster when the group actually has more liberties, though it gives up real big captures. I want to see how often the missed big captures involved that exact situation, but I can't tell from the current data.

I'm going to check whether the self-atari condition, computed with the true liberty counts, actually blocked the missed big captures, or whether they were missed for some other reason.

