In the v8 run, the first pass comes in 74% of games, around ply 79 instead of 90, so games end shorter. That suggests we're passing too early and handing the opponent extra moves. I'm wondering whether the pass threshold of −10 is the issue, since many late moves now carry strongly negative logits.

I'm weighing whether a flat self-atari penalty is too blunt, since moderate values like sa=5 might still catch legitimate dame-filling and blocking moves in the endgame. A more precise check would compute the merged liberty count of the placed stone together with adjacent own chains, and flag a self-atari only when that count is one or fewer.

I'm checking whether a stone that touches one of our own chains is actually safe, and that depends on whether the chain's other liberty could be the same point we're playing on. If the diagonal point between the neighboring chain and the empty spot is also one of our stones, the chain probably reaches that spot too, so that case needs care. A neighboring chain with three or more liberties would settle it outright, and two separate chains with two liberties each, pointing at different other liberties, might be enough as well.

I'm working out a refined self-atari check: a stone with one liberty is only safe if a neighboring own chain has enough liberties to rescue it, and with zero liberties we'd need stronger evidence from two neighbors. A single relu over these terms could encode that, but the neighbor-liberty counts need a clipped indicator rather than the raw unbounded evidence values.

I'm designing two separate units for the self-atari check: one that fires when a stone has no liberties of its own and no neighboring own chain has two or more liberties, and a second that handles the zero-neighbor case when no neighboring own chain has three or more.

I'm settling on a penalty made of two separate units: one that fires when a stone has no liberties and no neighboring own chain has two or more, and a second for the zero-liberty case when no own neighbor has three or more. The total would subtract the first, add the second, and cancel both when every neighbor is an opponent stone.

I'm now checking how the self-atari penalty combines its units in the corner case, where a stone has no liberties but touches one own chain with exactly two liberties. The question is whether that case should get a moderate penalty, and whether connecting moves between two-liberty chains would get caught by it too.

I'm weighing whether a neighboring chain with only two liberties can actually rescue a lone stone, since its liberties might include the very point being played. A chain with three or more liberties would settle that, but dropping every capture where the neighbor is weaker would lose real captures of large opponent groups, which are valuable. I'm leaning toward keeping those captures and deciding based on how big the opponent chain is.

I'm going to check whether the real captures in the weak-neighbor case are mostly small chains, since a size-based rule would only need to cover the big ones.

