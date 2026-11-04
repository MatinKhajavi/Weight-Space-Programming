I'm weighing the candidate moves in v5: the (2,0) cap scores about 11.2 after the territory and shape terms are netted out, which clears the pass value of −8, so it still wins even with the reduced flag.

I'm checking the parity mass around (2,1): the XE values at the neighbors sum to about 0.53, which gives the 0.47 AP. The evidence at (4,1) only arrives at layer 5, one hop beyond what the readout can reach, so I'm considering whether the readout should look two hops out.

I'm considering a readout that sums over the radius-2 diamond, gating the distance-2 terms on the middle cell being opponent. The risk is that an opponent stone two steps away can belong to a different chain, which would hide real atari.

I'm checking whether a gated readout over the distance-2 cells would avoid leaking evidence from a different chain, but the diagonal and empty-middle cases are common enough that the leak looks like a real risk. A single relu can't express "subtract only when the middle is opponent," since the gating term pushes the score the wrong way. So I'm turning to a second readout unit for the extended range instead.

I'm now looking at a two-hop version of the propagation, where evidence from a neighbor two steps away only passes through when the middle cell is opponent. The per-path gating would need one unit for each triple of source, middle, and target, which gets expensive quickly, so I'm wondering whether the evidence could instead be stored on the edges themselves.

I'm working out whether the readout could reach two hops by letting layer 1 already include one propagation step, so that evidence from an even-parity neighbor is folded in early for each odd cell.

I'm checking whether a single relu over the neighbor sum can encode "only count empties beside opponent stones," since any non-opponent neighbor would have to knock the whole term down.

I'm checking whether one layer's propagation could be folded into the first layer, but a per-direction unit for every point would fill the whole layer. So I'm weighing a different route: since (2,0) still wins over pass at this point, the real question is what passing early would cost in the wider game.

Passing at −8 is still far below the first-line moves, which sit around −1.3 once the contact terms are added. Passing too early would mean leaving dame unfilled on the edge, and each dame is a point lost in area scoring, so the pass value can't simply be nudged upward. The repetition-trigger moves are the real problem: they get SA = 2 and then a flagged capture, so raising the self-atari scale with the number of own stones put into atari is the direction I'm weighing next.

I'm weighing whether to accept the roughly 10% of repetition losses and look for gains in positional play instead. I'm also considering not recapturing at (1,0) after the opponent takes at (0,0), since the leftover single stone is not an immediate loss.

I'm considering a ko-style check instead: if we play a self-atari pair, the opponent captures both, and our recapture would recreate an earlier board, then the repetition is the cycle to flag.

I'm checking the cycle the other way round: if we play A and the opponent captures both A and B at C, a recapture at B would only restore the earlier board when C's other neighbours are already ours or the edge. In the corner that makes C=(0,0) the enclosed point, so a self-atari at A could be flagged as the start of a repetition.

I'm turning to the repetition cases now, aiming for a local detector that flags the corner cycle where our self-atari stone can be captured and then retaken to restore the earlier board. I'll set aside the broader evaluation work for the moment and check whether that targeted fix is cheap enough to try.

I'm narrowing the repetition detector to the corner pattern: a single stone of ours on the edge next to the corner, with the corner point empty and our other corner-neighbor already in place. I'm checking whether the empty point beside the candidate move leaves it with only those two liberties.

I'm considering a set of eight detector units for the corner cases, each an AND over six local conditions, and I want to place them in the spare capacity of layer 6, since they only need the own and opponent features already carried from layer 5. The output would be a large signal flagging the repetition pattern.

I'm checking whether a corner detector would wrongly block a real capture, since if the opponent chain only has A as its last liberty, playing there is the right move. The recurring corner shape in the losses still points to the cycle as the main failure, so I'm weighing that against the cost of missing a true atari.

I'm adding eight corner-pattern detector units to layer 6, each an AND over the own and opponent features at the corner, its two adjacent edge points, and the diagonal and outer neighbours. When the pattern matches, the detector would apply a penalty of about −30 at the self-atari point, which would stop the repetition cycle without touching most real captures.

I'm checking that the corner pattern holds up: after playing A, the chain {A, B} would still have two liberties, so there's no immediate self-atari. I'm also confirming that B's remaining liberties are only C and A once the diagonal neighbour is opponent. That gives eight corner-configuration units, and I'm setting aside the non-corner enclosure cases as too rare to be worth it. Next I'm looking at restoring the v6 base settings.

I'm settling on the v4 base, since its slightly higher observed win rate outweighs the small gap with v6, and I'm keeping its OT and LIN features while adding the corner detector units on top.

