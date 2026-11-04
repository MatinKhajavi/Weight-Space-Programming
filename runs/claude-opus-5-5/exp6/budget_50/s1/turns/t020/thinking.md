I'm comparing the single-channel H=5 rule against the parity version, which has two channels and scores 813 against 1158 false, a gap of about 345. Dropping to one channel saves roughly 162 units per layer, which could leave room for other features like two-liberty detection or positional influence. I'm also wondering whether a hybrid that brings in some parity signal would work, with rule b evaluated at L2 over adjacent pairs.

I'm weighing a hybrid where both colors get the single-channel rule but only opponent captures get parity channels, which would leave about 107 units spare per layer. That spare could go toward positional and shape signals, since the strong bot wins on shape and not just tactics, and linear features near the move at layer 6 look cheap given the carry already planned there.

I'm considering a 5x5 linear kernel over own and opponent stones near each point, which captures contact and extension shape but not empty-area or between patterns. Those might need nonlinear units in the spare capacity. An influence-style heuristic could also work: score points by a decaying sum of stone colour, favouring contested points near zero and penalising points deep in either territory, using two ReLU units per point.

I'm checking whether the influence units can sit at L6 at all, since they would need the full L6 budget (324 + 162 = 486) and leave no room for the own/opponent carry. That would mean giving up the separate linear own and opponent terms, and the contact count of adjacent opponent stones would no longer be directly available.

I'm considering putting contact-specific units in the L6 slots that would otherwise hold the carry: one for adjacency to opponent stones and one for adjacency to my own stones, each summing over the neighbours of a point.

I'm now trying to pack the linear part of the logits into a single unit per point, since any nonnegative linear feature can be carried that way with the offset folded into the output bias. That leaves seven units per point at L6 (567 total), which is over the 512 budget, so I'm wondering whether the influence units can absorb the linear features instead of keeping them separate.

I'm considering whether the linear part could be folded into the influence pair: if the two thresholds are both zero, the difference of the two ReLUs gives back the influence value as a linear term. That only gives one free linear direction, though. I'm also wondering whether the separate EYE unit could be merged into the safety term to free up another slot.

I'm deciding whether the spare capacity in layers 2 through 5 can help, since any feature computed there would still need a slot in layer 6 to reach the logits. Layer 6 is already full with the current budget, so I'm checking whether the own/opponent carry could be swapped out for something else.

I'm thinking about folding the linear terms into one unit per point, since contact and adjacency could be packed together that way. That would leave around 107 units spare at layer 6, enough for one more per-point feature, perhaps a one-sided influence unit that penalises playing inside one's own territory.

I'm trying to fit an own-territory unit, perhaps a single ReLU over a weighted sum of own and opponent stones, to penalise playing inside one's own strong area. Folding the eye detection into that same unit would free a slot, but an eye would need a large penalty while ordinary own territory only needs a moderate one, and one unit may not separate those cases.

I'm considering switching to the single-channel rule with a readout hop, and fixing how the capture and safety units interact. I'm also thinking about packing the linear terms into one unit per point, and about adding separate own and opponent territory units to test. For the prior, I'm wondering whether sharpening it, by roughly doubling the positional terms, would reduce randomness. Two-liberty detection for atari-making is another idea, though it might need extra channels.

I'm considering whether to count opponent chains with exactly two liberties as atari targets, which would need a separate three-liberty evidence channel for both colours. The catch is that a large chain can look like it has only two liberties when it actually has more, so that "no three-liberty evidence" signal is unreliable.

I'm checking whether a single unit per point can separate "adjacent to an opponent chain with at least two liberties" from "adjacent to one with three or more," by subtracting the three-liberty evidence from the opponent indicator, so that the remainder marks chains with exactly two liberties.

I'm checking whether the layer-1 budget can fit the extra three-liberty initialisation, since 648 units would exceed 512. I'm wondering if a layer-2 unit computed from the existing S0 and Y' channels could produce the ≥3 signal instead.

I'm trying to simplify the ≥3-liberty rule to just "a stone with at least three liberties", since the layer-2 combined threshold would cancel the other evidence terms.

Still working through the tactical side, and I'm weighing whether a rule that forms evidence from adjacent stones with at least three liberties would be cleaner than the separate combined thresholds. Then I'll test the revised version with the S-channel and see how it compares to the stronger baseline before refining further.

I'm thinking about whether parity evidence can go into the layer-2 initialisation, with separate even and odd channels per colour, so that the layer-1 budget still fits.

I'm working out how to make the first-layer evidence robust: clipped indicators for even and odd parity, combined at the next layer so the neighbourhood check becomes a single threshold.

I'm settling on the single-channel stone-evidence rule with a one-hop readout, which leaves spare capacity in layers 2 through 5. The first layer has little room left, but the later layers can carry the extra features. My plan is to add the linear-feature unit at layer 6 and then look at two-liberty tactical signals next.

I'm weighing whether a few more concrete tactical features are worth their budget cost, but I can't easily count liberties with what I have, so I'm leaning toward a positional evaluation instead. I'll look at a few more strong-game losses at plies 20 to 40 to see whether the problems are tactical or positional.



---

The classifier flagged something earlier, though I can't tell what it was, so I'll keep going with the task and avoid repeating that content. Next I'll build the improved version, v2, directly instead of inspecting the setup further. Its first layer keeps the v1 layout: OWN and OPP spans, plus S0W, S0P, YW, and YP.

Still laying out the v2 layers, with the L6 readout, the spare units, and the scaling choices for the output parameters.

Now I'm writing out the v2 layer 6 layout: the output-readout units for the captures and the escape checks, plus the position-scoring and linear units, with the spare slots at the top.

