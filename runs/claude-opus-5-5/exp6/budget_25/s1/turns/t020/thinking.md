I'm noticing our play has been passive, with many clumped, solid-looking connections, while the opponent has walled off the bottom. Our stones there eventually died, and early on I was mostly playing next to our own stones.

I'm weighing whether an influence-based evaluation could fix our clumping, since points where both sides hold influence look most valuable. I'm also considering that linear filters over the whole board are cheap, though a purely linear one can't capture how contested a point is, which might need nonlinear units.

I'm working out an influence-style evaluation where each stone radiates influence that decays with distance, and I'm considering a nonlinear term that rewards points where the two sides' influence roughly cancels out. The catch is that empty regions far from any stones also read as zero, so I'd need a term for stones nearby to separate contested points from empty space.

I'm now weighing whether the nonlinear influence features could fit within the layer's limited spare units, and I'm checking whether merging the eye-detection unit into the self-atari unit would work. It wouldn't: a point surrounded by a well-liberty'd group wouldn't get flagged, so the eye unit has to stay separate, though I'm still considering whether the interior-territory unit could absorb it.

I'm now looking at whether the early layers, which have spare units, could hold a contestedness feature, though carrying it forward through the later layers would eat the budget that's already tight.

I'm weighing whether dropping the ≥3-liberty channel would free enough budget for shape-based atari detection and influence features, since atari pressure seems valuable against weaker opponents. I'm also thinking about whether carrying own/opp maps more cheaply, through a combined gating channel, could save the rest.

I'm weighing a color-specific gating scheme, where a shared stone channel would leak safety signals across opponent stones between separate own groups. The fix I'm considering is subtracting an opponent-stone indicator so the own-side signal doesn't pass through them.

I'm considering whether the stone channels could be merged into one graded channel, where a value distinguishes no evidence, weak evidence of two or more liberties, and strong evidence of three or more, so I don't need separate chains for each.

I'm trying to keep the values from one of the two evidence levels from being mistaken for the other after propagation. The plan is to seed the "≥2 liberties" evidence with a small clamped value and the "≥3 liberties" evidence with a much larger one, so the sums across a few hops still stay well below the threshold.

I'm working out how to encode the three-way evidence in one graded channel: a small seed for two or more liberties and a large one for three or more, so that after a few hops the sums still separate cleanly. Then I'd derive the thresholds with a pair of units rather than a single ratio, since a ratio lands on fractional values that aren't binary.

I'm now working out how to get a binary "fewer than three liberties" signal from the summed graded values, since a plain ratio would give fractional results. I'm considering a scaled threshold, and whether two units, one for each side of the cutoff, could produce a clean 0/1 output.

I'm checking whether a single graded channel can carry both liberty levels: small seeds for two or more liberties and a large one for three or more, so propagated sums still separate. Turning that sum into a clean binary signal would take two relu units rather than one ratio.

I'm trying to merge the two evidence levels into one channel by not clamping the seeds: the two-liberty seed is small, the three-liberty seed is multiplied by a large constant, so the propagated sum stays under the threshold unless three-liberty evidence is present. I'm checking that the bounds hold after a few hops, and then I'd read atari status off that combined value.

I'm checking whether the propagated values stay exact enough at the thresholds, since the sums grow large and I'd be relying on floating-point precision holding up. The gap between the two-liberty and three-liberty ranges looks wide enough at fp32 and bf16 precision, so the threshold should still separate them cleanly.

I'm checking whether a single merged channel could drop the parity-style chains, but a clamped binary "three or more" flag would take two units per point per color, which blows the budget. So I'm considering keeping the three-liberty value unclamped and making the layer-six units tolerate that.

I'm checking whether the three-liberty signal can stay unclamped, since a clamp would cost two units per point. The worry is that an unclamped value could make the save and attack units fire by mistake, so I'm checking whether the attack unit still stays off when a three-liberty group sits next to the target.

I'm now checking whether the three-liberty signal can be an unclamped relu of the summed merged channel, so that layer five stays within budget at 488 units. That would free up the 162 units per layer saved in layers two through four, and I'm weighing what to spend them on, since any influence features computed earlier would have to be carried through layer five to reach layer six.

I'm checking how many units the later layers would need if the evidence features were carried forward. Layer five has about 24 spare units, but it's the real bottleneck, since the own/opp maps it carries are needed by every feature at layer six. Moving the eye and influence features earlier wouldn't save anything, because they'd just have to be carried through layer five instead.

I'm trying to fold the eye check into the influence unit, so one relu covers both "inside own territory" and the no-two-eyes condition. That would save a unit per point if the linear combination holds up.

I'm folding the eye check into the same unit as the influence term, weighting the opponent-within-two count so that an eye still trips the penalty while ordinary interior points only do when there's no nearby opposing stone.

I'm weighing whether a bigger gain would come from strategy than from more tactical fixes, since the losses look like repetition and large group deaths. I'm planning around a budget of about 23 passes: merge the atari channels, add shape seeds at the first layer, and then test the result in a games run.

I'm settling on the layer-five budget: the three-liberty signal can stay unclamped, which keeps the atari and "three" channels at 488 units. That still leaves L5 with no room for extra features, and the L6 layer is nearly full as well.

I'm moving the features that don't need atari information into layer five, since they only need own and opponent maps from layer four. Their outputs can then be folded linearly into the layer-six LIN unit, so they cost nothing extra at layer six.

I'm weighing a trade-off: dropping the three-liberty channel would free room in layer five for the eye and influence features, but I'd lose the own-side and attack signals that depend on it. Attacking an opponent group that sits next to a three-liberty group would be suppressed, which seems an acceptable cost against weaker bots. I'm also wondering whether there's a cheaper way to get that three-liberty signal, but both places that use it already need it.

I'm considering folding the SAVE and ATT units into a single "urgent tactical" unit, with the coefficients inside adjusted so they still behave differently on the output side.

I'm now weighing whether to spend the remaining budget on tactics or strategy. I'm leaning toward improving atari detection first, then shaping the opening and the influence weights in the existing LIN patterns, which don't need extra units.

I'm thinking about whether a more principled linear evaluation could help, weighting nearby opponent stones more than my own and letting the influence decay with distance. I'm also planning to check the game records for where territory was lost, since walls and clumped stones seem to be the main problem.

I'm writing the value of each candidate move as a kernel-weighted sum over board points, where each point's contribution depends on how contested its current influence is.

I'm considering whether the contestedness term could be computed at layer one from the raw input, since the influence values are linear there. That would cost about 243 units, which is more than the 186 spare units I have at that layer.

I'm considering whether to merge the atari and three-liberty signals into one graded unit per stone, which would free room in layer five. The idea is to compute the contestedness value at an earlier layer and carry only the summed result forward.

I'm checking whether dropping the opponent-three-liberty channel would free enough room in layer five, giving up some own-side and attack signals to make space for an influence value that gets carried forward.

I'm considering dropping the own-side three-liberty channel from layer five and replacing it with a plain own-minus-atari signal, which would leave room for the influence value and keep the total at 488 units. I'm also looking at spending the layer-one spare units on per-point influence bumps, around two units per point, to set up that value.

I'm checking whether a triangle-shaped bump on influence can be built from fewer units, but a bump needs three breakpoints, so two units won't do it. I'm also considering an asymmetric version, since what matters is points where influence is near zero yet the total influence is still meaningful, which rules out simply counting empty regions.

I'm checking whether the contestedness term can be built from the influence sums in two units per point, since three per point won't fit in the spare room at layer two. I'm trying a min-style combination of influence and total influence, but it isn't yet clear it can be expressed linearly with the relu units available.

I'm seeing that the contestedness term can be written as T minus the absolute value of I, and since T always bounds |I|, that quantity is never negative. Expressing it through a single relu of negative I would make the whole value linear in own, opponent, and that one extra unit, so it could fold straight into the LIN unit.

I'm noting that the negative part of each influence value can be captured with one relu per point at layer five, so the whole contestedness term becomes linear in those units and can be folded into the LIN unit. Since the total influence always bounds the absolute difference, that term works out to twice the smaller of the two influence sums, which is the quantity I want.

I'm trying to get the budget for layer five to fit: the atari and three-liberty units, the carried values, and the per-point contested and empty-region units already come to more than the 488 available. Dropping the emptiness unit would help, but I'm still about 57 units short.

I'm checking whether the per-point feature can instead go at layer six, since the output layer sums over all of those units with kernel weights anyway. A relu of the negative influence at that layer, combined with the total and influence sums, would keep the value linear and avoid spending the layer-five budget.

I'm checking whether layer six can take the nonlinear influence features, since it only needs own and opponent stones, and the budget there is over by about 160 units. I could merge the save and attack units or drop the influence channel to make room, though I'm not sure which trade-off is worth it. Given that each test pass only gives about a 4% standard error on winrate, I think the tactical fix should come first.

Now I'm sketching out the trunk layout for the next version: block-seed inputs in the first layer, followed by a combined seed channel per color in the second layer, which should keep the tactical fix ahead of any strategy tuning.

Working out the seed units for the block-based input: a stone's seed value depends on its own evidence plus its neighbours', and I'm thinking of block-level seeds that use the count of external liberties, since those points don't overlap between stones in a 2x2 block.

I'm working out how to fit the block-seed and per-stone seed units into the layer budget. Two seed units per color at layer two leave only about 24 spare units, while the merged channel at layers three and four frees roughly 186 each, so the merge will have to happen there.

I'm now considering whether a 2x2 block of same-colored stones, like the bent pair in game 34 with an occupied corner, could be picked out with block seeds at layer one instead of straight-triple seeds.

I'm now checking whether the bent pair in game 34 and the large white group in game 46 can both be caught by 2x2 block seeds, since each block's external liberties are distinct points I can count directly.

I'm now checking whether a straight triple with an empty middle point can be caught the same way, by pairing each end stone's liberty count with its neighbours'.

I'm now testing whether a triple of same-colored stones along an edge, where the middle stone has no liberties, can be caught by counting the liberties of the two end stones. That would need about 252 units at layer one, which doesn't fit, so I'm looking at computing it at layer two from the neighbouring liberty counts instead.

I'm weighing whether to fold a corner-occupancy check into the seed units, so that a bent pair with the corner taken by the opponent could be caught without spending extra units on separate triple seeds per axis.

I'm going to add the block-based seeds now, since they're cheap, and leave the rest of the layout as it is. Then I'll run a test pass against the strong version to see whether the repetition and losses shift. I'm also tempted to adjust the LIN weights at the same time to reduce clumping, though that would make it harder to attribute any change to a single cause.

I'm weighing a spacing term in the linear evaluation: a small penalty for stones packed within radius two of the same color, and a small bonus for one-space jumps from an existing stone. Since a term can't depend on whether an opponent is nearby, I'm still working out how to make the adjacency preference hold without that condition.

I'm sketching the block-seed units at layer one now, one for each 2x2 position, gated so they only fire when all four stones are the same color.

I'm sketching the seed term for each 2x2 block: each block contributes to the liberty counts of the stones inside it, but the block-level evidence only reaches a threshold of two or three liberties. I'm working out the offsets so that a block with two liberties still registers, but a single-liberty block doesn't push the seed value on its own.

I'm checking whether a single block-level unit can carry the liberty threshold for three liberties, since the linear weight would push a two-liberty block into a false count. A plain relu offset doesn't separate two from three cleanly, and a separate unit per block would cost 128 more slots at layer one, which I don't have.

Now I'm settling the gate constant for the layer-four unit, since bf16 can't represent 20000 exactly and a rounded value could let non-own stones pass the gate, so I'm switching to 16384.

Checking the gate constants: the non-own case stays negative even with the block terms, so the L2 gate of 20 holds. I'm making sure the L3 and L4 thresholds are exact in bf16, and the L1 block threshold is what I need to settle next.

Next I'm setting the gate constant for the layer-two block unit, adding offsets so it stays off for non-own points, and planning a v2 run with block seeds against the strong and weak opponents.

