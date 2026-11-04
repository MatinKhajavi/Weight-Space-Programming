I'm checking the move at (8,4) for black: it's adjacent to the large white group, and it would leave our (8,5) group with only one liberty at (8,6). The estimate is 0.94 despite the self-atari penalty, so I'm working out that penalty: no captured stones, one non-atari stone, which comes to -15.

I'm working through the endgame: every other option looks worse than the self-atari move, so even a slightly bad move could end up preferred over passing. The fix I'm considering is making self-atari detection decisive, since a merged group left with one liberty should be strongly negative unless it captures something real.

I'm weighing whether to bring back the opponent-liberty channel for self-atari safety, since I dropped it to make room for the small-group channel. Adding it back would need more budget than the spare slots in the later layers can provide, so the real choice is between it and the opponent-atari channel, and I lean toward keeping self-atari safety.

I'm working out a way to get approximate binary behavior from a single graded channel: a relu of (1024 times the opponent count minus group mass), which is zero for groups with seeds and roughly 0.88 to 1 otherwise. That would let the value channel also serve as the any-atari signal, and it could free up the SX channel so I can keep self-atari safety.

I'm checking that the value channel stays precise at each layer: with a multiplier of 1024, the summed values reach about 542k at layer 3 and 2.7M at layer 4, so the gating thresholds of 2^20 and 2^23 still separate the cases cleanly.

I'm checking whether the bf16 rounding matters here: values under 256 stay exact, so the no-seed groups are fine, and the gating weights are powers of two, so those are exact too.

I'm checking whether the gating works for non-opponent groups, since a neighbor's large accumulated value could push the sum past the threshold. Bounding the worst case with the multiplier of 1024, the values from the neighbors could reach about 10.8M against a threshold near 8.4M, so I need to recompute the limits more carefully.

I'm bounding the value channel at layer 3 and layer 4 for non-opponent groups: the summed values reach about 563k and then about 2.8M, so the gating bound still has to hold against those larger sums.

I'm settling on keeping the value channel at a multiplier of 1024 so it can replace the spare-slot channel, which would free room to add back the own-group seed count and keep self-atari safety.

I'm checking whether dropping the small-group channel is acceptable, since it's only needed to exempt small groups from the self-atari penalty. Another option is to make the any-atari channel graded, so that single stones and small groups score high while large groups score low, which would make it a stand-in for the small-group signal.

I'm weighing dropping the opponent-low-liberty channel instead, which would free the slots needed to keep the self-atari check and the small-group exemption. That would leave the value and any-atari channels and the own-group count in the later layers, but it means the attack-related signal goes away.

I'm weighing whether attack-oriented liberty tracking is worth its slots, since it pushes weak bots into atari and yields captures. The capture rate against weak opponents fell sharply after the last changes, so I'm wondering whether the capture bonus or the shape weights are behind it.

I'm checking the single-stone capture case: the self-atari penalty comes out to a tie with the capture bonus, but the small-group exemption zeroes it out, so single captures are fine. That leaves the question of why captures against weaker bots dropped, since game length also shrank from about 112 to 86 moves, and I want to look at how the weak player's losses actually unfold.

