The score is 0.34, still well below v9's 0.54, with 28 captures, so LIN alone doesn't explain it. I'm comparing v13 and v9: v9 has OT and eight corner units, while v13 drops OT and adds 56 edge-cycle units. Either OT matters a lot, or those cycle units are penalizing too many legitimate moves, and the noise explanation seems unlikely

The gap between v13 and v9 is still too big to put down to noise, and the cycle units alone don't seem to account for it either. Captures fell from 43 to 28 per game, so something about the fighting changed. I suspect OT's own-territory penalty was pushing moves toward contested areas, and without it the moves inside my own territory get wasted while the opponent gains ground.

Since v11 also lacked OT but scored 0.49, the missing own-territory penalty doesn't fully explain v13's drop to 0.34. The gap between v11 and v13 points to either the A2W bonus being valuable or the 56 cycle units hurting. I'm going to check whether the cycle units fire in ordinary positions where the move is actually good, and I'm also wondering whether a block placement in the L6_OT range might be misaligned.

I'm checking whether the cycle units in the L6_OT block, which span indices 405 to 460, could be overlapping with weights written elsewhere, since the OT code was removed earlier and I want to confirm nothing else is writing into that range.

I'm checking whether the A2W bonus matters more than I assumed, since v11 with it scored 0.49 while v13 without it sits at 0.34. Comparing across the versions, the pattern doesn't line up cleanly with either OT or the defense bonus alone, so I still can't tell whether the cycle units are the cause.

I'm estimating how often the cycle-unit penalty fires on a capture that was actually available, using the pass_18 games. The idea is to compare the positions where the pattern appears against the moves the net actually made, to see how much game-deciding capturing it might be suppressing.

I'm weighing whether to skip the recapture at B, since the 52 legitimate recaptures are a much larger group than the 18 bad ones, and the throw-in stone at C would just stay in atari for a later turn unless the opponent connects at B.

I'm weighing whether to penalize the B recapture moderately instead of blocking it outright, since it's only a single-stone throw-in and the cycle-unit penalty could be costing more than these small recaptures.

I'm settling on the plan now: the A-penalty units go, and the penalty moves to the B recapture after the opponent's C throw-in, since the cycle depends on the opponent always capturing at A and B.

I'm weighing whether the opponent's throw-in at C is worth taking, since 52 legitimate recaptures outnumber the 18 bad ones. Penalizing the B recapture rather than the self-atari at A might preserve the big captures while still breaking the cycle.

I'm weighing the repetition numbers against each other: v9 had only 4 repetition losses per 128 games, v11 had 13, and that gap alone could explain part of v11's lower score. Still, the generalized cycle units might be blocking big captures, so I want a cleaner comparison before deciding anything. I'll rerun v9 to get a fresh baseline.

I'm weighing whether to build a bigger version that keeps OT, but the parameter budget in layer 6 is tight: the existing blocks already fill most of it, so 56 extra units won't fit alongside them. I'm now considering whether the EYE block, which covers only a few rare eye points, could absorb part of the OT computation to free up room.

I'm trying to fold the eye penalty into OT itself: weight the direct neighbours much more heavily, so an eye point saturates to a large value while ordinary territory stays small. That would free the EYE block, but I still need to check whether the ring terms disturb that separation.

I'm considering whether to fold the LIN term into the OT unit, since OT only has a nonzero output when its pre-activation is positive, so LIN could be absorbed there. I'd need to check that the merged unit still behaves correctly in the cases where OT is negative.

I'm thinking about folding the LIN term into another unit, but it's not clear that's possible without losing its effect whenever that unit's output is zero. Looking at the AP and AW units, they don't seem to offer a clean way to absorb LIN either. I'm also checking whether the carry-over at layer 6 could free up room, but that doesn't look promising since none of those units carry anything forward.

I'm weighing a cheaper option: penalize the B recapture after the opponent's C throw-in instead of blocking the self-atari at A. Next I want to check whether the captured-chain flags at A can tell real captures from false ones.

I'm wondering whether the long-range evidence idea could help with the false-capture flags on edge chains, but that's speculative. The more solid option is to accept the repetition rate of roughly 4 to 7 per 128 games that v9 had, since v9 still scored best at 0.54. I want to rerun v9 first to confirm that result isn't noise before trying anything else.

I'm going to keep v9 as the baseline, since every later variant scored lower, and I'll look for gains in the positional features like contact and the OT unit rather than fighting the cycle problem directly.

I'm planning a set of short A/B tests, each at 128 games, starting with a v9 baseline re-run to calibrate the noise. The list covers stronger contact, a stronger OT weight, and some LIN variations, though the 18 spare units in layer 6 leave little room to add anything.

