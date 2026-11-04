I'm looking at the 3-cycle case, where we passed at about 0.8 but the self-atari move still got sampled at 0.15 to 0.2. Its logit sits only about 1.3 below pass, so at temperature 1 that's too risky, and I want it at least 5 below. Making the network more decisive on clear positions could widen that gap.

I'm checking game 4, where the white group has two liberties and our self-atari move gets a positive attack term from the network, which is probably why its logit sits close to pass.

I'm working out the threshold for the attack term so it only fires when the move leaves our group with at least two liberties, and I'm checking whether a modest value can satisfy both cases.

I'm settling on a threshold of 2.5 for the attack term with a weight of 1, which gives a small positive value when the white group has one liberty left and our move leaves us two, and no value otherwise. Checking the cases: one-liberty groups still score a little, but a move that would leave us in atari does not get the full reward.

I'm weighing a simpler fix: the attack term's reward is too strong relative to the self-atari penalty, so I'm considering letting the self-atari penalty dominate instead of reshaping the threshold itself.

I'm considering a different attack term where the opposing-stone count is weighted differently, but it can still fire when the group has a single liberty, so I'm checking whether making the self-atari penalty dominate is a cleaner route.

I'm raising the self-atari penalty to -20 so it outweighs the attack reward, which tops out around 14 against a self-atari cost of about 12 at the current weight. The same change makes lone one-liberty stones and moves with no own support or capture cost roughly -30, which is acceptable since those moves are bad anyway. I'm also checking that the capture compensation for large lone captures still balances against the stronger penalty.

I'm weighing whether to reshape the attack term so it can't reward a move that leaves our own group in atari, or to make the self-atari penalty large enough to override it. Raising the penalty weight is simpler but it has side effects on the capture compensation, which is fragile to adjust.

I'm trying a simpler attack term that scales the opposing-stone count and the own-liberty count together, so it only gives a small reward when our group keeps at least two liberties. Checking the cases shows it still leaks a little reward for opposing groups with one liberty, and for open points it can fire too, so the threshold needs more care.

I'm looking at whether the attack term could be scaled down for open points and single-opponent cases, but capping it linearly doesn't cleanly separate the two-liberty and one-liberty situations. I'm considering raising the self-atari penalty weight to -20 so it dominates the attack reward, and then checking how the capture compensation for large lone captures would shift as a result.

I'm weighing whether to rework the self-atari term by separating own-stone and opponent-stone atari counts, since the current U1 adjustment leaks extra reward into the penalty. Subtracting out the opponent contribution this way doesn't come out clean, so I'm looking for a cleaner separation instead.

I'm trying to separate the own-stone and opponent-stone atari counts in the self-atari term, since the U1 mixing is what's leaking reward. Replacing the SAFE3 check with the non-atari own-neighbor count would match the earlier approach, but that alone leaves the same false penalties from big captures, so I'm considering a separate AO and AX pair instead.

I'm thinking about whether to drive the pass-versus-move gap through a global sharpening of the policy instead of retuning each term, since the 3-cycle endgame positions are where passing is correct. Raising the temperature scale from 1.5 to 2.5 would cut the chance of sampling that near-pass self-atari move down a lot, though it would also sharpen opening choices.

I'm swapping the self-atari weight to -16 and bumping the capture compensation to match, then folding the eye and interior terms into a single territory slot with a new ko slot after it.

I'm working through the terrain term now: the eye penalty comes out around -38 when there are no opposing neighbours and about -14 when there are four, both still below pass. The interior case with no opposing neighbours lands near -4, and I'm checking whether the adjusted weight of -2.5 keeps it in the same range as before.

I'm tuning the scale to 2.0 and setting the self-atari weight to -16 with the capture compensation at 55, and I'm folding the ko-shape exclusion into the capture term so single-stone captures don't get the bonus.

