I'm checking the cases where a capture happens at a tight spot, so the resulting chain ends up with fewer than two liberties. A single captured stone leaves just one liberty, which looks like a snapback-type situation, even though an adjacent opposing chain is large.

I'm seeing that the self-atari moves without captures are far more common than the captures themselves, roughly four per game, and I'm wondering why the penalty isn't stopping them. The escape bonus for extending a chain in atari into a dead end seems to outweigh the self-atari penalty, so the self-atari weight probably needs to be raised.

I'm weighing whether raising the self-atari weight to 12 would fix the escape-into-dead-end case, since the net would turn negative. But captures at zero empty neighbors would then get penalized more than they gain, so I'm considering a separate override for captures, which runs into a limit on the L5 block space.

I'm considering moving the opponent parity propagation out of L5 and computing the atari readout at L5 instead, which would free up room but shrink the capture detection range from five hops to four. I'm also weighing an extra L5 block for that readout, bringing the total to 567.

I'm considering dropping the own-chain S3 check and using the plain two-liberty test for self-atari, but a lone own stone with two liberties, one of them the move point, would then pass as safe and miss exactly the corner case. Dropping W/O isn't the fix, since the L5 budget is the real constraint.

I'm weighing whether the self-atari weight can be handled without a dedicated capture override, since the L5 block budget is already full. One idea is to push the own-chain liberty readout up to L5 so the own-side propagation stops at L4, but adding that readout block would still take the total past the limit.

I'm thinking about adding a term to the self-atari unit that penalizes a move when any adjacent opposing chain lacks evidence of capture, using a linear combination of the opponent-parity and evidence values from L5.

I'm testing whether a self-atari weight alone can block the case where an adjacent opposing chain is in atari, since the value is pushed negative. The trouble is that a safe opposing neighbor would contribute a large positive term and make the self-atari unit fire when it shouldn't.

I'm weighing whether the self-atari penalty can be overridden for captures without a dedicated L5 block. One option is to push the capture-related output into the final layer as a product term, but any linear combination of the per-neighbor values would let a safe opposing chain cancel the atari signal.

I'm considering a lower self-atari weight of 7, but then escaping into a dead end would still net +2, so I'd need to shrink the escape bonus as well.

I'm considering whether the carry cost of the opponent and own-side propagation can be reduced by letting the channels handle the masks themselves through an offset trick.

I'm trying to make the self-atari masks work by having a channel carry a value that's slightly above one on own stones, so the mask can be read off linearly with a small error, and I'm checking whether that error stays within budget.

I'm trying to keep the variance contamination from neighbor counts out of the threshold test, since a tiny evidence offset could still leak through the sum.

I'm weighing whether the carry can be cut in half by using one combined channel, C = relu(1 + own − opp), instead of separate own and opp channels at layers 2 through 5, since each mask could still be recovered with a scaled negative term.

I'm checking whether a per-neighbor capture output can be built from a single combined channel, since the empty-neighbor count and the eye test both need a clean empty indicator that a nonlinear absolute value won't give me.

I'm working out whether the sign of the self-atari term can be recovered from a single carried channel, since the empty-neighbor count is the piece that doesn't come out linear.

I'm checking whether SA could be computed at L5 instead, using the L4 carry for the neighbor and opponent-atari information, so that the override is applied one layer earlier.

I'm checking whether single-stone atari can be detected early at L1 and carried forward, so the self-atari unit can see captures without another override block. The budget is the problem: at L5 there are too many channels to carry, even if I swap in a combined channel at the earlier layers.

I'm trying to see whether the self-atari check can rely on a single empty-neighbor count, since that would need an empty-point indicator I don't have. One option is to phrase the test as occupied neighbors reaching deg − 1, but that still needs a nonlinear absolute value. So I'm considering a different split: take single-stone captures through a dedicated atari channel carried from layer 1, and handle multi-stone captures separately. That would push the L1 block count past the limit, though.

Checking whether the S3 init for the W side can move to L2 by going through the S0 channel with one more relu, since e isn't linear in C at L1. That would delay S3 by a layer and leave L1 with room, but it would also shorten how far S3 can propagate before L5.

I'm checking whether the S3 init can be pushed to layer 2 through the S0 channel, which would free a slot in layer 1 for the single-stone atari channel. The cost is a one-layer delay on S3, and the neighbor-liberty rule that needs the layer-1 info would be lost.

I'm going around in circles on the self-atari weight, so I'll step back and focus on what matters most: stopping self-atari blunders while still allowing captures. The counts show that non-capturing self-ataris are common, so the override can't be a blanket exemption. I'm wondering whether the capture bonus could live inside the atari-response unit itself, so the self-atari penalty is only cancelled when that capture signal is reliable.

I'm weighing whether to just set the capture bonus well above twice the self-atari penalty, since big-chain captures are the unreliable part. I'm comparing how many large captures are real against how many are false positives, and wondering whether the decisive ones are worth keeping. I'll look at the repetition and big-capture stats from the last pass against the earlier version to see why it underperformed.

I'm comparing the repetition rates and strong scores across the three passes, which shows the ε change didn't reduce repetition. Next I'll look at the trigger moves' probabilities and chain sizes in pass_9.

