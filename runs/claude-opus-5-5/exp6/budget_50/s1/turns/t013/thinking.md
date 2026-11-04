Many captures are false, mostly on large chains of 5 to 25 stones with only 2 to 5 liberties, and about 332 of roughly 7000 moves are flagged as confident false captures or escapes. Since the big chains are where the trouble is, I need better liberty evidence for them, and I'm considering whether a rule that looks for two stones of the same chain with distinct liberties in the local window would help, or whether the real fix is more propagation hops.

I'm weighing whether to widen each propagation layer to radius-2 so evidence covers more ground per hop, but that risks leaking evidence between separate same-color chains across an empty point or a diagonal, which would produce false "safe" verdicts. I'm considering gating the jumps through the middle point to prevent that.

I'm working out whether gating a two-step jump through a middle point would help, but since the middle has to be own anyway, that just reduces to two ordinary hops, so there's no speedup without per-edge units. I'm also thinking about whether a non-local "at least two liberties" rule could give sound evidence for large chains.

I'm considering whether two stones of the same chain sitting at distance three or more must have distinct liberties, which would give sound evidence of at least two liberties without needing extra propagation hops. The open question is how to track which lib-having stones came from different places, and I'm thinking about carrying the signals in separate direction or residue-class channels so they can be told apart.

I'm considering whether to tag each liberty with a class, so the classes can be OR-propagated across the chain and an atari shows up as exactly one class present, though two liberties could still share a point and collide.

I'm working out whether a parity split could give sound evidence for large chains: a single stone's liberties all share one color parity, so a chain that has lib-having stones of both parities must have at least two distinct liberties.

I'm now checking whether the even and odd parity channels could be cleanly combined: each would need to be clipped to a 0/1 indicator before the AND, which costs a couple of extra units per step.

I'm now checking which evidence types are complete: a chain has at least two liberties exactly when two of its lib-having stones have distinct liberties, and a chain with only one liberty has all its lib-having stones clustered around that point, so two such stones more than two apart would already prove at least two liberties.

I'm now looking at coarse binning of rows, aiming to detect whether a chain's lib-having stones span a row range of three or more, which would prove at least two liberties.

I'm considering whether to drop the bin-based approach and instead accept heuristics that are strong but not provably sound, perhaps requiring a local group size threshold together with some liberty evidence. I'm also thinking about switching to a local check: verifying atari directly around the liberty point p, examining whether every liberty of the adjacent chain is covered.

I'm checking whether a liberty can be confirmed locally around the point p, by seeing whether a neighboring stone of the chain has an empty point besides p.

I'm now testing whether an atari can be checked from the local side instead: for a liberty p next to a stone q, the chain is in atari only if q has no other empty neighbor and every other stone of the chain is either liberty-free or touches p with exactly one liberty. The difficulty is that a stone of the chain next to p can be far away along the chain's hops, so I'm considering whether comparing the count of liberty-having stones against the local neighborhood of p would be enough.

I'm now testing a binary-feature version of the liberty classes, where each feature splits libs into two values and the chain would need libs with both values present to rule out atari.

I'm trying a version where each stone's liberty gets a binary feature, like the parity of its row or column, and the chain needs liberties with both feature values present to rule out atari. Two libs that match on every feature would still slip through, so I'd need four features, which costs far too many channels. That makes me want to encode the liberties as numeric sums with multiplicity rather than one-hot indicators.

I'm looking at whether min and max propagation of liberty IDs could work: a chain is in atari exactly when its min and max coincide. Propagating max as a + relu(b − a) stays exact when all values are equal, so the test only fails when the chain truly has at least two distinct liberties.

I'm checking whether the min and max of liberty IDs can be propagated exactly: if all of a chain's liberties share one ID, the max and min coincide, so the gap being nonzero proves at least two distinct liberties. To keep unset values from interfering, I'm storing max-based channels with 0 meaning no information, and encoding the min as a max over reversed values.

I'm working through a max-and-min scheme for liberty IDs, where a chain is in atari exactly when the max and min coincide, and the gap between them proves two distinct liberties. I'm checking whether the relu-based propagation keeps that test exact, including how unset values are masked.

I'm checking whether a single combined max value can stand in for the per-edge comparisons, since doing those per directed edge would cost far too many units. The trouble is that each stone would have to take the max over its neighbors while masking out points that aren't its own, so I'm looking at a single-unit approximation that keeps that masking.

I'm considering whether the variance of the liberty IDs could serve as the test, since zero variance means every liberty shares one ID. That would require sums of IDs and their squares, weighted by walk multiplicity, but the division and squaring needed to compute it are nonlinear and would cost many units to approximate with ReLUs.

I'm now checking whether the per-point test can be written as a sum: expanding the squared gap gives three linear quantities, the sums of the squared IDs, of the IDs, and of the weights, and the candidate point's own ID is just a constant.

I'm checking whether, at an empty point p, summing the squared gaps between each neighboring chain's liberty IDs and p's own ID could work as a test: the total is zero only if every liberty in those chains is p itself.

I'm now checking whether a per-point sum over the adjacent opposing chains could detect atari, though a safe neighbor could mask an atari one, so I'm looking at a per-neighbor version instead. Each stone would carry linear sums of its liberty IDs, their squares, and a count, propagated along the chain, which keeps every channel to one unit per point per color.

I'm now looking at a per-stone version, where a stone with exactly one liberty could carry that liberty's ID as a local reference, so each chain is compared against a value it can check without the quadratic cost.

I'm now working out the per-(p,q) version of the test: for each empty point p and each neighbor q, the atari check reduces to a quadratic expression in the chain's propagated sums, with p's ID as a fixed constant. Counting the directed adjacent pairs gives 288 such units per step, and I'm checking how many channels that needs at the previous layer.

I'm weighing a shortcut: a single channel that tests whether the liberties all share one ID, to avoid three separate propagated sums. Using only the first-order sum fails because the terms can cancel, so I'm looking for a fixed ID choice that makes one channel enough.

I'm now looking at the test from a geometric angle: a chain is in atari at p exactly when none of its stones has a liberty other than p, so I can count liberty adjacencies per stone and check whether the chain's total, excluding those touching p, comes to zero.

I'm checking whether the exclusion can be handled directly: the chain-wide count of lib-adjacencies minus those landing on p should be zero exactly when the chain is in atari. The catch is that the propagated sum from q still picks up contributions from the stones next to p, so the subtraction has to be applied per neighbor rather than once over the whole chain.

I'm trying to see whether the walk-weighted sum of each stone's empty neighbors could confirm atari at p, but the weights make the right-hand side hard to pin down. So I'm considering a version that excludes lib-adjacencies pointing toward p.

I'm now checking which stones next to p are actually in the chain, since the diagonal ones might be reached through q's neighbor at q + d⊥ and the straight-across one sits at distance 2.

I'm weighing whether stones of both colors' liberty parities could give a sound two-liberty signal: two stones at odd distance can't share a liberty, and stones at even distance four or more can't either, so lib-stones of mixed parity would prove at least two liberties. I'm considering that as sound evidence, though it would still leave out the cases I'd want to catch.

I'm noting that same-parity lib stones far apart, like (4,2) and (4,6), still slip past the parity check, and that two lib-stones at distance 2 that don't share a liberty would too. So I'm looking for a second binary feature that tells apart lib-stones adjacent to a common empty point, since those have row and column values within a span of two.

I'm testing whether a feature constant across each atari chain's liberty neighborhood can exist, since that would rule out the remaining false positives. Any nonconstant grid feature differs somewhere within some liberty neighborhood, and neighborhoods overlap enough that the constraint chains across the board, so I'm checking whether parity alone is the only sound option.

I'm now looking at a different kind of evidence: lib points themselves, where any binary feature splits them, and a chain with liberties on both sides of that split has at least two. Separating every pair of liberties would take about seven such features, each costing two channels per color per step, and I'm checking whether unclipped sums of lib adjacencies by feature value could make that affordable.

I'm now looking at a feature where lib points split by row parity, f2 = r mod 2, along with (r+c) mod 2, since a stone's libs all share one parity class and the two parities together cover the cases that matter.

I'm now considering alternatives: pairs of distinct liberties that differ by an even amount in both coordinates would slip past the parity features, so I'd add a feature based on halved coordinates. I'm also considering a per-stone test that uses q's single liberty as a reference, so that the chain is atari exactly when every liberty adjacency in it points to that one point.

I'm testing whether a per-direction version works: for each neighbor direction of q, a quadratic check against that liberty's ID would tell me whether every liberty adjacency points to the same point. That would need four units per point per color, which is about 648 in total and too many to be practical, so I'm looking for a cheaper way to encode the reference.

I'm trying a version of the test that is relative to q's own position: since (r_q, c_q) is fixed for each unit, I could sum the squared offsets from q to each chain liberty instead of comparing liberties to each other.

I'm now checking whether a squared-distance sum measured from q's own position can serve as the test: each lib adjacency contributes its squared distance to q, minus one, so the total is zero only when every liberty sits next to q. Combined with q having exactly one empty neighbour, that would pin the chain's only liberty to q's, which is the atari condition.

I'm defining a per-stone test value from the propagated sums, so that it is zero exactly when every liberty sits next to the stone. I still need to gate it on the stone having exactly one empty neighbour, and I'm weighing how to encode that count with a single relu.

I'm checking the penalty for stones with two or more empty neighbours, and the gating term that keeps the test at exactly one liberty. I'm also wondering whether the quadratic test Z_q still needs a separate indicator for the stone's colour.

I'm checking whether this range-limited test really beats the earlier parity and pairwise evidence: a lib outside the propagation range could still leave a false atari, but any lib within range that isn't adjacent to q would already rule it out, so the test looks stricter.

I'm checking whether a single quadratic sum relative to q's position could stand in for all four channels, since the coefficients depend only on q's coordinates and the total would still need four propagated quantities.

I'm weighing a 1D id encoding as a cheaper way to express the distance test, but the neighbour differences don't come out as a clean threshold. I'm now considering whether splitting the requirement into separate row and column checks could make the all-libs-equal condition cheaper to encode.

I'm checking the four-term version: the constant term multiplies the count, so I'd need four channels, A2, A1r, A1c and A0, per colour. That's 648 channels on top of the 162 carry channels, which comes to 810, more than the 512 budget, so this route doesn't fit.

I'm now checking a variant that measures distances to a candidate empty point p rather than to the stone q, since the squared-distance sum over libs of the chain would again need several separate propagated statistics.

I'm trying a modular-id version instead: a cosine-and-sine encoding of each liberty's id would show whether all liberties share one residue, using about three channels rather than four. The signed terms would need shifting to stay nonnegative under relu, and I'm still choosing the modulus so that distinct points don't collide.

I'm checking whether a cosine embedding of each liberty's position could work as the test, since mapping every point to its own angle means matching angles would mean matching points. The sensitivity looks small, though, so I'm wondering whether float32 precision would hold for the deviations I need to detect.

I'm trying to cut the channel count, since the per-color statistics already exceed the budget. One option is a circle embedding with three statistics per color, but 162 plus 486 still overshoots. I'm now asking whether two linear statistics could work, as long as the test term is nonnegative and zero only when the liberty matches the candidate point.

I'm considering a one-dimensional encoding where each point's index serves as its own coordinate, so the squared difference between a liberty and a candidate point is zero only when they coincide. That needs only three statistics per colour, but the squared coordinates get large enough that float32 cancellation could become a problem.

I'm now trying to keep the squared-offset sums exact in float32: the raw values are too large, so I'm shifting the coordinates to center them around zero, which should bring the magnitudes back inside the range where integers stay exact.

I'm narrowing the id range to nine classes so that squared differences stay small enough for float32 to keep exact. The cost is the real constraint, though: three statistics per colour, and because the test is taken at the candidate point p, those statistics have to be read at q.

I'm weighing whether a per-candidate test at each liberty could work if I use the neighbouring opposing stones' statistics, though a safe neighbouring chain would break the total, so I'm thinking about a per-stone version instead.

I'm checking whether the product A2·A0 − A1² could be approximated with ReLU units, since the variance test is zero exactly when every liberty value matches, but products of propagated sums aren't linear.

I'm checking whether a ReLU gate keyed to q's known liberty id could replace the product term, since that id takes only four possible values from q's neighbours.

I'm trying to make a single test at q work: each neighbour should give zero when it matches the one empty point. A 1D id doesn't quite do it, since on a grid no 1D labelling puts all four neighbours at the same distance from q.

I'm testing whether the 1D projection can be fixed by switching to a 2D distance: the squared distance from each liberty to q, minus one, should be nonnegative and zero only for the four neighbours.

I'm working out whether a 4-statistic-per-colour version fits: opponent chains alone would need 324 plus 162 carry channels, leaving no room for own chains. I'm looking at whether the carry or the A0 term could be dropped, or whether the modular encoding could bring the count back under budget.

I'm ruling out the full precise version, since computing four statistics per colour for every stone needs about 648 channels in the first layer, over the 512 budget. Next I'm looking at a cheaper two-tier scheme: an exact test over a single hop of statistics, with the rest of the evidence carried forward through propagation.

I'm now considering parity evidence as a cheaper alternative: tracking whether a chain has liberties of both colours' parity, which generalises the adjacent-liberty rule and propagates with two channels per colour.

I'm thinking about merging the parity idea into the walk-sum itself, but a single channel can't carry both the "even" and "odd" evidence, so that doesn't work directly. The combined cost comes out to about 648 channels, which is still over the budget, so I'm looking at folding the single-stone rule into the parity evidence somehow.

I'm settling on a parity scheme: each stone contributes to an even-channel and an odd-channel depending on its position, and a stone with two or more liberties counts toward both. That gives 486 channels per step, which fits the budget with a little room left, so I'm now working out the final test at the last layer.

I'm checking whether the parity channel can give the atari test directly: a stone next to an empty point always has a liberty of its own, so the test reduces to reading the opposite-parity channel for q, and the choice of channel depends only on q's fixed parity.

I'm checking whether parity evidence misses anything: chains where every lib-stone has the same parity and only one liberty each, like the ends of an odd-length line, would go undetected.

I'm checking whether the distance-2 straight evidence could be added in the first layer, but per-unit (s, d) computations don't fit, so I'm setting that idea aside. Next I'm looking at the range of the E/O propagation, starting from the initial contributions E0 and O0 that depend on the liberty-count indicators.

I'm checking whether the even and odd evidence channels can be computed in the first layer with one unit each: for even stones the presence test is just a clipped sum of the stone's liberty count, and for odd stones I need at least two liberties, so the threshold shifts by one. That brings the first layer to 486 channels per step, which fits the budget, so I'll move on to the propagation layers L2 through L5.

I'm checking whether the propagation depth still holds: starting from L1 gives four hops through L2 to L5, which is more than the old rule-b path managed. The catch is that L5 is already at 486 channels, so the atari flags I used there for the capture override and the eye check have no room left, and I need to find where else they can go.

Now I'm tallying the layer-six budget: the atari override needs a nonlinear term that can't be nested, so I'm weighing whether to compute the atari flags at L5 from three hops, though there's no room there. The remaining units at L6 are AP, AW, SA and EYE at 81 each, plus 162 carry channels for the linear output terms.

I'm weighing whether the second-order override for the attacking stones at L6 can be written as a single linear-plus-relu term over the L5 evidence, since the atari flag itself is too nonlinear to nest.

I'm checking whether the saturation penalty can be folded into the capture bonus in the logit, so that captures get a net positive score even where the penalty applies, and I'm weighing whether the escape case still holds up under that trade-off.

I'm considering whether the EYE check can work without the A5 atari flags: a plain sum over own neighbours would miss a neighbouring own chain in atari, which means a legal fill might be wrongly discouraged, though it's a rare case. I'm weighing whether a separate unit could pick up that adjacency, since the nonlinearity blocks folding it into the eye term directly.

I'm weighing whether to drop the L5 atari flags, since there are 26 spare channels but not enough for them. I'm also thinking about whether the positional side needs more attention, because our strength against the strong opponent is stuck around 37.5%, with repetition losses adding to the gap.

Next I'm looking at whether the positional side needs work, and what the strong opponent is doing to our chains: we keep losing stones to ladders and two-liberty groups even after our escape logic runs.

I'm going to check how our stones got captured in a strong game: for each capture, I'll look at the board on our previous move to see whether the chain was already in atari and what we played instead.

