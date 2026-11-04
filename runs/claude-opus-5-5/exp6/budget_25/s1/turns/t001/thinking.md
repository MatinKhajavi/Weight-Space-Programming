I'm starting by restating the task: a hand-designed 6-layer MLP policy for 9x9 Go, with weights set by formula rather than trained, and no search or sandbox runs. I'm noting the key constraints, namely the 243 binary inputs covering own stones, opponent stones, and the color plane, and the 6 hidden layers of 512 ReLU units with 82 output logits sampled through a temperature-1 softmax.

I'm weighing what makes a Go policy play well at this level: avoiding filling its own eyes and self-atari moves, capturing opponent groups in atari, extending its own groups that are in atari, and playing sensible shape moves early. I'm also keeping in mind that passing too early, the 162-move cap, area scoring with 7.5 komi, and the superko rule (where repeating a whole-board position loses) all shape the endgame.

I'm considering when passing actually matters: in the endgame, if every remaining legal move fills my own eyes or puts a group in self-atari, passing should be the preferred choice, so the pass logit needs to be moderately high. Two consecutive passes end the game and it gets scored, so passing is a good move when I'm ahead and not much use when behind.

I'm working out how to get local features like eyes and liberties from binary inputs with only 512 ReLU units per layer, since flood-fill connectivity is hard to express in a few layers. I'm leaning toward local pattern detectors for each point instead, though with about six units per point per layer that budget looks tight.

I'm checking how many units each board point can get: 512 hidden units over 81 points leaves about six per point per layer, plus a few spare. I'm thinking about whether the first layer's linear combinations can carry those per-point features forward, since ReLU on non-negative values passes them through unchanged.

I'm thinking about the channel budget per point: with six channels, a 6-layer net can carry a few local features forward, and the question is which ones are worth the space. Against the RL opponents, which are only moderately strong, avoiding self-filled eyes and taking captures matters more than deep reading, so I'm weighing what a small set of channels can encode for those.

I'm ranking the heuristics a simple Go bot would need: never filling a true eye, capturing stones in atari, extending my own stones that are in atari, avoiding self-atari, and playing near the center early. The hard part is that liberty counts depend on group connectivity, which a few local layers can only approximate.

I'm weighing whether a simple approximation for capture detection can work, since exact group liberty counts need global connectivity. I'm considering checking each opponent neighbour of an empty point, and whether that neighbour's only liberty is the point being played.

I'm weighing whether a propagation scheme across the six layers can compute group liberties well enough, since exact liberty counts need connectivity. I'm also reconsidering whether a sharp, tactically sound heuristic (avoiding eye-filling, capturing stones in atari, escaping atari, and avoiding self-atari) could already beat these opponents.

I'm now weighing a propagation scheme where each point carries an emptiness signal across the layers, so that liberty counts for a group could be approximated without a full flood fill.

I'm working through how shared liberties complicate counting: two stones of one group can touch the same empty point, as in an L-shape, so summing per-stone liberties would double-count, and the propagation has to deduplicate them.

I'm still weighing whether a radius-limited exact check for atari on the candidate's neighbourhood could work, since a 6-channel budget per point makes full pattern enumeration impossible. I'm also looking at how Boolean logic like AND and OR might be built from ReLU units within each layer.

I'm working out how to threshold counts from integer-valued sums, where an indicator like "at least k" takes two ReLU units, or one if the saturation can be left alone. I'm also considering whether the per-point budget of about six channels can absorb that cost across 81 points.

I'm sketching the first layer now: each point gets a feature for emptiness, which comes for free as one minus the own and opponent inputs. From there I want per-point channels that score each candidate empty point for its capture value, its save value, and its positional prior, which together form the logit.

I'm weighing the pass logit: it should be moderate so the network passes only when every remaining move is harmful. Passing early isn't automatically safe, though, since if the opponent has just passed, my pass ends the game and I lose unless I'm ahead. I'm considering estimating the score from stone counts, which the network could approximate.

Checking whether the opponent's pass is even visible to me: the input carries no move history, so I can't tell if they just passed. The endgame worry is that once they pass, my own non-eye-filling moves inside their territory would just get captured, which costs nothing in area scoring but gives them tempo to respond.

I'm noting that filling a point surrounded entirely by my own stones is the only move that really costs anything, since under area scoring any other move inside my territory is harmless. I'm also considering whether playing into the opponent's territory matters: it doesn't lose points, though a stone that gets captured could still give them a useful tempo.

I'm now checking whether repeated self-atari moves in opponent territory could recreate an earlier whole-board position under superko, since the opponent's recapture would fill the board back up.

I'm checking whether a pass in the endgame is a real option, since when every remaining move is an eye-fill or a self-atari inside their territory, the pass logit has to beat those.

I'm now checking an eye feature for empty points: a point counts as an eye for me when all its on-board neighbours are my own stones, which a single layer-1 unit can compute with a threshold, and the same unit works for the opponent's side.

I'm checking the opponent-eye case: if an empty point is ringed by opponent stones and the move is still legal, it has to capture something, so that's a free capture signal. Next I'm working out how to detect captures for opponent groups adjacent to a point by propagating liberty information through a few layers.

I'm checking whether a per-stone feature can capture atari globally: if an opponent group is in atari and p is an empty neighbour of it, then p must be that group's only liberty, so capture(p) reduces to an OR over p's opponent neighbours.

I'm now working out how to detect whether a group has only one liberty, since a single stone with two empty neighbours can still be part of a group whose liberties overlap with others. Checking whether two nearby stones share a liberty point is the hard part, and I'm considering whether a propagated signature could track that without enumerating every liberty.

I'm weighing a cheaper atari check that approximates group liberties by summing each stone's empty neighbours along the group, which is simple but double-counts any liberty shared by two stones, so it would miss some true ataris. The alternative is to assign each empty point a contribution counted once per group, which avoids the double counting but needs a way to tell which stones belong to the same group.

I'm stuck on whether counting distinct liberties across a group can be done without double counting, since the feed-forward layers can't easily track which empty points have already been counted. A propagation scheme over stones and liberties would need far more than six channels per point, so I'm weighing whether a depth-limited approximation is good enough.

I'm settling on how connectivity can spread through the network: each layer can extend a group's reach by one step, keeping only own stones, which means the AND-of-paths structure gets built up incrementally instead of enumerating every path.

I'm trying to see whether a ReLU unit can encode a two-level state for own stones, reached versus not yet reached, so the connectivity spread can carry that information forward one step per layer.

I'm weighing whether a ReLU unit can hold a three-level state for own stones, with empty and opponent at 0, own-but-unreached at 1, and own-and-reached at 2, so connectivity can spread one step per layer. A single unit can't cleanly separate a neighbour at level 1 from one at level 2 without a nonlinearity on each input, so I'm leaning toward two channels instead of one.

I'm weighing a simpler route: give each stone its count of empty neighbours and sum those across a group, which is cheap but counts shared liberties more than once. Some Go engines get around this with a pseudo-liberty trick that adds a sum of squares alongside the plain sum, and I'm checking whether that could be approximated here.

I'm looking at a way to test for atari without summing liberties across a group. The idea is that a group has just one liberty exactly when all its pseudo-liberties land on the same point, which a linear aggregate alone can't check, so I'm considering whether a different formulation could.

I've ruled out exact atari detection for large groups, since summing liberties over a connected component can't be done with a feed-forward net. So I'm turning to local patterns that handle small groups, starting with the simplest case: a single isolated stone.

I'm checking the small cases now: a lone stone with exactly one empty neighbour is already in atari, and for a two-stone domino the liberties are just the sum of each stone's empty neighbours, since no point touches both. Three-stone shapes are next, where a bent group can share one liberty.

I'm weighing a weighted count where each empty point's contribution is split by the number of same-colour neighbours it has. That gives exact liberties when all those neighbours belong to one group, but if they come from different groups, each group gets only a fraction of the point, so it would undercount and falsely flag atari.

I'm testing a shortcut: a group with a shared liberty weighted at one half can be misread as in atari, but I'm still stuck on the real issue, that a group's total liberties need connectivity to sum over the whole group. I'm now considering a flood-style signal from each empty point through adjacent stones, though telling distinct liberties apart that way is hard, and the group-level test still isn't settled.

I'm looking at whether a group has at least two liberties: if any stone has two empty neighbours, it does, and otherwise two stones would need to touch different liberty points. Tracking the distinct liberty points across a group by propagating a max and a min through the group seems like a way to test that, though a max over several neighbours takes several ReLUs to express.

I'm considering whether liberties could be encoded with binary codes so that two distinct points differ in some bit, but seven bits per point would need fourteen propagated channels, which is far more than I can afford. Using a cheaper code based on coordinate parity might still separate most distinct liberties.

I'm now weighing which parts of the game matter most against weak RL opponents: avoiding eye-filling and passing sensibly probably matter most, then capturing stones left in atari, escaping atari, and opening territory.

I'm laying out the per-point channel budget now: 81 points at six channels each uses 486 units and leaves 26 spare. I'm starting the first-layer features from own stones, opponent stones, and empties, with each point's neighbour count tracked so the channels can be defined from those.

I'm sketching the per-point features now, with liberty information for own and opponent stones, and an eye-like channel for empty points surrounded entirely by own stones. Then I want to work backward from the final logit: a legal empty point gets a positional prior, plus a bonus for capturing an adjacent opponent group that is in atari, plus a term for escaping atari.

I'm sketching a propagation scheme for atari detection: each stone starts with its count of empty neighbours, capped at two, and then that value gets passed along same-coloured neighbours on each layer.

I'm testing whether a simple sum of neighbour counts can detect atari, but it double-counts through the group: each stone's total feeds back into its neighbours, so one liberty can look like two. I want the signal to track distinct liberties, not repeated paths, and I'm now looking at a flood approach that starts from each empty point and tries to identify which liberty each stone actually reaches.

I'm weighing a scheme where each single-liberty stone carries a code for its liberty point, and then taking the max and min of those codes across the group. A group is in atari only if every stone sees the same single point, but a max over several neighbours is costly to express with ReLUs.

I'm checking whether a max over five neighbours is affordable with ReLUs, and it would take several layers, so that's too costly. I'm considering an alternative that encodes liberty points in some other way.

I'm trying to compute an OR over neighbouring binary flags with ReLUs, and the trouble is the magnitude: an unsaturated sum grows, so I'd need an extra pair of units to clip it back to 0 or 1.

I'm checking whether OR-style propagation without saturation still works: values stay nonnegative integers, so a final threshold can read off "nonzero," but gating with a large constant G has to exceed the largest accumulated value or the masking fails.

I'm trying to encode the mask and the propagated signal in a single channel by offsetting values, so a point that isn't own stays at zero while an own point carries a value of at least one plus the reached signal.

I'm checking whether a value that starts at zero for non-own points can stay zero through the propagation, and I'm stuck on the threshold: it has to exceed the largest neighbour sum, yet an own point with one unreached neighbour should still read as a plain own point, not as reached.

I'm checking whether a bit that shrinks to a value between the two ranges can keep the unreached and reached cases apart after each layer. The offsets have to grow, so the scheme gets fiddly. Since the opponents are weak, I'll first consider whether local features such as eyes and captures of small groups matter more.

I'm working out how to keep a running logit accumulator per point, shifted by a constant offset so the ReLUs stay non-negative, and folding the final bias in at the end instead of copying every feature forward through all six layers.

I'm planning a running logit accumulator channel per point that carries a partial score plus an offset, leaving five channels for other features, and I'm considering a layer-one eye-detection feature that would subtract a large penalty when all neighbours are already own stones.

I'm now looking at whether each opponent stone's liberty count can be computed in the first layer as a linear function of its empty neighbours, gated by whether the stone is the opponent's.

I'm working out a per-stone liberty count for opponent stones, gated so only opponent stones carry a value. Then I want to count distinct liberties for a whole group, since one shared empty point could otherwise be counted several times.

I'm checking whether a group is in atari by asking, for each empty point, whether it is the only liberty that the adjacent stones' group has. That would mean tracking the other liberties separately for each direction, which costs too many channels, so I'm looking at a per-stone signal of whether it still has an empty neighbour.

I'm defining, for each opponent stone, a signal for whether it has a liberty other than a given empty point e. Only stones next to e need that check, so the group-level test becomes an OR over its stones of whether each has enough empty neighbours beyond e. I'm now considering a per-stone signal that tracks whether a stone has two or more liberties, to feed that propagation.

I'm now looking at whether the group-level test can be done with counts: a group is safe from atari if it has a reachable stone with two or more liberties, and otherwise I need to know if any liberty-stone sits outside the neighbourhood of the candidate point e.

I'm now trying to avoid counting the same liberty twice by looking for a code per liberty point, checking whether every single-liberty stone in the group names the same empty point.

Trying to tell distinct liberty points apart by their coordinates: a per-point code like row and column modulo 3 gives nine classes, but that alone would need nine one-hot bits propagated by OR.

Now I'm trying to see whether a single propagation of a count across a group can work, given that it echoes. I'm leaning toward accepting exact results only for small groups within a fixed window around the candidate point, computed directly in a few layers. For single stones, I'm checking what atari means: no same-colour neighbours and exactly one empty neighbour.

I'm settling on a single-stone atari test: an opponent stone next to the candidate point e whose other neighbours are all own stones or edges. That can be a single first-layer AND, one unit per stone rather than one per direction, which keeps the channel count down.

I'm defining, for each opponent stone next to a candidate point, a first-layer test: the stone is opponent-coloured, all its neighbours but one are own stones, and that one remaining neighbour is empty, which would make the empty point a capture point.

I'm noting that a stone with all but one neighbour already its own, and no opponent neighbours, can be flagged in the first layer, so a single unit per point picks up this single-stone atari. Since a fully surrounded stone would already be captured, exactly one empty neighbour should remain, which is the capture point. Summing these flags over a point's neighbours at the second layer would then give a direct capture-single signal for that point.

I'm noting that a single-stone capture can be flagged in the first layer, and an own stone in atari escapes only if the point I play next to it has at least two empty neighbours. I'm now turning to two-stone groups, where the same kind of single-layer test runs into trouble with the "exactly one" condition.

I'm settling the domino test: two adjacent opponent stones, with exactly one empty point among their six surrounding points and the rest own stones, which would make that empty point their capture point. Edge cases get fewer surrounding points, and capture at e then becomes an OR over the dominoes next to e, costing about 144 units in total.

I'm weighing whether 144 layer-one units for domino detection is affordable, since layer one only has 512. Enumerating larger shapes like trominoes blows up the unit count, so I'm turning toward a propagation-based approach to test whether a group has two or more liberties.

I'm now defining a group-level condition: it has at least two liberties if one of its stones has two or more, or if two of its stones have different single liberties. From a candidate empty point's view, the group is stuck on that point only when every liberty-carrying stone touches it and has no other liberty.

I'm working out when a group is in atari: a stone counts as "bad" for a candidate point e if it has two or more liberties, or if its one liberty is some point other than e. Tracking the liberty points each stone sees could then check whether every liberty is e.

I'm weighing an absolute position encoding instead: every liberty-stone adjacent to e sits in a small diamond around it, so the group is in atari exactly when all of its liberties share one row and one column. The catch is that propagating those coordinates across the group would need separate channels for row and column.

I'm weighing a propagation that tracks, for each stone, up to two distinct liberties within a few steps, since summing counts would double-count through multiple paths.

I'm now weighing a min-based test instead of a sum: a group can't be in atari if any of its stones has two or more liberties, which local counts can flag and OR can propagate. The hard remainder is groups where every stone has at most one liberty, since those are either in atari or have two liberty-stones pointing at different points.

I'm now working on distinctness: two liberty-carrying stones are guaranteed to have different liberties if they are adjacent, since neighbouring points share no common neighbour. Stones two steps apart might still share a liberty, so I need to handle that case separately.

I'm checking the Manhattan-distance cases: stones two or more steps apart can't share a liberty, so distinct liberties there settle it. The awkward part is that a local per-stone flag for "differs from a nearby liberty-stone" can't tell whether that neighbour is actually in the same group, so I'd need a connectivity check too.

I'm looking at whether a group has a liberty that isn't next to the candidate point, since that would rule out a capture there. The per-point view is what matters: I need to know whether the neighbouring group has any liberty outside that point's neighbourhood or any stone with two or more liberties.

I'm checking whether a stone in N(e) with a single liberty still counts toward the group's other liberties. Its only liberty is e, so I'd rather track a signal that means "a liberty-stone whose liberty is not e," since that signal stays correct for every candidate point at once.

I'm checking whether a neighbouring liberty f must be distinct from e: if f is not adjacent to the stone n, it can't be e, but if it is adjacent and differs from e, then n itself has two liberties.

I'm checking which distance-2 neighbours of n can supply a liberty outside N(n), since a straight two-step neighbour shares only the middle point with n while a diagonal one shares two.

I'm working out that for any empty neighbour e of stone n, the group has another liberty exactly when n is in atari-breaking shape or the group has a liberty not adjacent to n, so the test no longer depends on which e I pick.

I'm checking that a stone n can only be in atari at e when its own single liberty is e and the group has no liberty outside n's neighbourhood. So the test for n is that it has exactly one liberty and the group's remaining liberties are not adjacent to n. I'm setting up that per-stone flag as the next piece to compute.

I'm now trying to pin down X(n): whether the group has a liberty not adjacent to n. Every other stone in the group falls into one of a few cases by distance from n, and the open question is how to propagate that signal across the group without passing back through n.

I'm settling on a path argument: any stone in the group other than n can be reached from one of n's own group-neighbours without going back through n, so a liberty anywhere in the group is visible from those neighbours.

I'm now checking whether the group's other liberties can be seen from the neighbours of n, since any stone in the group can be reached from one of them without passing back through n. The obstacle is that a simple "group has at least two liberties" signal would be true for every group on the board, so it isn't useful on its own.

I'm restating the test for a stone n with a single liberty e: its group is in atari at e exactly when no other stone in the group has a liberty different from e. Since only stones adjacent to e can have e as a liberty, I'm checking whether any stone elsewhere in the group carries a different liberty.

I'm working out Other(n) for a stone n with one liberty e: it's true if some stone in the group, outside N(e), has a liberty, or if some group stone in N(e) has two or more liberties. I'm about to handle the stones in N(e) separately, since their liberty counts are local, and then propagate a global signal everywhere else while leaving those excluded.

I'm testing whether a per-stone signal can tell if a stone's group has any liberty other than the candidate point. Splitting the stones into those with one liberty and those with two or more, the question becomes whether one-liberty stones carry a liberty different from the candidate's.

I'm checking whether some liberty-stone sits far from n, at Manhattan distance three or more, since such a stone would already give a liberty distinct from e. Closer liberty-stones still need their liberty compared against e, so I'm looking for a way to propagate a signal of "liberty-stone far from n" across the group.

I'm trying to find a cheap way to tell whether a group has a liberty-stone far enough from n to count, but grid distance and graph distance don't line up, so a plain propagation along the group won't work cleanly. I'm weighing a walk-based signal over the group, though it would pick up backtracking.

I'm trying to approximate whether a group has a liberty far enough from a given stone, and the idea is to work on empty points rather than stones, counting whether there are two distinct liberties adjacent to the group.

I'm now looking at a per-group, per-point check: does a group have a liberty other than the candidate point e? I'm considering how to express that, and I plan to define it for each group and candidate point so I can then test captures at e.

I'm now thinking about whether a per-direction signal can carry "a liberty other than e" through the group with the excluded point held fixed, so each candidate neighbour gets its own channel.

I'm trying to get a group's liberty count without double-counting shared empty points, so I'm weighting each empty neighbour by how many same-coloured stones touch it. If every such neighbour belongs to a single group, the weighted sum over the group's stones equals its exact liberty count, and the only error comes from liberties shared between separate groups.

I'm weighing whether propagating a single signal through the group can work, since summing liberties by propagation runs into echo and multipath problems. A window-restricted sum has the same issue of pattern blow-up, and a threshold unit can't capture connectivity. I'm leaning toward a multi-layer scheme that saturates at "two or more", accepting that echoes will overcount some liberties and miss a few ataris.

I'm weighing which mistakes cost the most: a false capture or escape is mostly wasted moves, while missing a capture or walking into self-atari loses stones. I also recall that simple rule-based Go policies only reach roughly 1000–1200 Elo on 9x9, which makes these opponents, beating random play 79–98% of the time, look quite weak.

I'm weighing how strong a heuristic player needs to be to beat these opponents, since random-level play is an easy bar. My working idea is that solid basics (not filling eyes, capturing what can be captured, avoiding self-atari) should win most games against weak networks.

I'm thinking about the priority order: first never filling my own eyes, then capturing or rescuing groups in atari, then avoiding self-atari, then good positional play. For the atari detection I'm considering a per-stone encoding of liberty indices that propagates through the group.

I'm weighing how to propagate a maximum across the group, since a plain max-propagation runs into the same echo problem as summing, and I'm considering a threshold-style trick to get around it.

I'm checking whether a group that is all one-liberty stones can be counted as having two or more distinct liberties, since walk-count propagation only tells me that some liberty exists, not how many distinct ones there are.

I'm pinning down the atari condition: a group is in atari exactly when it has no two-liberty stones and all its liberty-stones touch one shared point. For each one-liberty stone, I'm now checking whether the group has any liberty-stone outside that point's neighbourhood, starting with the group stones that sit next to the liberty.

I'm checking whether a liberty-stone outside the candidate's neighbourhood would give the group a distinct liberty. For single-liberty stones, that reduces to whether their liberty differs from e, so the test becomes whether any such stone's liberty is not e.

I'm assigning each empty point a code so that liberties within distance two of each other get different codes, which would let a single propagated signal tell distinct liberties apart. Points farther apart might collide, so that would only cause an occasional missed atari.

I'm checking whether a five-colour pattern like (r+2c) mod 5 keeps every pair of points within distance two distinct, and I'm finding that some offsets collide, so the simple version doesn't hold.

I'm checking the mod-5 colouring against every pair of points within distance two, and the offsets I've tried so far all land on distinct colours. So the scheme holds for the pairs I've examined: nearby liberties get different codes, which is what the distinct-liberty signal needs.

I'm checking whether a five-class coding of the empty points, with each class propagated as its own channel, can tell distinct liberties apart. Two far-apart liberties can still share a class, so the scheme will sometimes miss a distinction, and I'm weighing whether that false-atari rate is acceptable given that its mistakes are fairly mild.

I'm weighing whether two propagated signals, one for liberty points coded 0 and one for those coded 1 in a checkerboard pattern, could separate distinct liberties more cheaply than the five-class scheme, which would use up the whole channel budget.

I'm considering whether graph-distance exclusion works better than grid distance: a group can wrap around an empty point so that its stones next to that point are far apart along the group's own connections, meaning the exclusion has to be based on the group's actual connectivity, not on distance alone.

I'm considering that a group with a single eye and no outside liberties would be captured by playing at e, since legality already implies capture when every neighbour is an opponent stone. So for points whose neighbours are all opponent stones, I could add a large bonus whenever the move is legal, and handle that case separately from the propagation scheme.

I'm now looking at a propagation idea for deciding whether a group has a liberty not adjacent to a given stone, starting from the observation that an opponent-surrounded empty point is already handled by legality. I'm defining a per-stone signal that accumulates a liberty indicator from group members within a bounded graph distance, and I'm checking whether backtracking through neighbours will spoil the OR.

I'm checking whether passing a signal through a neighbour can loop back to the stone it started from. Going only along directions that don't return to the sender avoids the echo, and I'm weighing whether a loop back to the stone can still cause trouble.

I'm checking a 2x2 block: a non-backtracking walk going around the four stones can return to the starting stone after four steps, so the signal from its own liberty could loop back and echo.

Now I'm working out how the liberty signal should treat the neighbours of n, since a stone's own liberty could echo back through the group's connections.

I'm weighing whether a group-level flag like "two liberty-stones with different liberties adjacent within the group" can stand in for a full liberty count, accepting some approximation error from a few propagation steps.

I'm turning to a stronger test: a stone with one liberty whose group-neighbour also has a liberty must mean two distinct liberties, since adjacent stones share no neighbour. That gives a flag for groups with two or more liberties. For stones two apart with a shared empty middle point, the same reasoning needs its own check, because a shared middle point can hide that the two liberties coincide.

I'm working out which pairs of liberty-stones can share a liberty: stones more than two grid steps apart can't, so only nearby pairs need checking. That means the group is in atari only when every liberty-stone touches the same empty point, and I'm considering counting liberty-stones as a way to test for that.

I'm thinking about whether the group's liberty count can be estimated by looking only at a local window around the candidate point, since a full propagation over all liberty-stones may be too expensive or miscount through loops.

I'm noticing that a window-limited propagation would need separate channels for every candidate point, which quickly gets too expensive. I'm stepping back to think about what actually matters in weak play: groups get left in atari and small groups get captured, and that's what a policy needs to handle reliably.

I'm now trying to fit the propagation into the budget: two signals per stone over three or four layers, one for each colour. The catch is that the weights are fixed linear maps and ReLU features can't be negative, so a shared channel can't carry colour; the propagation would need to know each endpoint's colour to block flow between opposite stones.

I'm working out how a per-stone signal can encode both ownership and a liberty flag in one channel: a value of 1 for own stones and 2 for those carrying the signal, so each layer can gate propagation by ownership.

I'm working out how the signal should behave at each stone: an unreached own stone gets 1, a reached one gets 3, and empty or opponent points contribute nothing. Then a single linear step with a bias should separate the three cases: no signal, own but unreached, and reached through a neighbour.

I'm working out the reached-stone value so the gap holds: if it's large enough, a single stone's unreached neighbours can't pull its response up into the reached band.

I'm working through the bit-width constraints for the signal: the reached value has to be large enough that the bands stay separate, but the scheme only carries one bit per channel, so I'd still need a separate channel for colour, and adding a channel per layer quickly gets too costly.

I'm weighing whether the two-bit case can be handled at all: with a single channel per colour and one ReLU threshold per layer, two independent flags don't seem to survive propagation together, so I'm inclined to simplify and keep only the most important signal.

I'm listing what signals a policy would need per stone at the end of propagation, for both colours: whether a group is in atari, whether it has two or more liberties, and its size. I'm also noting that a candidate move's self-atari depends on the liberties of the merged group. Then I'm considering a simpler but exact way to test for two or more liberties at limited depth, by propagating an atari flag with an extended definition that also checks neighbouring stones in the group.

I'm checking whether propagating a liberty-stone flag with distance can catch the case where a long chain has liberties far apart, since local flags miss that.

I'm checking whether a liberty-stone at graph distance of three or more from the stone n can still share the same liberty e, since the group would then wrap around e. If e has several neighbours of the same colour, they may belong to one group curling around it, so I need to look at e's surrounding stones directly to settle whether the group is in atari.

I'm checking the single-neighbour case first: if only one stone of the group touches the empty point e, the group is in atari exactly when that stone has one liberty and no other stone in the group has any. The question left is how to handle e with two or more neighbours from the group, since those can all carry liberty e at once.

I'm checking whether any liberty-stone outside the stones adjacent to e exists, and that check depends on e. Since only a few stones touch e, I'm looking at whether propagation can exclude those, which would need sum-based walk counts rather than OR-propagation.

I'm considering whether the opponent-stone walk counts can be subtracted out per candidate point, but that would need a separate computation for each empty point. So I'm flipping the direction and thinking about propagating outward from each empty point instead, which would still need one channel per point.

I'm weighing an averaging update instead of a sum, where each point takes the mean of its own value and its neighbours in the group, which should settle to a weighted average rather than a count.

I'm considering whether a max over a sum-style signal, passed through a ReLU, could stand in for the counting I need. I'm also thinking about treating each empty point as a token that feeds into adjacent groups, with deduplication only mattering when the point touches several same-colour stones. I'm now asking which atari cases are covered by simple local rules, starting with the single-stone case.

I'm now checking the two-stone case as an exact local rule: a stone with one empty neighbour, exactly one same-colour neighbour, and that neighbour having no other opponent neighbours and no liberties would put the pair in atari. Layer 1 can compute each stone's liberty count and same-colour degree, so layer 2 could combine them into this test.

I'm checking the simple two-stone case as an exact local rule: a stone with exactly one same-colour neighbour, whose partner has no other liberties, would put the pair in atari. I'm also considering whether propagating a liberty flag along non-backtracking paths for a few steps could approximate larger groups, though that gets expensive quickly.

I'm reconsidering a propagation trick that could flag liberty-stones beyond the one adjacent to the empty point, but it runs into the same problem of needing a separate channel for each point. I'm going to look at which atari cases simple local rules can handle exactly, and then think about what logit signals a candidate move needs: capture value, escape value, and self-atari avoidance.

I'm now looking at self-atari for a candidate move: if the new stone has at least two empty neighbours it can't be in atari, and if it has exactly one, the question is whether every adjacent friendly group is also out of liberties apart from that point.

I'm working out self-atari for a candidate move: if the new stone has two or more empty neighbours it can't be in atari, but when only one is empty, it depends on how many liberties each adjacent friendly group has. Groups with three or more liberties are safe, while a two-liberty group only matters if its other liberty differs from the remaining empty point, so I'd need per-group liberty counts to judge this exactly.

I'm settling on a local scheme: each stone gets its own empty-neighbour count, and a flag for whether it has enough visible liberties, which then gets passed through its group for a few steps. The open question is whether that propagated signal stays accurate enough to tell a group in atari from one with more room, or whether the double counting will blur the cases that matter.

I'm now testing a distance-two pattern: if a liberty-stone at one end of a chain and another at the far end are two steps apart with only a stone between them, they give the group two distinct liberties, so I'd flag that case with a separate strong-flag signal.

I'm now considering whether propagating a signal only through stones with no liberties could tell apart two liberty-stones that actually give distinct liberties, though it doesn't yet resolve the distinctness question.

I'm checking which liberty-stones belong to the group around e, and I'm labelling each one by where its liberty sits. Then I'll see whether paths from the stones next to e reach any liberty-stone that lies outside that set.

I'm now thinking about a way to pin down where each liberty point sits, by coding its location into a few classes so that two liberties can be told apart. A five-class code would separate any two liberties within distance two of each other, though liberties farther out could still collide.

I'm checking whether a liberty-stone far from the neighbour n, at grid distance three or more, could still share the same liberty point with n, since both are adjacent to e.

I'm checking whether a liberty-stone more than two grid steps from n must belong to a different liberty, since any stone in N(e) sits within distance two of n. Stones close to n can be tested locally for a shared connection, but a link that only exists through a long path would be missed.

I'm now trying to find a way to tell whether some liberty-stone in the group sits at grid distance three or more from n, using a propagated signal. Coding the position of each liberty into a few classes doesn't work, since points within distance two of n cover every class. A hop-count style signal along paths is the remaining option I'm weighing.

I'm checking whether a group that wraps around e could be missed by the graph-distance test, since two stones can sit close on the grid yet be far apart in the group. If e has some neighbours in G and others owned by the opponent, that case needs separate handling.

I'm checking whether a liberty-stone adjacent to n could be a different liberty from e: two neighbours of a point never touch each other on the grid, so an adjacent stone's liberty can't be e itself. Stones at graph distance two or more are where shared liberties get harder to rule out.

I'm now checking the case where a liberty-stone sits two steps away from n through a middle stone, with the shared liberty at e. Those four points form a 2x2 square, so that pair can be excluded from the "other liberty" test.

I'm weighing a sum-based propagation over walk counts on the group's subgraph, since an OR signal would echo back the stone's own liberty status through its neighbours. The walk count from a stone back to itself would give that echo, so it could be subtracted out to isolate liberty-stones at other positions.

I'm looking at whether liberty-stones far from n can be told apart from those close to it. The hard part is finding whether any liberty-stone in the group sits at grid distance three or more from n, since a propagated signal would echo back through n's own neighbours. A two-class coding of liberty positions is the idea I'm about to test next.

I'm trying a three-class position code, (r + c) mod 3, to see whether it can separate a liberty from e using just the neighbours of e. Points adjacent to e get codes different from e's, but a neighbour's liberty could still share a class with e, so the comparison doesn't yet settle whether the other liberty is distinct.

I'm weighing whether a three-class position code, (r + c) mod 3, can separate a distant liberty from e, since diagonal and far offsets keep colliding with e's class. Max-style propagation with one signal per point might avoid that, so I'm starting to test it.

I'm now looking at whether summing walk-weighted codes and walk counts across the group could recover the distinct liberty information without needing a separate channel for each class.

I'm now checking whether summing walk-weighted squared deviations from a code value could detect any neighbour with a different code, since cancellations are possible if I only use signed sums.

I'm now seeing that three running sums over the group, the total weight, the weighted codes, and the weighted squared codes, could be enough to test whether every liberty shares e's code. Since the code of e depends only on which neighbour of n holds the liberty, I can precompute four candidate test values per stone and pick the one matching its liberty direction.

I'm now checking whether the quadratic form works for detecting a differing liberty: liberty-stones sharing e's liberty contribute zero, so only stones with a different liberty raise the sum above zero.

I'm working out whether the total over all stones in the group, each weighted by walk counts and summed over its liberties, comes out above zero: that happens exactly when some liberty differs from e's code.

I'm finding that the per-stone quantities C, V, and Q are linear in the board state, with the colour gate applied through a ReLU, so each stone's three values can be computed in the first layer for both colours.

I'm trying to cut down the three propagated sums by picking codes that let fewer channels do the job. Using 2D coordinates instead of a single code would mean tracking sums of the row and column values separately.

I'm thinking about whether three running sums over the group, the walk-weighted count, the code total, and the squared-code total, are enough to cover the test. That would mean three channels per colour, six per point, which is most of the budget before the propagation layers even start.

I'm gating each propagation step by colour, so that a point only takes in signal from same-colour neighbours, which means the gate has to be available in the previous layer.

I'm checking whether a stone's own colour can be encoded in the channel as an offset, so the mask survives propagation even when a stone has no liberties. The worry is that the propagated values could grow large enough to swamp the gating constant.

I'm worried the sums will get too large for float32 precision to tell a zero from a difference of one, so I'm looking at whether smaller codes or a different test would keep the comparison exact.

I'm now checking whether the propagated values can grow large enough that float32 rounding swamps a difference of one. The concern is the ratio between a single far-liberty term and the total walk weight, since with small index-based codes that ratio gets tight.

I'm checking whether codes in a small range, such as (r + 2c) mod 5, would keep the propagated values small enough for float32 to still tell a zero from a difference of one.

I'm checking whether combining both colours into a single channel would work, but propagation only sees neighbours and a ReLU can gate only at the receiving node, so opponent stones would leak into the sums.

I'm now weighing whether each colour needs its own copy of the three moments, which would push the per-point budget to six channels. One idea is to handle the colours by magnitude or by offset, but ReLU only gates at the receiving node, so opponent stones would still leak into the sums. I'm considering whether a code relative to each stone could cut the moments down to two.

I'm considering whether the propagation weights can be arbitrary per pair, so a relative-frame translation of the codes could work, though that still costs the same channels. I'm now wondering whether a single channel with a carefully chosen nonlinear-free test could detect whether any liberty differs from e.

I'm weighing whether a single propagated channel per stone can work if its weights are chosen so that a total matching the expected value for e forces every contributing point to be e. Because the walk weights are unknown, I'd need a matching count channel as well, and a single function g can't rule out a convex combination of other values landing on g(e) unless e is the unique extreme among the candidates. Since e varies per stone, that extremal choice doesn't hold uniformly.

I'm considering a per-stone test function that is minimized at the liberty e, but since the codes must be expanded into moments, that again costs channels. I'm also weighing whether a local frame relative to each liberty stone could cut the work, though relative positions shift under propagation, so that doesn't obviously help. The bigger constraint is that two colours times three moments makes six channels per point, which uses most of the budget.

I'm counting layers to see whether the propagation steps fit, and the mask x(s) has to be available inside each propagation layer, so I'm considering folding it into one of the moment channels.

I'm trying to merge the colour mask into the count channel, so a stone's presence survives propagation, but subtracting the separate weight sum would bring the extra channel back.

I'm checking whether the gate can be read off the propagated count itself: a stone of the right colour leaves a value of at least K in that channel, while a wrong-colour point stays at zero. If so, the mask could feed the pre-activation of the next moment without spending another channel.

I'm checking whether the gating mask can ride along in the propagated count channel, so that a wrong-colour point stays at zero and a stone of the right colour carries at least K. The open question is whether the correction term that the gate adds on own-colour points can be removed in the next layer, since the propagated count tracks the same quantity.

I'm checking whether the corruption introduced by the gate can be tracked through propagation, since the correction is a linear function of the previous layer's count channel. I'm defining the true moments and a corrupted version offset by the gate term, and seeing whether that offset propagates in a predictable way.

I'm now trying to see whether a single mask channel per colour would keep the gating exact through each propagation layer, but that brings the per-point count back up to eight channels, which is too many. So I'm looking at whether the moments themselves can be trimmed to free up room for the mask.

I'm testing whether a single 0/1 colouring could replace the per-code moments, but it fails: a checkerboard colouring would ignore same-colour liberties at distance two, which are common. Two channels would be cheap, yet the approach doesn't cover every liberty, so it's ruled out. That leaves the six-channel cost, which would crowd out the accumulator and the eye detection.

I'm thinking that with only two propagation steps, groups beyond graph distance two would be invisible, so a long group with a single visible liberty would look like atari. That means a capture check could fire falsely, which is a one-directional error I'd have to accept.

I'm noting that the atari-like false positives from short propagation are acceptable for some moves, while self-atari overestimates would make the policy too cautious. I'm also looking at how the depth budget splits across the layers: local moment features first, then propagation steps two and three, and finally the combining layer.

I'm now working out how to test a stone's atari status per direction: the quadratic form on its neighbour decides whether it has exactly one liberty, and that gates a capture check at the candidate point.

I'm checking whether a per-neighbour quadratic test can flag a capture without any gating: for each neighbouring stone, the test value should be zero exactly when all its liberties sit at the candidate point. I still need to work out how to test that zero condition cheaply alongside the opponent-colour check.

I'm working out how to tell whether an opponent stone is in atari at a candidate point. The quadratic test is zero only when all its liberties sit at that point, but it can't tell an opponent stone from an empty point, since the moments are zero in both cases. So the colour check has to be carried into that layer somehow.

I'm using the moments at the neighbouring stone to form a test that is zero exactly when all its liberties sit at the candidate point. That test alone can't distinguish an opponent stone from an empty point, so I'm looking at whether the liberty-count sum can serve as a presence signal, since it's positive for any stone with a liberty and zero otherwise.

I'm now trying to build a test unit that is zero for empty points and fires only for opponent stones with no other liberties. The difficulty is that the quadratic test alone doesn't separate the two cases, so I'm weighing a weighted combination of the liberty sum and the test value, with the constants chosen so the opponent case stays positive while the rest is squashed to zero.

I'm checking the precision requirement: the test value T has to be an exact integer for the thresholding to hold, which means the codes and walk counts must stay integral, and any float rounding would break the zero-versus-positive split.

I'm checking whether summing the per-neighbour tests could replace testing each neighbour separately, but an OR over neighbours needs each test kept apart, so that doesn't save any units.

I'm trying to test per stone instead of per candidate-direction pair, checking whether all of a group's liberties collapse to a single point. The variance-style condition for that would be a product of moments, which a single linear layer can't compute.

I'm looking at a per-stone version: for a stone with exactly one liberty, that liberty's code could be computed linearly in the first layer, and then the quadratic test would only need the stone's own moments and that code.

I'm weighing a per-point test unit for each neighbouring stone, but that adds four units per colour on top of the six channels, which is over budget. I'm considering whether the test could instead be folded into the final propagation step or built from a different local frame so fewer units are needed.

I'm considering a different option for detecting atari: propagating a single-bit "safe" flag through OR operations instead of summing weighted counts. The concern is that the flag would echo back to the neighbouring stone itself, so I'd need to handle that self-reinforcement somehow.

I'm looking at whether a propagated flag could mark a liberty stone as lying in atari, by only letting the signal travel through zero-liberty stones. The worry is that a path could loop back to another liberty stone next to the same empty point, which would still count as a false positive.

I'm checking whether a propagated flag that only travels through zero-liberty stones can separate a true other liberty from the blocking stone next to the candidate point. The trouble is that a path can stop at a stone that has only the candidate as its liberty, so the signal wouldn't reach the rest of the group. Keeping the one-directional error in mind, I'm considering whether to let the signal pass through those blocking stones as well, which would bring back the circularity I was trying to avoid. I'm now turning to Plan D, which is to keep direct moments but share the colour channels to reduce the unit count.

I'm considering whether fewer moments could work with a small code set: codes in {0,1,2} still need three moment channels, and a two-colour encoding gets down to two channels but only detects differently coloured liberties. I'm now weighing whether a code in {-1,0,1} changes the count, though it still needs three moments.

I'm rewriting the opponent-count term as M0 minus the walk weights from the candidate's neighbours, which would make the test depend only on the nearby stones. I'm checking whether those walk weights stay local when the walk length is three or less.

I'm now weighing whether the opponent-count term can be written as M0 minus the walk weights from n to the stones adjacent to the candidate point, and whether those weights can be propagated to n as a signal that depends only on nearby stones.

I'm wondering whether the walk-weighted liberty count can be rewritten as M0 minus the weights from n to the stones touching the candidate point, which would make the test depend only on nearby stones. I'm also considering whether two channels, the count and a second moment with carefully chosen codes, could carry enough information to replace the full set of three.

I'm now checking whether the squared-code test can be replaced by a complex-valued version: giving each point a unit vector with a distinct angle, so the weighted sum's dot product with a candidate's own vector only reaches the full liberty count when every contributing point is that same candidate.

I'm checking whether a two-channel encoding can work by dropping the plain count and using only the weighted vector sum, since with unit-circle codes a norm test could in principle stand in for it.

I'm checking whether the "mask" from the M0 channel can stand in for the third moment, but M0 is zero for stones with no liberties nearby, so it can't tell those apart. That leaves three channels as the minimum for an exact test.

I'm considering whether a fourth channel, a plain mask count, could gate propagation so zero-liberty stones still pass the signal, with M0, M1 and M2 kept as the separate moments.

I'm checking whether the colour mask needed at each layer can be derived from the previous layer's mask rather than from a dedicated channel, since a plain count of weights can't serve as a 0/1 gate.

I'm now trying to get the gating to work: the colour mask x(s) is needed at every layer, but it isn't available after the first layer, so I'd need to carry it forward somehow. Packing own and opponent into one channel doesn't work because a 0/1 mask isn't a linear function of the combined value. I'm considering instead folding the mask into the moment channel itself, as x(s) times an offset plus M0(s).

I'm working out how to make the gating coefficients work: when the cell isn't of the colour in question, its pre-activation has to stay non-positive, and when it is, the offset has to keep it positive. From those bounds I'm deriving how large the scaling factor α must be, and then writing the new output as a constant shift plus α times the walk-weighted moment.

I'm noticing that the extra channel for the colour gate adds a constant plus a term that counts the candidate's same-colour neighbours, so the gate itself leaks into the moment. That count is just another walk-weighted quantity built from local stones, so it may be possible to absorb it into the existing moment rather than spend a separate channel on it.

I'm checking whether the gating term can use a combined channel instead of a separate mask, scaling the stone's own moment so the gate only fires for the right colour. The concern is that the extra term would spill into the moment, so I'm working out whether the bounds still hold.

I'm checking whether tracking the pollution term directly could work: each layer's correction would be the masked sum of the previous layer's corrections plus whatever new pollution appears at that step.

I'm now working through a plan that uses the input's limited depth, though I'm still weighing whether a simpler approach could capture most of the benefit without all this gating bookkeeping.

I'm weighing a lean design that keeps per-point channels, since the other option of pattern-level units doesn't fit a variable number of liberty points.

I'm weighing whether to focus the moment-based detection on opponent groups, since captures are probably the highest-value signal, and whether own-group atari can be handled more cheaply. I'm also thinking about how many channels the gating would take if the two colours share resources.

I'm trying to keep each layer's colour gating from needing its own mask, by storing the moments for own and opponent stones in separate channels. Then a receiver can gate on its own colour while the senders are already restricted to the right colour by their channel.

I'm noticing that the accumulator at stone points doesn't matter, since the output is masked there, so that channel could carry extra information on occupied points. I'm asking whether it could double as the own-colour gate for the moment channels, which would save a dedicated mask.

I'm thinking of using a fourth channel, a stone-indicator value, as a gate: stones of the wrong colour get a large negative shift, so the gated signal only passes where it should. I'm checking whether the accumulator value on occupied points can carry this, since those points are masked out anyway, and whether a scaling factor large enough to suppress the gated-off cases still keeps the valid values non-negative.

I'm settling on an accumulator channel that is zero at own stones and at least one elsewhere, so it can act as a gate without corrupting anything. The remaining question is whether a second accumulator can cover opponent stones the same way, since one channel can only zero out one colour.

I'm trying to make the accumulator at own stones zero by subtracting a large term there, but that requires knowing own-colour status again at the next layer. I'm considering whether the previous layer's zero/non-zero pattern can stand in for it.

I'm trying to make the accumulator zero at own stones by subtracting a large term there, but that would mean knowing own-colour status again at the next layer. I'm considering whether the previous layer's zero/non-zero pattern can stand in for it.

I'm working out a way to make the accumulator at own stones zero without needing own-colour status again at each layer. The idea is to subtract a large constant there, keeping non-own points far above that threshold so they stay positive and the gating survives into the next layer.

I'm checking that the accumulator stays large enough at non-own points while shrinking by a fixed amount each layer, so a starting offset around 1000 with a shift of 50 still leaves it well positive after a few layers. I'm also noting that float32 precision at that magnitude is more than enough. The next question is whether the same trick could make the accumulator zero at opponent stones too.

I'm now asking whether a second accumulator, zero at opponent stones, could be combined with the first to exclude both empty and opposing receivers when gating own-colour moments.

I'm now checking whether the own-colour and opponent-colour moments can share channels, since reducing the moment count would close the channel budget gap of two.

I'm considering whether the moment-based codes can be made distinct only within a bounded neighbourhood of each empty point, since the search reach is limited to a few hops, which would relax the injectivity requirement.

I'm now considering whether a two-class colouring could replace the three-moment scheme, since a binary code only needs two channels to detect whether a neighbour's class differs from the receiver's.

I'm weighing whether a two-colouring of the liberty points could replace the three-moment scheme, and I'm checking whether a neighbouring stone of the same group would create a mismatch I'd need to rule out.

I'm checking whether a two-colouring of the liberty points could work, and whether a checkerboard split would fail at even distances. I'm also weighing whether local liberty counts on neighbouring stones could stand in for some of the distinctness checks.

I'm now asking whether own and opponent moments could share channels if they're distinguished by sign or code offset instead of by separate slots. The receiver mask would still have to split the colours, so I'm checking whether that split costs less than the two extra channels I'm short.

I'm considering whether the mask channel that zeroes out own stones could also hold the own-colour moment at those points, since the gating would then need to tell a small moment value apart from the large accumulator value on non-own points.

I'm checking whether the own-stone accumulator can carry its own correction: if each step subtracts the own-stone value with a unit coefficient and adds a constant, the self-term cancels and the update reduces to a neighbour sum plus a fixed offset per layer.

I'm now trying a different form for the cross-colour term: a function g of the liberty point and the receiver that is zero only when they coincide and at least one elsewhere, written as a sum of separable products so it can be carried by moment channels.

I'm checking whether a distinctness matrix D(f, e) can be written as a sum of products of per-point channels, with zero on the diagonal and nonnegative elsewhere so the walk sums don't cancel. The full distinctness pattern needs many channels, so I'm looking at whether an approximation within a local neighbourhood is enough.

I'm testing whether a rank-2 distance-like matrix with zero diagonal and positive off-diagonal entries exists, using a cross-product form as a candidate.

I'm checking whether a rank-2 matrix can have zero diagonal and positive off-diagonal entries, since that would cut the channel count. The squared-distance form needs rank 3, and I'm wondering whether anything with rank 2 can meet those sign conditions.

I'm checking whether a rank-2 real matrix can have zero diagonal and positive off-diagonal entries. Writing D(f,e) as an inner product of row and column vectors, every other point has to lie strictly on one side of the line where its own entry vanishes, which I'm trying to see if three or more points can satisfy at once.

I'm checking whether a rank-2 matrix can work: each point's own entry has to vanish while every other point lies strictly on one side, and in a 2D cone only two points can be extreme. So rank 2 fails for three or more points, which pushes me toward rank 3, such as points on a circle lifted into a plane.

I'm checking whether a rank-2 matrix could still work if the sign condition only has to hold for a restricted set of liberty points near each candidate, with the rest allowed to be negative as long as they never co-occur. That would be the one way around the rank-3 minimum I just established.

I'm now weighing whether the exact distinctness check needs three moment channels per colour, which would bring the total to eight against a budget of six.

I'm checking whether the mask channel could also carry the own-colour moment, by letting the accumulator for each colour include a constant per stone so the gating reads from it directly. I'm concerned that this brings back the self-term corruption from before, but I'm going to trace through the update rule to see whether the error actually matters for the quadratic test.

I'm working out how the gate term changes the self-weight at occupied points: with the gate at (1+γ) and the threshold set to (1+γ)K, the own-stone update comes out as a scaled offset plus the neighbour sum.

I'm checking whether the mask could be folded into the accumulator itself: at a stone, the update becomes a weighted self-term plus the neighbour sum, but the threshold that identifies stones may drift as the self-term is reweighted, so I'm working out what minimum value the accumulator takes at those points.

I'm checking whether the gate can be folded into the accumulator itself: the self-weight becomes (1+γ), and the threshold has to be set so that stones still clear it. That would make the moment channels gate off the same value, but the gate constant grows roughly as K², so I'm weighing how much precision that costs over a few layers.

I'm checking whether the corruption term δ(A − A_min) in the B channel can be cancelled or bounded, since it adds a multiple of the mask-plus-offset signal into the moment that the quadratic test relies on.

I'm checking whether the masks can instead be carried by the accumulators themselves, so that the zeroed channels that gate the moment updates come from data the layer already holds.

I'm settling on the mask question by keeping two accumulator channels, one zero at own stones and one zero at opponent stones, since the final logit needs them anyway. That leaves four free channels per point per layer, and I'm weighing whether those should split as three opponent moments plus one own-colour channel or two and two.

Since exact separation of both colours looks out of reach with six channels, I'm weighing whether to keep exactness for one colour and approximate the other. Interleaving the colours across layers wouldn't help, because both still need channels at the same time. The walk reach matters too: propagation layers each add one hop, so a test at layer 4 sees two hops of neighbourhood.

I'm checking whether the test unit can work without a separate mask: at a neighbour that isn't the colour in question, the moment is zero, so the test term drops out and the unit stays silent, while a qualifying neighbour gives a positive value that flags the candidate.

I'm now looking at whether the test can be aggregated over a candidate's neighbours instead of built per stone, but summing over opponent neighbours would miss captures when only some of them are in atari.

I'm weighing a different route: letting each stone carry its own liberty code, so the test can be built per stone rather than per neighbour. A second option is to sum the per-neighbour test units at the candidate, one for each of its four neighbours, though that costs four units per colour at every point.

I'm now trying to reduce the test to two units per point, since the four per-direction units at every stone would use up almost all the channel budget.

I'm now considering whether the test can be placed on each edge between an empty point and a stone, using one unit that checks both orientations at once, since each edge has exactly one empty endpoint.

I'm checking the both-stones case: if both points are X stones, the sum still picks up their liberty terms, so the test may fire without a real atari, and I need to see whether that false positive can be ruled out.

I'm checking the case where one endpoint is an X stone and the other is an opponent stone: the X-moment at the opponent point is zero, so the test term vanishes and the pre-activation stays below zero because the stone's own term is penalised by β.

I'm noticing the test can be done with a single unit per edge per colour, since each edge has one empty endpoint and the moments zero out the wrong orientation. That gives 144 units per colour at the test layer, which fits within the channel budget.

Now I'm sizing the saturation step: the edge unit's output varies in magnitude, so I'm weighing whether a second layer of units is needed to turn it into a clean indicator.

I'm checking whether a signed mask channel could isolate the opponent-stone condition on each edge, but the sum of two zeroing channels is always large unless both endpoints are opponent stones, so that doesn't work. I'm now looking at a count-style moment instead, though it has the same magnitude problem.

I'm checking whether the raw test magnitude can be turned into a clean indicator: it scales with the walk-weighted liberty count, so it isn't a bounded signal. I'm considering a saturation step over each point's incident edges, using two units to clamp the sum to a 0-to-1 range.

I'm weighing whether the capture bonus needs to be bounded, since the magnitude varies with the walk-weighted count, and I'm concluding that an unbounded value is acceptable as long as it doesn't swamp more important priorities like saving a larger group.

I'm noting that the walk-weighted liberty count scales with group size near the liberty, so a capture and an escape could be compared by magnitude, though that comparison is rough. Since the walk weights grow with each layer, I'm planning to rescale the propagation so values stay moderate, and to use integer arithmetic so the moment-based test stays exact despite cancellation.

I'm checking that the bound on the integer arithmetic holds at two hops: with codes up to 80, the moment terms stay around 640,000, well under the float32 exactness limit, so the cancellation in the test should remain exact.

I'm checking the integer bounds for larger hop counts: at three hops the sums stay under the float32 exactness limit, but once the penalty multiplier grows large the product may exceed it, and I need to see whether that loss of precision could change the sign of the test.

I'm checking whether β can stay at 1: then the pre-activation is M0 − T − 0.5, so once T is at least 1 the unit is already off, and the large cancelling terms can be kept small enough for float32 to stay exact.

I'm checking whether β can be set so the penalty on the test term always outweighs M0, since T can be as small as 1 while M0 is large. I'm considering choosing the codes so distinct points differ enough that T has a guaranteed minimum, and separately computing T in a first layer before applying the penalty.

I'm checking whether computing the test term exactly in a separate step avoids float32 rounding, since the expression M0(1 − βc²) with an integer β keeps every weight exact.

I'm checking whether a large β would break exactness: products like 1000·6400 times a sizeable M0 land above float32's exact-integer range, so rounding could spoil the cancellation at T=0. Shrinking the codes so that nearby points stay distinct might keep every term small enough to stay exact.

I'm looking at whether a lattice colouring with a modulus around 25 can keep every pair of points within distance 5 distinct, so the codes stay small enough for exact float32 arithmetic.

I'm weighing whether the test can run at β = 1 if the codes are spaced so that any nonzero difference already pushes T above the largest M0, which would avoid the large products that break float32 exactness.

I'm working out that the codes can be spaced by a factor s so every difference yields a test term at least s², which lets the penalty stay at β = 1. With s around 11 the codes reach about 209, and the expanded pre-activation has terms up to roughly 4.4 million, which float32 still represents exactly.

I'm checking the walk-weight bounds on the propagation, since each hop sums over the occupied neighbours and the test at an edge depends on how large those weights can get.

I'm checking how far the liberty test reaches: with k hops, it only catches atari when every liberty of the group falls within that radius, so larger groups with distant liberties would be misjudged as capturable. Capture checks would need reach of about two or three hops, and I need to see what layer count that implies.

I'm considering whether the first hop of propagation can be folded into the base layer, but a single linear-plus-ReLU unit can't express the degree-three products needed. So the plan stays with base moments in the first layer, then one propagation hop per layer, which keeps the test reach at two hops.

I'm laying out the depth budget: base moments in the first layer, then one hop per layer through four hops, with the test in the sixth layer and the output summing those edge units directly. That pushes the reach to four hops, though the magnitudes grow with each layer.

I'm checking whether magnitudes stay exact at four hops if I use the lazy propagation, since the walk counts grow to about 625 and the codes would have to shrink to keep the products under float32's limit.

I'm checking whether the code spacing can be shrunk enough to keep the products under float32's exact-integer limit at four hops. With the half-weight walks the magnitudes reach about 1.3e7 at four fractional bits, which is past what float32 can hold exactly, so I need the test term to keep the penalty dominant without such large codes.

I'm checking whether a single test layer can stay exact: with β set to the max walk total, the coefficients grow as W²q², which at three hops already exceeds float32's exact-integer range. That confirms the single-layer test only works at two hops.

I'm noting that codes must stay distinct between the empty point and any stone within reach, and that with two hops of reach the liberties sit within distance four, so the spacing has to cover that range.

I'm working out whether a distance-4 colouring of the grid needs a modulus of about 13, and I'll check that by computing the offsets directly. The question is whether reach of two hops is enough, given that stones within two hops of n can have liberties up to distance four from the empty point.

I'm weighing whether a saturating output layer could keep the edge contributions bounded, since the test units would sit at the sixth layer and the logit sums them linearly.

I'm considering a test unit that outputs zero when the point is empty and no test fires, a constant when it is an empty-with-liberty case, and nothing when T is at least one. That would take a linear form like a plus b times M0 minus βT, with a at most zero so the empty case stays off, and b large enough that a single occupied point turns it on.

I'm still stuck on the masking problem: a single linear unit can't tell "one stone and one empty point" apart from "both occupied," since min isn't linear. I'm weighing whether an empty-point channel, zero on stones and large on empty points, could gate the edge test instead.

I'm now trying a scaled-down version of the M0 term, where a small multiple of the walk count is added to a constant before subtracting the penalty. The aim is a single unit that switches on for a lone opponent stone beside an empty point without needing two units per edge.

I'm settling on a single-unit edge test that stays exact, and checking how the output magnitude varies with M0 when a lone stone sits beside an empty point.

I'm checking the self-atari case next: for a candidate point e that captures nothing, the new group's liberties are the empty neighbours of e plus the group's remaining liberties.

I'm now turning to self-atari: a candidate that captures nothing is risky if it leaves its new group with one liberty, and the simplest case is when the empty point has only one empty neighbour.

I'm working through the self-atari check for a candidate with exactly one empty neighbour: a quartic test over that liberty and the candidate would work but is heavy. I'm also considering whether a per-neighbour variance-style quantity could replace it.

I'm now looking at self-atari for a candidate with no empty neighbours, where the merged group would only be safe if an adjacent own group still has a liberty other than the candidate.

I'm checking the self-atari case where the candidate has only one empty neighbour, and the test only catches it when every adjacent own group has no liberty other than the candidate itself. That covers placing a lone stone into a one-liberty spot, but it misses merges into a two-liberty group.

Working out escape moves now: a play at an empty point counts as an escape when it sits next to one of my own groups already in atari and leaves at least two empty neighbours, so the group gets a second liberty. Capturing an adjacent opponent group would also free up liberties, which I still need to account for.

I'm weighing how to handle ladders: an escape that looks fine on paper can still fail later, and reading it out would need search, which I'm leaving out for now. I'm also setting aside the "group in danger" check for groups that have only two liberties left.

I'm checking the layer-one budget: the moment channels for both colours take 486 units, which leaves too little room for the accumulators and masks I still need.

I'm weighing two ways to cut the budget: an exact three-moment test for just one colour, or a single shared channel where own-moments sit at own stones and opp-moments at opp stones. The shared channel would save units, but the receiver would then sum in neighbours of the other colour, so masking at the receiver becomes the hard part.

I'm considering whether the cross-colour contamination at an own stone, the sum over opponent neighbours' moments, could be cancelled or isolated with a separate scaling, rather than relying on a single linear test to separate them.

I'm now trying to cut the budget by letting the two colours share channels, but the sum at each stone would pick up contamination from opposite-colour neighbours, and I can't separate that linearly without a second channel that undoes the saving.

I'm checking whether two moment channels could carry enough to test each edge, since a rank-2 scheme would need every liberty's vector to sit on the outer side of the one I'm testing, and I'm looking at whether a circle-like placement in two dimensions could make that work.

I'm checking whether two-dimensional vectors could work: each point's vector would need to sit on the outer side of its neighbours', but in a plane some point always lands in the middle of its neighbourhood's spread. Unless the neighbourhood vectors span more than a half-plane, with the test normal chosen so every other vector stays strictly on one side while the point's own sits exactly on the boundary.

I'm weighing a hybrid where exact local checks handle the nearby cases and a propagated flag covers the rest. The question now is how often a liberty other than the candidate can be confirmed by looking only at points a short distance away.

I'm checking whether a single OR-propagated bit could replace the per-candidate moment test: for each stone t in the group, a liberty other than e exists if t is at least two steps from e and has a liberty at all, or if t sits right beside e and has two or more liberties. The open question is whether the stones next to e can be picked out without a separate propagation for every candidate.

I'm weighing whether to restrict the exact three-moment test to the opponent's colour for captures, since that's the most valuable case, and handle own groups with a cheaper check instead.

I'm now setting up the opponent-moment propagation, and the receivers have to be opponent stones only. I'm planning to gate that with a mask in the accumulator that is zero at opponent stones and large elsewhere.

I'm budgeting the layers now: opponent moments plus their accumulator take 324 units at layer one, leaving 188 for own-colour local features, and the later layers repeat that split for hop propagation and edge tests.

I'm weighing whether M2 can be dropped by encoding the three moment values in a smaller alphabet, such as {−1, 0, +1}, while still keeping enough separation for the local tests.

I'm looking at whether a single gated channel could replace the separate moment channels per colour, but the sender-side mask would need a product for each edge.

I'm weighing whether to restrict the exact moment test to opponent stones for captures, and handle my own groups with cheaper local rules, since the opponent's ataris matter most for material. Both sides still count, though, so a plain capture-first approach would leave my own groups exposed to being taken.

I'm weighing whether own-colour moments could be started in layer one, but I have no spare units there. Shortening the reach to one hop would free up layers for the own-colour test, though it would still need its own base computation.

I'm looking at whether the moment test can be made local: an opponent stone's liberty count could come from the moments of its neighbours, but encoding that for every point would need too many units. I'm considering whether a cheaper per-direction encoding of the same contributions is possible, though each direction alone would cost about twelve units per point, which is too expensive.

Sharing one channel between colours would work at each point, since a point is only ever one colour. The trouble is at the next layer, where a receiver reads its neighbours' shared values without knowing which colour they belong to.

I'm considering a scheme where the shared channel carries opponent moments scaled by a large factor, so an own stone can tell its opponent neighbours apart by magnitude, without needing to know each neighbour's colour.

I'm weighing a shared-channel layout with opposite-colour stones gated to zero, and considering whether a larger scale factor on opponent values could keep them distinguishable. Modular recovery would need many units, so exact moments for only one colour looks like the practical choice. I'm sketching a cheaper alternative: a two-class checkerboard code per colour, backed by exact local checks for the nearby cases.

I'm checking whether a two-class checkerboard code per colour could stand in for the exact moment test, with a local flag covering liberties that sit at even distance from the candidate.

I'm settling on a plan: start with a simpler network covering eyes, a positional prior, and local capture and escape rules, then test it in games to see how strong the opponents are before investing in deeper tactics. With 25 passes and 120 code runs available, I can iterate, so I'll build the infrastructure first.

I'm checking whether I can verify the network's behaviour in the sandbox, which I can't since running it there is ruled out. So correctness has to come from careful construction, with activations from a positions pass serving as the debugging check against ground-truth features computed in the sandbox.

I'm laying out the design with a symbolic circuit builder that records each unit's intended meaning, and I'll check activations against a separate Python implementation of the Go features, computed directly from the rules rather than from the weights.

I'm running through the policy terms now: a capture bonus for adjacent opponent groups in atari, an escape bonus for my own threatened groups, and penalties for self-atari and for filling my own eyes. I'm also weighing an optional atari bonus against opponent groups with two liberties. Next I'll work out how to get the group liberty properties this needs without blowing the unit budget.

I'm sketching a cheaper, roughly exact method for liberties, and then laying out a layer-by-layer budget: opponent moments and their accumulator in the first layer, with the remaining units going to own-colour features that are only approximate.

I'm trying to see whether the own-colour analysis can get by without a separate mask channel. The accumulator zeroes opponent stones for opponent gating, but own-colour propagation would need a mask that zeros own stones, and with only two channels per point that doesn't fit. So I'm considering whether the accumulator could be zero at every stone instead, leaving the moment channels to carry the colour distinction, though the receivers would still need to be filtered somehow.

I'm weighing whether an own-stone receiver could be filtered out using a count channel that stays at least K on own stones, so opponent contamination gets suppressed there.

I'm checking whether the opponent-moment term can be recovered with a count-plus-moment combination, since a plain zero-moment indicator might be missing when an opponent stone has no nearby liberties. That would require a separate per-stone count, which runs into the same unit budget problem.

I'm wondering whether the group masks can be dropped and receiver gating left out altogether, letting signals leak into empty and other-colour points, as long as they are cleared before the liberty test. Adding a phantom liberty to every count changes the moment sum by a term that isn't zero, so that trick doesn't work either.

I'm checking whether leakage through same-colour stones could contaminate neighbouring opponent groups within two hops, since a plain gate on empty points doesn't stop it.

I'm considering whether a second accumulator channel could handle own-stone gating, so that each colour gets its own mask and the two together fit in the two channels per point.

I'm weighing whether to spend the six available channels as three opponent moments and one own-colour moment, or as two each, and considering a third layout that uses the three masks differently.

I'm wondering whether the propagation step could be folded into the edge test itself, so that the test reads the neighbours' moments directly. The catch is that if the candidate point isn't one of the relevant stones, its own moment is zero while its neighbours might not be, so the gating would have to handle that case.

I'm checking whether leftover signal from neighbouring same-colour stones could spoil the edge test when the middle point is the other colour, since that case isn't obviously ruled out.

I'm checking whether skipping the gate on the final hop is safe for the edge test: if a neighbour of an empty point doesn't touch the candidate, its liberties can't match, so the test stays negative.

I'm checking the edge test with the ungated hop: if one point is a stone and the other is empty, the empty point's neighbours can only add liberty counts that differ from the stone's colour, so that side of the test stays at or below zero. The combined expression therefore still comes out negative when it should.

I'm checking whether the masked-atari worry actually holds: if the one liberty b carries garbage from a's own neighbours, the bracket M0(b) − T_a(b) could go negative and swallow the true signal. I'm working through what T_a(b) has to be to tell whether that term can really cancel the atari.

I'm now checking whether the ungated last hop breaks the two-orientation edge test: if a true atari point's garbage term goes negative, the fire gets masked. Since both orientations need separate units, that would mean 288 per colour, so I'm weighing a simpler approximate atari detector instead, keyed on whether a stone has exactly one liberty.

I'm checking whether a simple one-channel OR propagation, blocked at empty neighbours, could flag a stone that has a single liberty without the full moment machinery.

I'm now weighing a design where each colour gets a propagated "strong" flag for stones with at least two liberties, plus a second channel meant to track liberty-stones, though pure OR propagation can't avoid echoing back through the group, so I'm considering a walk-weighted count instead.

I'm checking whether the echo of the group's own liberty-stones can be cancelled by subtracting the count restricted to the neighbours of the candidate point, so that atari reduces to comparing two sums.

I'm counting the unit budget for both colours: reach 1 for own stones and reach 2 for opponent stones, each hop taking a full layer of 243 units, so the first layer alone comes to about 486 before the test layers.

I'm now working out whether the base layer can share channels across both colours, using a gated unit that picks out own-stone moments at one point and opponent-stone moments at another, but a single relu over a linear mix doesn't obviously give that selection.

I'm noticing that the colour-agnostic base moments can be shared across both colours at layer one, which saves channels. The snag is that the second-layer hop then mixes in stones of the other colour, so I'm considering whether the own-colour hop could be split per direction to keep the colours apart.

I'm now looking at a design that cuts the moment dimension by only counting liberties that actually sit within reach of the candidate point, leaving out the rest.

I'm weighing whether a compact encoding could replace the three-moment design, such as coding liberties by row and column separately, though that needs five moments in total and looks worse. I'm also considering a two-moment parabola trick on a restricted range of codes, which still seems to need three. I'm going to see whether the local structure of a single-liberty stone next to the candidate point can cut the number of moments further.

I'm noting that a per-stone quantity counting only the other liberties would be zero for single-liberty stones, but the group's remaining liberties sit beyond the stone itself. I'm concluding that exact tracking for both colours doesn't fit the per-layer budget, so I'll look at exact handling for one colour with an approximation for the other.

I'm working out whether the accumulators can serve as the own and empty masks: an indicator for own stones can come from thresholding the opponent-zeroed accumulator, so the empty flag is just what remains.

I'm now checking whether the accumulators can carry the full board state through every layer: if each one stays zero at its own colour's stones and at least a threshold everywhere else, then own and opponent indicators, and therefore empty points, can be recovered with about one unit per point.

I'm now turning to the own-colour side: those moments have to be built from indicators computed in an earlier layer, so the own pipeline runs a couple of layers behind the opponent one.

I'm sketching a layered layout where the opponent moments take the first three layers and are tested at layer four, with the accumulators handling the board state.

I'm sizing the layers now: 243 units for the opponent moments plus 162 for the two accumulators gives 405 in the first layer, leaving 107 spare. I'll check whether the second accumulator is really needed, since it's the one that tells own stones apart from empty points once the opponent propagation is running.

I'm laying out layers 3 through 6 and tallying the units: the opponent test takes 144 plus 162 for the carried masks and 162 for own indicators at layer 4, so the own pipeline runs too late for its test to fit before the output.

I'm settling on two moments per colour, a two-class code, which fits the six-channel budget exactly. The catch is that with only two classes, half the points share a class with the candidate point, so I'm looking for a class assignment that depends on more than position to avoid those collisions.

I'm now testing whether the class of a liberty could depend on local board structure, such as its number of empty neighbours, rather than just its position. A direction-based scheme fails, since a group's liberty stones see the same empty point from different sides.

I'm checking whether moment channels can double as gating masks, so I'm weighing a count-based encoding where one channel tracks walk counts to the stones of each colour, which would avoid spending extra units on separate mask layers.

I'm checking whether one channel per colour could serve as both the count and the gate, so the total stays under budget.

I'm checking whether a phantom liberty with a constant code could cancel the extra c² term, since the test center c shifts with each candidate point, so the per-walk cancellation would need the walk-count sum kept separately.

I'm checking whether the empty-point mask can come from the accumulator, with the own-colour contamination limited to the propagation structure itself: an opponent stone that receives own-colour moments can pass them on to neighbouring own stones.

I'm checking whether gating each colour's propagation with its own count channel could work, though that would add a constant offset to the test that I'd still need to cancel.

I'm checking whether a phantom liberty with a constant code could cancel the extra c² term, but the phantom contribution depends on the code of each stone, so it can't be made zero for every candidate. I'll set that idea aside and consider an exact approach instead.

I'm trying to cover the opponent side exactly with reach two, using three layers of moments and a test at layer four, while handling own-colour groups with cheaper local features in the spare units. I'm also checking whether an eye feature for own stones fits, since that would mean folding it into the accumulator at the second layer.

I'm checking the own-colour side now: a single stone in atari next to the candidate point can be treated as escaping if the point leaves it two or more empty neighbours. That's a local per-stone feature I can afford in the spare units. For the own-group gating, though, I don't have a channel that is zero at own stones and large at empty and opponent points, so that part is still open.

I'm weighing a single own-colour channel that propagates an OR-style "strong" flag, which would mark own-atari stones with one liberty unless a nearby pair of liberty-stones is found. It would be cheap, but it would misfire on longer groups whose liberty-stones are spread apart and each look like they have only one liberty.

I'm weighing the OR-flag idea against the self-atari case: a false atari reading just leads to an unnecessary extension, which is harmless, while a false self-atari reading could wrongly rule out a good cramped move, but only when the point has at most one empty neighbour, so the impact is small.

I'm now weighing whether the own-colour side can rely on a single OR-style propagated flag over reach two, plus a few local rules, instead of exact moments.

I'm weighing an exact treatment of captures against the cheaper OR-flag approximation, since a false capture reading would waste moves, and I'm checking how often the approximation misfires on two-stone groups with liberties at each end.

I'm checking whether a simple OR-style "strong" flag can tell when an own group still has a liberty, using a local rule: a stone counts as strong if it has two or more empty neighbours, or if it sits next to a same-colour stone that also has a liberty. I'm then propagating that flag across the group to see whether any stone in it qualifies.

I'm testing whether a line of three stones with liberties only at its two ends would be missed by the local rules, since the end stones aren't adjacent. That case is fairly common, so I'm adding a rule for stones two apart with a middle stone between them. Diagonal shapes mostly stay distinct as long as the corner point isn't the shared liberty, which keeps the rules local. Liberty stones further apart than that still need a separate check.

I'm checking whether a stone can learn it has a liberty-stone at graph distance three or more by propagating that information through the group, since such stones can't share a liberty and so the group must have at least two.

I'm now looking at whether propagating the liberty signal only through interior stones avoids the echo, since a liberty-stone's own neighbours would otherwise send its signal straight back to it.

I'm working out whether the echo of a stone's own signal back to itself can be computed exactly instead of avoided. For two hops, that echo is the count of closed lazy walks, which depends on how many same-colour neighbours the stone has.

I'm working out the echo term at two hops: a stone's own signal returns to it through its same-colour neighbours, which gives a local weight of 1 + deg(n). So I can subtract that exactly and keep only the contributions from other stones.

I'm checking whether the echo term can be subtracted exactly at two hops, so that the remaining sum counts only other liberty stones. For stones straight across the liberty point, there's no shared neighbour stone, so those contribute nothing.

I'm checking which diagonal stones within two hops of the candidate can count as liberty-stones with a liberty other than e, and summing their weights. The stones straight across e contribute nothing, since there's no two-step path through an empty point. For neighbours of e I'll use the stone's liberty count instead of the general test, so the total V'(n, e) only picks up other liberties.

I'm checking that the echo term is exact at two hops: subtracting the self-contribution leaves only other liberty stones within reach, and the only ones in the neighbourhood of the candidate's liberty that need correcting are the diagonal stones whose liberty count is exactly one. Once that correction is applied, the walk-weighted count tells me whether any other stone within two hops still has a liberty besides the candidate point.

I'm checking whether the same echo-subtraction trick holds at three hops, since the walk-weight count from a stone back to itself would then depend on more of the local shape.

I'm now checking the echo at three hops, where the closed walks back to a stone pick up extra terms from each same-colour neighbour, so the correction depends on the local shape again.

I'm working out the three-hop echo: the closed walks from a stone back to itself now count 1 + 3·deg(n), so that term is also local. For the diagonal stones beside the liberty point, the walk weight comes from the single shared corner stone, which gives 3 when that corner is occupied, so the correction stays local too.

I'm checking the straight-across stones at three hops: their only two-step route runs through the empty point e, so they contribute nothing. Diagonal stones beside e are the ones that still need their corner-stone weight checked.

I'm working out how the three-hop echo corrections apply to the diagonal stones beside the empty point, using the lazy-walk count for each two-step path through a corner stone.

I'm checking how the four-hop echo behaves, since walks of length four can return to a stone through different group shapes, so that term isn't a fixed local constant the way the two- and three-hop cases were.

I'm working out how the echo correction at three hops applies to the diagonal stones beside the empty point, and whether that correction stays local when a corner stone is involved.

I'm now checking whether the echo corrections stay local when a 2x2 block includes the corner stone m. Each block around the empty point e feeds a correction into the edges toward its two adjacent stones, depending on whether those stones and the corner are occupied and whether the stone's liberty count is one. I'm also considering whether the corner stone m, being diagonal to e, still counts as an other-liberty stone in V(n).

I'm working out the echo term at k=2 and k=3 hops, where the self-return weight becomes 1 + c·deg_X(n) with a small constant c, and checking how the corner-stone corrections depend on which stones of the 2x2 block around the empty point are occupied.

I'm checking the three-hop echo weight: closed walks back to a stone come to 1 + 3·deg, since two moves can be placed among three steps in three ways. The edge test would then combine the stone's liberty status, its same-colour neighbour count, and the corner correction.

I'm settling on a per-stone local offset: its self-echo weight plus large penalties when its liberty count isn't one or it isn't the colour being tested. The remaining difficulty is that the correction depends on which corner point e is, so a single offset per stone won't capture it.

I'm now trying the alternative of putting the test on each stone instead of each edge: for a stone with exactly one liberty, its correction depends only on which corner points lie beside that liberty and whether those corners are occupied.

I'm now writing the correction for each stone as a sum over its adjacent corner configurations, checking whether a stone with a single liberty has both the diagonal stone and the shared corner occupied.

I'm now weighing whether to drop the corner corrections altogether, since tracking them per stone would need eight separate patterns for each colour. The cost is missed ataris in L-shaped groups that wrap around the empty point, where every liberty is that one point.

I'm weighing whether missing the corner-correction case matters in practice, since those L-shaped groups around an empty point are rare in atari situations and are often dead anyway. For my own groups, though, overlooking the atari would mean failing to escape, which is the cost I need to judge.

I'm turning to the test layer now, where each edge from a candidate point to a neighbouring stone needs a check that the stone's walk-weighted count matches its self-echo term plus the corner correction.

I'm now setting the test as a comparison: the neighbour's walk-weighted liberty count should equal its self-echo weight, 1 + c·deg, plus the corner correction, and I'm considering whether a per-point feature can carry that self-echo term with a large penalty when the stone isn't the right colour or doesn't have exactly one liberty.

I'm trying to keep the gating signal non-negative, since relu outputs can't be negative. The fix is to shift the combined channel so a failed condition gives a clearly lower value rather than a negative one.

I'm settling on a single channel G'(n) that folds the self-echo weight and a large bonus gated by "n is in X with one liberty," so that the test fires only when the gate holds, and I'm checking how to build that gate from local features in an earlier layer.

I'm now checking whether a single unit can gate on both conditions, since a relu output can't go negative, so the stone-colour and single-liberty tests have to be folded into one nonnegative channel that stays low when either fails.

I'm working out whether the test can fold the self-echo weight and a large gate into one nonnegative channel, so that a stone not in the colour or with more than one liberty pushes the pre-activation well below zero. The gate needs to be a threshold on the liberty count, not an exact match, because a relu-style unit can only express one-sided conditions.

I'm checking whether a single unit can gate on both conditions: the stone must be the right colour and have exactly one liberty, so its self-echo term is the only thing left in the test. Since a stone with two liberties should drop the pre-activation by a large amount, I'm looking at a term that is zero for those cases and set to the self-echo weight otherwise.

I'm defining a gate for the test unit that is zero unless the stone has the right colour and exactly one liberty, using a large offset and a steep penalty on the liberty count so that the pre-activation goes negative when either condition fails.

I'm checking the case where the stone has two liberties: the cross-term gives 1 − β(V − w − corr) − B, which stays below zero as long as B exceeds 1, so the test stays off there.

I'm checking the cases where the test should stay off: a stone with two or more liberties gets pushed well below zero, and so does an empty or other-colour point. The unit should then output exactly 0 or 1, provided the walk-weighted count is an integer and the weight on the test is at least 1. I'm now wondering whether the walk-weighted values stay small enough to keep that precision.

I'm now looking at whether the corner correction can be folded into the per-stone channel instead, since for a stone with a single liberty that liberty is fixed by the stone's neighbourhood, which would make the correction a local function of each stone.

I'm checking whether the corner correction can be computed per edge instead, since the test unit for each edge would need its own version of that term, and that adds a lot of units per colour.

I'm checking the case where both endpoints of the edge are occupied by the same colour, to see whether the combined pre-activation still stays off when the stone at the far end has more than one liberty.

I'm checking the case where both endpoints of an edge are in the same colour. If both stones have exactly one liberty, each one's self-echo counts the other, so the combined value stays above the threshold and the test should stay off.

I'm checking the case where b is in the group but has no liberties of its own, so its gate pushes its term up to about 2B plus the weighted self-echo instead of dropping it, and I need to see whether the edge test still stays off there.

I'm checking the edge where b has no liberties of its own: its gate contributes a large term instead of cancelling, so the combined value needs to be re-examined to confirm the test still stays off.

I'm checking the edge case where b is in the group but has no liberties of its own, since then the U term should be zero instead of carrying the echo weight. I'm considering whether the formula for U needs a separate branch for zero liberties, so that the combined test still stays off there.

I'm checking the both-orientation case for an edge between two stones of the same colour. If the endpoint with no liberties of its own is treated as the empty end, its term has to stay bounded so the combined value doesn't push the test on, so I need a bound on that unit's output in terms of the liberty-free endpoint's weight.

I'm weighing a single nonnegative unit U for the stone at each end of an edge, and whether it can keep the test off when both endpoints are the same colour and one of them has no liberties. A fixed relu of the linear terms gives 2B plus the echo weight in that case, so I'm checking whether a different offset or an extra penalty on zero liberties can bring the combined value back below zero without breaking the single-liberty case.

I'm checking the edge case where one endpoint is a liberty stone and the other has no liberties of its own, since the single ReLU for U can't be V-shaped in the liberty count, so I need another way to keep the combined value off there.

I'm checking the both-orientation version for an edge between two stones: if the endpoint with no liberties is folded in, its term comes out to about 2B plus the echo weight, which pushes the pre-activation positive. So the shared unit only works if its term for a zero-liberty stone can be kept down, and I'm looking at whether a different base can do that.

I'm now checking whether a ReLU can build the "exactly one liberty" condition, since the liberty count needs to be pinned to 1 rather than just bounded from above.

I'm working out whether a single ReLU can isolate the case where a stone has exactly one liberty, since the threshold alone can't separate one from zero. I'm looking at computing x·L in a first step with a large offset, and then using a second unit to pick out the middle value.

I'm now working out whether a single ReLU can isolate the case where a stone has exactly one liberty, since the one-sided threshold only bounds the count from above. Given that a neighbour of an empty endpoint always has at least one liberty, a test that only needs L ≤ 1 might be enough for the single-orientation case, which would avoid the extra units.

I'm now testing whether an "a is not in X" gate can be read off the accumulator at a, by subtracting a term tied to a being an X stone, so the orientation with a as the empty end doesn't fire when both endpoints are X.

I'm considering whether a steeper slope on the liberty count could keep the shared unit off for a zero-liberty endpoint, and whether the Acc_X(a) value can serve as a gate that is large when a is not in the group and roughly flat when it is.

I'm now setting the accumulator offset large enough that its variation stays small compared to the gate's contribution, so the gate reads cleanly as "a is not in the group" and the shared unit stays off when it should.

I'm trying to get the accumulator offset large enough that the sign of the stone-is-in-group term dominates, which means the positive contributions from the zero-liberty endpoint, around 25 at most, have to be cancelled by it.

I'm weighing the single-unit edge test against a gate on whether one endpoint is empty, using an accumulator offset large enough to keep its variation small. The offset that makes this work costs about 288 units per colour, and the gate flips depending on which endpoint is checked, so the two orientations can't share one unit.

I'm checking the both-orientation trick for an edge between two same-colour stones, and whether U stays small enough to keep the test off when both endpoints have no liberties. If neither endpoint is a liberty stone, V(a) and V(b) could both be zero, so U(a) and U(b) each carrying 2B would push the pre-activation positive, and I need a fix for that case.

I'm trying to make U small when a stone has no liberties, since a single ReLU on the liberty count can't be non-monotone. I'm considering whether U could depend on the local input pattern instead, with a per-neighbour contribution so that the zero-liberty case stays off.

I'm seeing that a single ReLU on the liberty count can't handle L=0 and L=2 together while still firing at L=1: a linear threshold would need a negative slope and a high offset at once, which rules it out. So the both-orientation unit needs either two units or a different base condition, rather than trying to bend one threshold further.

I'm reconsidering the base signal: if each stone carries its liberty count instead of a yes/no flag, then a stone with two liberties automatically pushes the test past zero, so the explicit L=1 condition may not be needed.

I'm checking whether the liberty count can be used directly as the base signal, so that a stone with two liberties pushes the test past zero by itself. For the atari test, the stone's own liberty is subtracted, and any other liberties show up as extra positive weight, so the separate L=1 condition goes away. The gate can then be a single unit tied to whether the stone is in the group.

I'm now checking the both-orientation test in the case where both endpoints are the same colour and neither has any liberties, since the constant terms there could push the pre-activation positive. Switching the base signal from a yes/no flag to the liberty count itself looks like it removes the need for a separate L=1 condition altogether.

I'm seeing that the echo subtraction breaks for zero-liberty stones, since V(a) minus its own weight can go negative and push the test positive. To fix it, I'm gating the per-stone term on having at least one liberty, so an interior stone with no liberties contributes nothing to the both-orientation check.

I'm checking whether capping the L-dependent term at B3 ≤ β keeps the zero-liberty case off without breaking the one-liberty case, and I'm working out what that bound implies for the offset.

I'm testing whether the both-orientation unit can drop the L condition by subtracting the stone's own liberty from its weighted count, but that breaks for stones with no liberties. I'm now shaping the L-dependent term so the zero-liberty case stays off, and checking what that does to the one-liberty case.

I'm now checking the both-orientation case where two X stones are adjacent, tallying how their liberty counts feed into each other's sums, and seeing whether the per-stone term still keeps the test off.

I'm checking whether the both-orientation unit can fail on an edge where both endpoints are stones with no liberties. The arithmetic gives a positive pre-activation there, but the unit's output is only read at empty candidate points, so an edge between two stones never matters.

I'm checking whether the edge unit's output can ever reach a stone point's accumulator, since an accumulator at a stone must stay zero. The edge output only feeds the capture logits at its two endpoints, which are masked when both are stones, so the question is whether it leaks anywhere else.

I'm checking whether the edge unit's leftover output can break the zero-at-stone property of the accumulators. For the case where one endpoint is a stone of the moving colour and the other is of the opposing colour, I need to see what the pre-activation looks like there.

I'm checking the remaining cases: when only one endpoint is a stone of the moving colour and the other is empty, the test gives the right answer, and when both endpoints are empty the pre-activation stays at or below zero. The both-orientation unit works here without any extra liberty condition, since the echo term is subtracted directly.

I'm checking whether the atari test is exact: it should be zero exactly when the stone's own liberty and the other liberties of its group are accounted for. Writing out the difference, the own-stone term and the terms from the endpoint's neighbours each carry a factor of (L − 1), which is nonnegative for neighbours that are liberties...

I'm working out whether the correction term can be included cheaply, which matters for the case where a group sits on two adjacent sides of the empty point and connects through the corner.

I'm working out how to fold in the correction term: it's a sum over the perpendicular directions where the two 2x2 blocks on either side of the edge are both group stones.

I'm checking the leftover corner term: for each edge, the correction is the sum over the two 2x2 blocks that contain it, counting a block when its other edge is a same-colour pair. Both orientations give the same value, so the correction is symmetric, and the pair indicator for that other edge can be computed as a single per-edge AND feature.

I'm checking whether the corner correction can be folded into the edge unit using the channels already available at the test layer. The per-edge pair indicator would be too costly to carry as its own unit, so I'm looking at whether the accumulators at the test layer can supply that information instead.

I'm checking whether a non-lazy walk with odd/even separation could drop the diagonal contributions that cause the corner miss, since the two-move path through the corner is what gives those extra weights.

I'm checking the reach-1 variant next: with only direct neighbours counted, the diagonal corner term disappears, but it can only see small groups. Reach 2 covers larger groups yet brings back the corner-miss problem, and reach 3 would multiply those miscounts.

I'm working out the diagonal case with a three-step lazy walk: the only path from the empty point's neighbour to a diagonal stone runs through the corner stone, and it gets weight 3. Straight-across stones get no weight. So the leftover error is the corner configurations where stones sit on two adjacent sides of the empty point and connect through an L-shaped corner.

I'm checking whether the corner miss actually matters in practice: groups that wrap around their last liberty through a filled corner would go unnoticed, like a bent three-stone group with the empty point tucked inside the L. An eye surrounded on all four sides is already covered by the separate rule, but an empty point with three group stones and one friendly or edge neighbour is not.

I'm now asking whether a legal move at a point with no empty neighbours must be a capture, since a stone there would otherwise have zero liberties. If the point has no own neighbours with liberties either, the atari test may be unnecessary for those cases, and I'm considering whether the policy would still find these captures without the corner term.

I'm now considering a case the corner correction misses: an empty point with an empty neighbour and opponent stones forming an L around it, where a capture at that point would still be legal. I'm wondering whether the corner term can be approximated from the parallel edges of the two 2x2 blocks on either side of an edge.

I'm now considering a per-point pair indicator instead: for each horizontal edge, the corner correction would come from the horizontal-pair indicators on the rows directly above and below, and the vertical-pair indicators would cover the other orientation. That would mean carrying two extra indicators per point for each colour, which is the part I need to weigh.

I'm considering whether the corner correction could ride on the pair indicators already computed per point, but carrying those extra channels to the test layer looks too expensive. Another idea is to let the accumulated values at the test layer stand in for the correction, though I'm not sure that works. The correction only matters when the edge test would fire once the corner term is subtracted, so I'm checking whether that case can be detected directly.

I'm considering whether the threshold could allow for the missing correction by tolerating a small amount of extra signal, scaled by how many same-colour stones surround the empty point. The worry is that this allowance would also let through non-atari cases where the other liberties happen to be small, which would create false positives.

I'm weighing an allowance on the edge test that scales with a per-point count of nearly-enclosed blocks at the empty point, though that count would also include blocks that don't touch the edge in question, which risks false fires.

I'm weighing whether the allowance could scale with a per-point count of nearly-enclosed blocks at the empty point, but the stone endpoint's count would add to the allowance and make it too generous.

I'm simplifying the count: a per-point feature for the number of same-colour neighbours at each empty point, gated by emptiness, which costs one unit per colour per point. The correction only becomes nonzero once that count reaches two, so I'm weighing whether an allowance scaled by it could stand in for the missing corner term.

I'm weighing whether to let the test allow some slack scaled by how many same-colour neighbours the empty point has, since that could cover the missing corner term. The trade-off is between missing real captures in the L-shaped cases and firing falsely when the point has several friendly neighbours that don't form that shape, and I'm estimating which error is more costly.

Checking whether an allowance based on the empty point's same-colour neighbours could stand in for the corner term: for two friendly neighbours, the allowance is 1, but a stone two steps away along a straight line contributes exactly 1 as well, so the two cannot be told apart by magnitude. Weighted propagation might separate them, so I'm looking at that next.

I'm weighing whether the missing corner term could come from channels already available at the test layer. The idea is to reconstruct the parallel-edge indicators there, since carrying extra per-point pair channels up has already looked too costly.

I'm weighing whether the corner correction could be carried in some other way, such as folding the pair indicators into the propagation values, but that would still need extra channels per colour. Another option is to test from a different stone, though the corner point is not adjacent to the empty point, so that doesn't work directly.

I'm now considering whether the liberties shared between neighbouring stones could be divided up at the base level: each empty point would give a fraction of its weight to each adjacent stone of the same colour, so a shared liberty isn't counted twice. I'm working out what that split would do to the atari case where two stones of one colour meet at an empty point.

I'm considering whether the stone-to-liberty propagation could also flow through the candidate empty point itself, so the test would see that point as a node rather than relying on a corner correction.

I'm now checking whether the empty point itself can serve as a node in the propagation, so the test sees the liberty directly instead of needing a corner correction. The idea is to sum the propagated values of its same-colour neighbours and subtract the contributions that come from the stones adjacent to that same point.

I'm checking whether the sum over all of e's X-neighbours could work as a test, since it would capture the corner pairs in one number. The catch is that it only tells me whether every adjacent group has e as its only liberty, which is an AND across groups. If e touches two separate opposing groups and only one is in atari, the aggregate would miss the capture, so I'd need a per-group version instead.

I'm combining the two tests: a per-edge check without the corner correction, plus an aggregated check that includes it, which should cover the cases where every adjacent group is in atari. The open problem is that counting the corner blocks around a point isn't a simple threshold, so I'm looking at whether a carried per-point feature could stand in for it.

I'm trying to make the corner correction computable locally, since the per-edge test can't see the diagonal pairs. An aggregate over the empty point's X-neighbours would capture them, but building it needs pairwise products that would take too many AND units, so I'm looking for a cheaper way to get the same quantity.

I'm checking whether the empty point can act as a node in the propagation, summing the propagated values of its same-colour neighbours and subtracting the contributions that come from the stones adjacent to it. Then I'm weighing whether that aggregate could capture the corner pairs, though building it seems to need pairwise products that would take too many units.

I'm considering whether to accept missing the ⌐-shaped configurations altogether, since an aggregate test over the empty point's neighbours would need pairwise products that cost too many units.

I'm considering using the moment approach for the opponent colour alone: it's exact and avoids the corner problem, though it needs three channels. Alternatively, I could carry a second liberty-code moment alongside the count channel, but subtracting the empty point's own code only gives a clean signal if the codes are arranged so that its value is the minimum, which I can't assume.

I'm testing a combined condition: the first moment M1 minus the empty point's code times V should be zero when every liberty is that point, which catches the ⌐-shaped cases. It can misfire if other liberties' code deviations happen to cancel, though that should be rare.

I'm checking whether a one-sided test can work here: the difference M1 − c_e·V is zero in atari, but a two-sided test would need an absolute value, which isn't linear. I'm weighing whether large random-like codes make accidental cancellation rare enough to accept.

I'm weighing whether a two-sided check on the moment difference could be built from one-sided units, but that would cost two units per edge, which is over budget. I'm now looking at whether the V-test's own structure, where the corner term is small, could absorb the ⌐ case without that extra cost.

I'm checking whether the difference between the first moment and the empty point's code times the count could catch the ⌐-shaped cases. In atari that difference is zero, and with at most two other liberty instances I'm asking whether those could cancel out to zero by accident.

I'm now trying out a scheme where each empty point's code is chosen from a set with no three-term arithmetic progressions, so that two other liberties can't cancel the deviation from the point's own code by accident. The idea is that picking codes like numbers with only 0 and 1 digits in base 3 would make such coincidences impossible in the local region.

I'm checking the code-length bound for the codes: with the no-3-term-progression choice, the only way two liberties with at most two instances could cancel the deviation from the empty point's own code is if they are the point itself, so the moment difference separates the cases cleanly.

The one-sided threshold rejects only positive deviations, so negative ones from other liberties with lower codes would slip through. Covering both directions would take two units per edge, which is over budget, so I'm looking for a way to get the two-sided check without that cost.

I'm checking whether the correction term for the diagonal pairs could come from the first-moment channel instead: in atari with those corner stones, that channel equals the empty point's code times the count. I'm weighing whether to accept missing those shapes, and handling only the common eye case, where legality already covers it.

I'm wondering how often a group ends up in atari with its last liberty sitting in a corner-shaped spot, like an L-shape around the point or an internal concave gap. Single-point eyes are already covered by the legality rule, so the question is mostly about the larger shapes, such as a two-point eye space where filling one point changes what the other one means. That's getting complicated, so I'm stepping back to settle the overall architecture with the tools I have.

I'm laying out a six-channel layer plan for k=2: base own and opponent signals weighted by liberties, two accumulators that zero out at each colour, and two gated local-degree terms, leaving 26 units spare.

I'm considering folding the U channel into the Acc channel, since Acc is zero at the stones where U is nonzero, but that breaks the zero at the stones themselves.

I'm checking whether a threshold could mask the corner channel: a linear term can't stay flat over an interval and then change elsewhere, so that route is out. I'm also wondering whether the U values could be used directly in place of the masking.

I'm considering whether the echo term, the hop-2 self-walk count of 1 plus the number of same-colour neighbours, could be folded into the base signal itself by subtracting one from each liberty value, which would let the V channel carry the correction without a separate U channel.

I'm wondering whether the echo correction could simply be folded into the base signal, so the value channel carries it without a separate U channel. The catch is that subtracting one from every liberty value would also distort stones that aren't the empty point's neighbours, so I'm checking whether that side effect can be ignored.

I'm weighing whether the echo correction can be folded into a phantom-count base signal, but that would add an unknown walk-count term, so it doesn't work. That means the correction needs its own two channels for three layers, which fills the six-channel budget for the first three layers and leaves little room for eye detection, so I'm asking whether that could be folded into the accumulator channels.

I'm trying to fold the eye-fill penalty into the accumulator itself, subtracting a large term when the point is owned and a pattern-based term when all its neighbours are owned, since the AND can't be done without another unit.

I'm now testing whether the eye-fill penalty can be applied linearly to each own neighbour, but that would also punish useful connecting moves that sit next to several own stones.

I'm wondering whether the eye point, once its Acc value drops to zero, would be mistaken for an own stone by the gating, so it would pass value through and blur the group boundaries. Since it has no value of its own, it would only change the walk weights, not the liberty counts, but I'm still not sure that's harmless.

I'm checking whether the eye penalty could be applied from a later layer, but detecting an eye needs an all-neighbours-owned test, which isn't linear in the accumulator values. I'm also noting that letting an eye point pass as an own stone would distort the walk counts near it.

I'm checking whether the eye test could use the U channel directly: each own stone's U value sits between B and B plus a small margin, so a sum over the neighbours would clear a threshold only when every neighbour is own.

I'm looking at whether the eye test can be done with a threshold on the summed U values over a point's neighbours: if every neighbour is own, the sum reaches the full base level, but if one is missing, it falls short.

I'm checking whether the eye test can use the U channels with a large shared base B, so that the B cancels in the test and the echo correction still uses β. For points outside the set X, the pre-activation then simply comes out at 1 − B.

I'm checking the layer budget: six channels per point across L1 through L3 use all but 26 units. That leaves no room for eye detection in the early layers, so I'm pushing it to later layers, where it can come from the U channels.

I'm now trying to fit the eye detection into the fourth layer's budget, since the separate eye unit would push it past 512 units. The idea is to let the accumulator itself carry the nonlinear eye penalty, but a plain linear term can't express the all-neighbours test, so I'm working out how the relu could do it.

I'm checking whether the eye test can come from the U channels in a later layer, since the L4 budget is already nearly full. The problem is that Acc is relu'd, so a linear sum over neighbours can't give a clean threshold for "all neighbours owned" on its own.

I'm merging the two accumulators into one at layer 4, which frees up 81 units. That still leaves room for the 288 edge tests, the single accumulator, and the eye detection, for 450 total with 62 spare. The edge tests read only the V and U channels from layer 3, so they don't need the masks.

I'm now working out what the empty point's capture count needs: for each neighbour, whether it's an opponent stone in atari, and that depends on empty-neighbour counts I don't have yet, so I'll need to add those as local features in the later layers.

I'm looking at how to judge whether an empty point is a good capture or escape: the capture count comes from the opponent-atari tests at layer four, and an escape need from the own-group tests. I'm combining those with the number of empty neighbours into a single gated unit, and a self-atari penalty applies when the point has at most one empty neighbour, no capture, and no own neighbour group with spare liberties.

I'm working out the self-atari test for an empty point: it should count as bad when the point has at most one empty neighbour, captures nothing, and none of its own neighbouring groups has spare liberties, so I'm writing that condition as a single gated expression.

I'm now writing the self-atari test as a combination of three conditions: few empty neighbours, no capture, and every own neighbour already in atari. I'm working out how to express the "at most one empty neighbour" part, since a plain linear count can't gate it cleanly.

I'm working out a self-atari test for empty points: a ReLU-gated unit that should fire when the point has at most one empty neighbour and captures nothing. The E=0 case turns out to be illegal anyway, so its output magnitude doesn't matter, which lets me keep a single clipped expression.

I'm working out how to carry the per-point empty-neighbour count E(e) and own-neighbour count Nown(e) through the layers. A single channel can only hold one linear combination of them, and the self-atari unit needs them with different weights, so I'm considering two separate channels.

I'm checking whether a self-atari unit can depend on a single linear quantity, the number of opponent neighbours, since the empty-neighbour and own-neighbour counts only ever appear added together. If that works, the threshold is roughly a constant minus the opponent count plus the own-atari count, with the capture term subtracted. The case where no empty neighbours remain looks suspicious, so I'm testing whether it still fires there.

I'm checking whether a single combined expression can gate the self-atari case, and noting that it might fire in a few edge cases where the merged group still has room to breathe, which I'm treating as an acceptable approximation. I'm also starting to think about whether the escape bonus can be built without the separate empty-neighbour count, which seems to be the remaining obstacle.

I'm checking whether the U channels, which are zero at empty points, could also carry the neighbour counts there, so I wouldn't need two extra channels for those counts across the earlier layers.

I'm working out how a single relu unit can encode the neighbour counts at empty points: a count value when the point is empty, and the own-stone weight when it holds one of my stones, with the opponent case kept below the threshold.

I'm considering packing the neighbour counts into the accumulator channels at empty points, so the self-atari and escape tests can use them without extra channels, though the count would then shift the test magnitudes.

I'm checking whether Acc2 can hold both neighbour counts at empty points, but a single scalar can't yield Nopp and Nown separately with different coefficients. So I'm looking at whether Acc1 could carry the own-neighbour count instead.

I'm thinking about whether Acc1 can carry a per-point constant plus a scaled neighbour count, since a fixed prior is already known at each point and the count could be read off by subtracting it. That would let the self-atari test recover Nown linearly without spending extra channels, as long as the accumulator's logit role is undone at the end.

I'm checking whether the L4 layer can compute the self-atari logic directly inside the test units, since carrying both the own-neighbour and opponent-neighbour counts forward as separate channels would blow the channel budget (612 > 512).

I'm checking whether the own-neighbour and empty-neighbour counts can share the budget at L4, and it comes out 19 channels over, so I'm considering whether the own and opponent tests can share channels to close the gap.

I'm considering whether the two test units, one for own stones and one for opponent stones, could be merged into a single edge unit, since each edge only ever involves one of those colours or empty. If the neighbour is my stone, the opponent part drops out, so the combined pre-activation might collapse into one expression.

I'm now seeing that a merged, colour-agnostic test could detect whether a neighbour's group is in atari, since only one colour can sit at any neighbour. The catch is that this loses the colour information needed to tell captures from escapes, so I'm considering whether the test's output magnitude could encode colour instead.

I'm checking whether the two test outputs can be separated at the next layer if the opponent case fires at magnitude 2 and the own case at 1, since the summed signal would otherwise mix the two.

I'm now testing whether the self-atari unit can be expressed as a threshold on the empty-neighbour count plus the own-neighbour count, with a penalty for opponent neighbours.

I'm weighing a merged test that combines both colours into one unit, with capture encoded as a larger output magnitude, but the summed signal then can't separate own from opponent cleanly. The self-atari gate would need a separate channel for the opponent-neighbour count or the capture flag to switch off when a capture is available.

I'm now asking whether the own-stone test and opponent-stone test can be cheaper, perhaps by dropping the own test where it's only needed for escape, or by moving some per-point information into the spare channels. I'm also reconsidering whether the Acc1 and Acc2 masks are needed at all, since the U channels might gate the receiver instead, which would save those two channels.

I'm considering whether the echo term could be folded into the base signal by pre-subtracting it there, so the own-stone gate could then be a clean constant with no leftover residue.

I'm considering whether the echo term at the last hop could be pre-subtracted from the base signal, so that the self-contribution cancels out. The worry is that stones with no liberties would be clipped to zero and look like atari, though those stones never sit next to an empty point, so the test shouldn't read them.

I'm trying to hold the correction term at L3 without needing a separate per-stone channel. The idea is to subtract the self-echo from the base signal, so it would need the stone's status and its degree carried forward from L2.

I'm seeing that a channel R(s) = x(s)(Big − w(s,s)) can do double duty: it acts as the own-stone mask and also cancels the self-echo term at the next hop, so the gating and correction come from one channel.

I'm trying a variant where the echo is subtracted at the first hop as well, but the coefficient needed to cancel it varies with w, so a single linear channel can't handle both hops.

I'm testing whether subtracting a constant of 1 at each X stone on the first hop cancels the self-echo exactly, since each neighbourhood then sums to 1 + deg(n), matching w₂(n,n).

I'm checking whether clipping at zero breaks the exact cancellation: if a stone has no liberties nearby, subtracting 1 there would go negative and get cut off, so the −1 wouldn't propagate and the echo correction would come out wrong.

I'm checking whether clipping could break the cancellation for the neighbours of an atari-adjacent stone, and it looks safe: any stone s in the neighbourhood of n that sits next to an empty point has V1(s) of at least 1, so the −1 correction never gets clipped there.

I'm working out a hop-1 gate that subtracts 1 at each X stone, with the mask applied as Big·x(s), so the −1 corrections sum to exactly the self-echo w₂(n,n) at hop 2. I'm checking that relu clipping can't break this, since any stone in N[n] ∩ X already has V1 ≥ 1 when n sits next to an empty point.

I'm checking whether the test can read a stone's status at all, since if a neighbour isn't in the stone set it would look like it passes. I'd need a positive mask at the last layer, or to force the value to 1 at stones and test for exactly 1, but a two-sided test like that doesn't work with this setup.

I'm noticing the test needs a mask that is zero only at X stones, so non-X endpoints get pushed far negative and can't trigger a fire. Without such a mask, a non-X node with both endpoints at 0 would falsely read as atari, so I'm checking whether the Acc_X term can supply that gating.

I'm checking whether the negative masks can be combined so that exactly one endpoint has to be an X stone. Summing the two Acc_X values fails here, because an empty endpoint next to an X stone already pushes the sum above zero.

I'm checking whether the two-sided test can be made clean: the term built from the two endpoint masks would still leave a positive residue of up to γ·Amin for a non-X endpoint, which could push a not-atari case over the threshold.

I'm checking the constants now: a small γ forces ε to be at least 1/γ, and Amin has to stay below (β−1)/γ plus ε so the garbage term can't push a non-atari case over the threshold. Picking γ = 0.01, ε = 100, and Amin around 1000 seems to satisfy both bounds, with Amax still needing to sit below 2·Amin − ε.

I'm checking whether a mask built from the two endpoint counts can give a clean 0/1 output, since the test's magnitude varies between about 1 and 10 and the downstream counts depend on it. Normalising that with a second relu would cost two more units, so I'm leaning toward positive masks at the X stones instead, which should avoid the problem.

I'm realising the positive mask U, set to a constant at X stones, can also serve as the propagation gate: it gives 0 at X receivers and a large negative value elsewhere. That would remove the need for a separate negative Acc channel and leave the echo handled entirely by the −1 correction at hop 1.

I'm settling on the test using two positive mask channels, one per colour, with a constant at X stones, so the same channel gates propagation and the test can read a single X endpoint cleanly.

I'm settling the hop-2 correction and the final test: the positive mask M = x·B can gate propagation directly, so the echo is removed by the −1 at hop 1 and no separate accumulator channel is needed.

I'm checking the layer-by-layer bookkeeping for the read case: a stone n next to an empty point should have its hop-2 value reduced by exactly its self-echo, and the −1 at hop 1 should survive clipping for each neighbour in N[n] ∩ X.

I'm checking that the two-hop echo subtraction works out: each X neighbour m of n has V1(m) ≥ 1 because n itself is one of its neighbours, so the −1 is never clipped. That makes V2'(n) equal to V2(n) minus the self-echo w₂(n,n), and since the remainder is a sum of non-negative terms, the test stays valid.

I'm checking whether the cancellation holds in general: the leftover sum should only vanish if L(n) = 1 and every other stone within two hops has L = 0, so I need to pin down what the clipped remainder actually requires.



---

I need to pick up from here without knowing what came before, so I'll first check what's in the sandbox, starting with /work and whatever was saved, before deciding anything else.

