I need to hand-design a policy network for 9x9 Go without any training, just setting weights by design to beat weak RL-trained opponents. I'm thinking through strategy basics: avoiding premature passes while being careful not to fill in my own eyes since the game ends after 162 moves or two consecutive passes.

There's also the positional superko rule — repeating a whole-board position causes an immediate loss, and since it's not masked like simple ko, this is a real risk during captures and recaptures in the endgame, though probably rare. Beyond that, I need the policy to prioritize capturing opponent groups in atari, escaping my own atari situations, avoiding self-atari, and favoring territory-expanding moves centered early on, all within the constraints of a 6-layer, 512-unit ReLU MLP.

With only 81 points and 512 units per layer, I'm tight on capacity — roughly 6 units per point — and computing liberty counts for groups requires non-local connectivity that's hard to propagate through limited layers. I'm budgeting units carefully: using input planes for own/opponent/empty stones, and reserving enough capacity in each layer to carry the board state forward via ReLU identity passes while leaving room to compute atari and capture features.

For the actual playing strategy, I'm prioritizing heuristics that matter most against weak opponents: capturing stones in atari, saving own stones from atari by extending, avoiding filling true eyes, avoiding self-atari, and favoring contact or central moves early. I'm also reasoning through endgame scoring — since area scoring counts stones plus territory, playing inside your own territory doesn't cost points the way it would under strict territory scoring, which changes how passing and filling dame should be handled.

Since games likely run to the 162-move limit, I need my policy to avoid filling its own eyes and prefer passing once only self-destructive moves remain, meaning pass should score higher than eye-filling moves in the logits. I'm thinking through how to scale logits so choices are near-deterministic while still allowing some useful randomness, and now considering how to compute group-level liberty features within a compact network.

For capture detection, a move at point p captures when p is empty, an adjacent opponent stone q exists, and q's entire group has exactly that one liberty. Single stones and small groups can be read from local patterns, but larger groups require propagating liberty information across the chain, so I'm considering an iterative message-passing scheme across the network's layers to approximate group liberty counts and atari status.

Given the tight unit budget per point, I'm figuring out how to encode board state efficiently — own/opp stone indicators can be carried with just two units since the empty indicator is a linear combination of those, letting later layers derive it without extra cost.

I'm shifting focus to what the final layer actually needs: rather than carrying the full board, it just needs per-point features that linearly predict move logits. I'm sketching out candidate features like capture value (whether a point is an opponent group's last liberty), save value (whether playing there rescues an own group in atari with enough resulting liberties), self-atari risk, eye-filling penalty, and a general positional prior.

Other factors to weigh: rewarding moves that put an opponent group into atari, favoring center/third-line placements over edges early on, and reserving pass as a fallback only when remaining options are all eye-fills or self-atari. I'm also considering the tricky endgame case where passing versus playing in contested territory needs careful judgment, since stones played in opponent territory may just die.

With only 6 layers, information can only propagate about 5 points along a chain, so detecting atari for larger groups becomes approximate at best — most 9x9 groups are small enough that this should still work reasonably well. I'm thinking about a local liberty count per stone and whether distinct liberties across a group sum to one, which would signal atari, and considering a message-passing scheme to track this.

Actually, I realize atari just means exactly one liberty, so I need a cleaner way to detect it: a stone's group is in atari if, counting distinct empty neighbors across the whole group, there's exactly one. The tricky part is avoiding double-counting a liberty shared by multiple stones in the same group, so I need some propagation scheme that tracks distinctness rather than raw counts.

Maybe instead I should count liberties per empty point rather than per stone, assigning each liberty a unique ID and propagating the max and min IDs seen across a group's stones — since a group has at least two distinct liberties exactly when its max and min differ, and max/min are expressible via ReLU operations that fit the network architecture.

I'm worried about the gating needed to restrict propagation to same-color neighbors, and whether computing a true max over five values (self plus four neighbors) is even feasible within one layer given the limited units available per point — it seems to require more depth or more carried values than the budget allows.

Given I only have six nonlinear layers total before the final linear logits, I'm stepping back to think about which heuristics actually matter most for beating weak RL policies — avoiding self-atari and eye-filling mistakes late in the game seem like the highest priority.

Since the RL opponents only beat random moves 79-98% of the time, they're genuinely weak, so even a modest heuristic emphasizing capturing atari'd groups, saving one's own atari'd groups, and basic shape play should win often. That points me toward cheap, local radius-limited features rather than exact global liberty counting, maybe with pattern-based exact counting only for very small groups.

I'm now trying to find an exact way to detect atari without full liberty tracking — checking that every stone in a group has all its empty neighbors mapped to the same single liberty point, which could work through some ID-propagation or distinctness trick across the group's stones.

Min/max propagation is idempotent so it survives repeated updates along a group, unlike naive sums which would double-count paths. So I'm settling on max (and min) ID propagation as the right tool, and now I'm weighing the cost of updating a single max value per point gated by neighbor color.

Computing a true max of several values with just one ReLU layer doesn't actually work linearly—max(a,b) needs a relu trick that only approximates two values at a time, and combining more than two neighbors risks overcounting and overshooting the real maximum, which would break the atari equality test. I'm now considering whether an OR-based propagation scheme avoids this overestimation problem.

I realize this is way too expensive: 7 bits for 81 points means 14 OR-propagations at 2 units each, roughly 28 units per point. I'm now wondering if we even need globally unique IDs — maybe just distinguishing liberties within a single group matters, though groups can span the whole board so a simple local coloring scheme won't guarantee distinct colors for distinct liberties in general.

Then I try a coloring c(p) = (row + 2*col) mod 5, which acts like a perfect code where each point and its neighbors get distinct colors, and even distance-2 points stay distinguishable. Liberties farther apart could still collide in color, but then the group likely has extra liberties anyway, so checking atari could work by propagating an OR per color class across the five colors.

But the unit budget is way too tight — five OR propagations per point costs too much, so I need a cheaper approach. I'm considering restricting exact liberty computation to small groups using local features, and approximating for larger groups, starting with cheap per-point counts like own/opponent neighbor tallies.

A single stone is captured when its last liberty is filled, which is easy to detect locally, but general group-level liberty counting is harder since it depends on shared liberties across the whole group. I'm trying to figure out a propagation-based way to check whether a group's only liberty is the point being played, without fully tracking liberty sets per stone.

Another way to get max could use threshold-counting, but that's expensive since it needs a separate unit for each possible value. I'm wondering if not every layer actually needs to do full propagation — maybe the budget can be spread differently across the six available layers.

Stepping back, I recall the pseudo-liberty trick from Go engines: track sum of liberty coordinates, sum of squares, and stone count for a group, where a group is in atari exactly when sum² equals count times sum-of-squares — this works because pseudo-liberties (which double-count shared liberties) still satisfy that identity. The catch is this requires knowing group membership first, and propagating these sums naively would double-count across multiple paths unless something constrains it.

Maybe instead of exact group tracking, I could limit exact computation to small groups (size 2-3), since those dominate tactical situations in weaker games. Given these opponents are RL policies trained via self-play that beat random only 79-98% of the time, they're likely weak and shape-focused, so games will be messy with frequent captures — a policy that reliably captures atari'd groups and saves its own should dominate.

Now I'm working through max-propagation with careful unit accounting: tracking own/opponent/empty per point plus max and min liberty counts, where min can be encoded as a large constant minus the value to keep everything expressible with ReLUs. The update needs each point's value to become the max over same-color neighbors, which for two values is achievable with a couple of units via a+relu(b-a), but for up to five neighbors in one layer this isn't exactly representable so simply.

I realize that max of three or more numbers is a known case requiring at least two hidden layers to represent exactly, so a single layer can only do pairwise max cleanly. This suggests alternating propagation along one direction per layer instead -- handling row-neighbors in one layer and column-neighbors in the next, spreading the full neighborhood max across multiple passes.

Alternatively, I consider a sum-of-ReLUs approximation that would overestimate the true max whenever two or more neighbors exceed the center value, which risks falsely concluding a group has two liberties when it actually has only one, breaking the atari test.

Maybe instead of max-ID propagation I should try an OR-style scheme over liberty classes, summing same-color neighbor values unclamped since growth is bounded (at most 5x per layer, so values stay manageable), letting positivity itself encode "true" and clamping only at the end — that could cost just one unit per point per class.

But the gating problem resurfaces: I can't gate individual neighbor terms inside a single sum before applying one ReLU, since the multiplicative same-color mask needs to apply per term, not to the aggregate. The fix might be to split propagation by color entirely — define an "own-stone" value that's nonzero only on stones of that color, then gate the whole sum against a bias tied to whether the center point itself is that color.

This brings the total down to only two gated OR-propagation channels per class (own and opponent), though with five liberty-count classes that's still ten units per point total. I start wondering whether a parity-based coloring trick (checkerboard based on row+column parity) could cut the classes down, but notice this breaks because two liberties of the same stone often share the same parity, not opposite.

So instead I shift focus to what's tactically essential: detecting captures, escapes from atari, and avoiding self-atari. For capturing, I want to check at an empty point whether an adjacent opponent group has that point as its only liberty — meaning every liberty-adjacency in that group routes through this single point.

Using a 5-coloring scheme based on board position, I could check per stone whether the group has a liberty in a different color class than the candidate point, since if all remaining liberties share that class they'd be unusually far away, making true ataris easier to isolate approximately.

I'm now trying max-based propagation across neighbors to combine information in one layer, but combining east, west, and self values genuinely needs more than one layer—so with only six layers of depth available, I'd need two layers per propagation step just to merge three directional values correctly.

Switching approaches, I consider encoding liberty identity as a large base raised to the liberty's ID, then summing these values along walks through the group. After several propagation steps, each stone's accumulated sum reflects the count of walks reaching each distinct liberty, weighted by that base—so if all reachable liberties share the same ID, the sum collapses to a single multiple of that base power, while multiple distinct liberties produce a sum spanning different magnitudes.

I realize testing whether that sum is a clean single-term power isn't something ReLU-based operations can easily check. I try alternative ideas like using row/column IDs with max/min propagation, but these hit the same ambiguity problems, so I start wondering whether an overestimating approach could still work if the overestimation never actually causes harm in practice, and whether a local radius-2 neighborhood check around a capture point might be sufficient instead.

Looking at it concretely: for single-stone captures, I just need the neighboring opponent stone to have no other empty neighbors and no further opponent neighbors. For two-stone groups I'd need to check one more stone out, and three-stone groups need one more again — so the required radius scales with group size, which with six layers only gets me so far since the real difficulty is the combinatorial explosion of possible local patterns, not just distance.

A cheaper angle: a group has at least two liberties if any single stone in it has two empty neighbors (an OR-propagatable property, costing about one unit per point per step), or if two stones in the group have differing single-liberty sets. The first case is easy to detect; the second — distinguishing differing singleton liberties across stones — is the expensive part, though most atari situations seem to reduce to simpler two-stone cases.

I'm trying a weighting trick: for each empty point adjacent to the group, divide its contribution by how many group stones touch it, so summing these fractional contributions across all stone-liberty adjacencies should recover the exact count of distinct liberties, as long as I can accurately determine how many same-color stones border each empty point.

But this approximation breaks when two separate groups of the same color share a liberty—like diagonal stones at (0,0) and (1,1) sharing points (0,1) and (1,0)—since each group only gets half-credit for that shared point, causing an undercount and a false sense that the group has fewer liberties than it actually does.

This also reveals that summing liberties across a group via message passing is fundamentally flawed, since multiple paths through the group structure would double-count shared points unless the group happens to be a simple tree. That leaves idempotent aggregations like OR, max, or min as the only viable options for local propagation, with OR being by far the cheapest since it only costs one unit per step per color-feature per point—so I'm now thinking about which OR-based features could reliably signal "at least two liberties" for a group.

I'm exploring an approach using liberty IDs split into bits: for each bit, track whether any liberty has that bit set to 1 or to 0, and a group has ≥2 distinct liberties if some bit shows both values present. But doing this across the full ID range would need too many units per layer. I'm now considering whether liberties really need globally unique IDs, or whether a coloring scheme with enough distance-based classes could guarantee any two liberties of an at-risk group get different labels.

Checking the unit cost, needing 5 classes per color and clamping ORs to detect "≥2 classes present" still ends up far too expensive. I'm now reconsidering whether avoiding plain OR propagation in favor of some kind of binary encoding could cut the cost down.

With a 4-class coloring, distance-2 distinctness fails since each point plus its 4 neighbors forces 5 distinct colors in Z². But maybe I don't actually need full distance-2 distinctness — two liberties of a single stone could sit at distance 1, straight-line distance 2 (opposite sides), or diagonal distance 2, so I should reconsider exactly which liberty configurations matter.

Thinking about the single-stone case: all four neighbors of a point need distinct colors, so I need at least four classes. I try a parity-based coloring using row and column mod 2, but that fails since opposite neighbors share the same parity pair.

I then try other modular formulas, but each one runs into the same obstruction: offsets of +1/-1 and some other step ±a can't all stay distinct mod 4 unless a is even, yet evenness collides elsewhere. This suggests linear mod-4 colorings can't work, so I start wondering if this is related to the injective chromatic number of the grid graph.

Switching approach, I recognize that an injective coloring requires vertices sharing a common neighbor to get different colors, which in Z^2 means points at distance exactly 2 must differ. That distance-2 graph splits by parity into two pieces, and each piece turns out to be isomorphic to a king's-move graph after a 45-degree rotation—meaning its chromatic number is 4.

Four colors per parity class suffice, and since even and odd points never share a common neighbor, I can reuse the same 4 colors across both classes, giving an injective coloring with 4 colors overall. But I'm now worried this doesn't guarantee the liveness property I actually need: adjacent liberties (distance 1 apart) aren't required to differ under this scheme, so a stone group with exactly two adjacent liberties could end up with both liberties colored the same, creating a false atari signal.

Thinking about the unit count for a richer 5-class version, each point would need 10 units (two colors times five classes), totaling 810 units — too many to fit within a 512 budget. I consider only tracking liberty classes for opponent groups to save space, but that sacrifices seeing own-group atari, which matters too. I also consider encoding liberties as a weighted sum using distinct powers of ten per class, so a single scalar could represent all the walk information.

But the counts per class can grow large with propagation depth, so the digit encoding overflows and loses separability — and even with a big enough base, extracting which digits are nonzero needs nonlinear thresholding across many digits. Checking the numbers, with base around 3125 raised to the fifth power, the magnitude exceeds what float32's 24-bit mantissa can represent exactly, so this approach isn't numerically viable. I start reconsidering scalar max-propagation with some tolerance for overestimation instead.

I realize I'd need to gate the "not own" case properly, but counting it up, this needs around 20 units per point per color just for one layer of exact liberty propagation — way beyond budget. Exact group liberty computation isn't feasible here, so I need to fall back on heuristics instead.

One idea is a single positive "score" unit per point that aggregates capture, save, self-atari, eye, and positional signals into one relu'd value, letting the final logits be linear combinations of these. I could also try pipelining liberty computation across depth, processing group OR-features in stages — or just settle for a cruder liberty-count approximation that ignores distinctness.

Since most captures in weak play involve small 1-3 stone groups, detecting atari locally within radius 1-2 should handle the bulk of cases, even though large-group captures are rarer but more decisive. Given that these RL opponents are policy nets with no search that barely beat random, a well-designed heuristic net should outperform them comfortably — so I should prioritize cheap, local, computable features first, starting with eye detection.

For liberty counting, I'm exploring whether cheap OR-propagated conditions can approximate the "group has ≥2 liberties" test without full connected-component analysis, using sufficient conditions like a single stone having 2+ empty neighbors.

I'm testing a parity-based heuristic: checking if a group's liberties span both even and odd parity, trying examples like vertical dominoes and L-shapes to find edge cases where this fails to distinguish distinct liberties.

But a straight three-stone line reveals a counterexample: if both end liberties share the same parity while the middle stone has none, the parity check would wrongly collapse them into a single liberty, producing a false atari reading.

Thinking it through, these false positives aren't actually dangerous — playing on a liberty of an opponent's two-liberty group just creates an atari anyway, which is a reasonable move, and the same logic applies defensively. So the real risk isn't the atari detection itself but how it interacts with the other clamp conditions, and the parity heuristic only breaks down specifically when all liberties share parity and no single stone has two empty neighbors.

Now I'm checking whether I can reduce the three OR-propagated features per color per point down further, noticing that a stone's empty neighbors always have opposite parity to the stone itself, which might let me simplify the even/odd liberty tracking.

That's too many units including opponent stones, so I'm trying to cut it down to two OR channels per color instead of three. I'm testing whether combining X = OR of (≥2 liberties or even-stone-with-liberty) and Y = OR of (≥2 liberties or odd-stone-with-liberty) lets me recover the ≥2-liberties signal just by checking X∧Y, walking through the case where a true two-liberty stone exists.

This seems to work, bringing it down to 4 units per point across both colors, but I'm now realizing the own-stone X value can be zero even for stones that simply haven't received a liberty signal yet, which breaks my ability to distinguish "no liberty" from "not yet reached."

So I need to carry own/opp separately too, pushing the total to 486 units, leaving only 26 units of margin for the propagation layers and final logit computation — tight but possibly workable. I'm double-checking the OR-propagation update rule restricted to own-colored neighborhoods to make sure it correctly zeroes out contributions from non-own neighbors.

For the clamping constant, since values can grow roughly 5x per step and start at most at 1, after five steps they're bounded by around 3125, so a constant like 10,000 should comfortably exceed any possible sum while staying well within float32 precision. Since the final X and Y values are nonnegative integer walk counts, I can build an indicator for "X ≥ 1" using two ReLU units, then combine the two indicators with another ReLU-based AND to get the final existence test.

I'm working out the base liberty indicators X0 and Y0 for each stone, depending on whether its position has even or odd parity, based on the count of empty neighboring points combined with ownership.

Beyond that, I'm mapping out how the propagation steps build up group liberty counts over multiple hops, and how the final layer needs to detect capture situations where an adjacent opponent group sits in atari.

For escape and self-atari detection, I'm approximating the liberty count a group would have after placing a stone by combining empty neighbors of the point with liberties of adjacent own groups — treating two-or-more total liberties as a successful escape, which is a reasonable approximation even though it won't catch deeper laddering.

I'm settling on an approximation that treats the remaining liberty count loosely, accepting it might miss some self-atari cases. Then I'm defining the core per-point features: capture detection (checking if an empty point neighbors an opponent stone in atari), save detection (checking if an empty point helps rescue an own stone in atari via escape routes), and starting to define self-atari detection for a given point.

Continuing with the feature definitions, I'm working out eye detection (points fully surrounded by own stones, which should be strongly penalized since filling eyes is bad unless it's reducing down to two eyes), and noting that true atari-detection on opponent groups with exactly two liberties isn't easily testable with the current approach, so I may need to skip or approximate it. I'm also considering adding a positional prior per point before moving on to think about how the final output layer should combine all these signals.

Still working through the atari arithmetic...

I realize clamping X and Y to binary values would need four relu units per color per point to reconstruct the threshold, which seems like too much machinery — so I'm reconsidering whether X and Y can stay unclamped, but then relu(X+Y−1) misfires when X alone is already 2 or more.

Instead, I'm trying a different representation: track Z as "no signal yet" rather than the OR signal directly. Propagating Z via AND over a neighborhood is just relu(sum − (k−1)), which is a single ReLU that's exactly binary and idempotent — but I still need to figure out how to treat neighbors that aren't the right color as neutral "1" inputs to that AND so they don't block propagation.

Still deriving the neutral-value trick...

Then combining the two no-signal bits for X and Y, safe2 (meaning safe from both colors' atari) is exactly own minus Wx minus Wy through a relu, giving a clean binary result, and atari becomes own minus safe2 linearly.

Now I'm counting units per propagation layer: 6 per point (own, opp, plus Wx/Wy for each color) times 81 points gives 486, leaving 26 units spare. For h1 specifically, I need to compute the initial liberty signal directly from the input, checking parity conditions on empty neighbor counts to determine whether even or odd stones have a local liberty.

Working out the exact relu combination that distinguishes nEmpty=0 from nEmpty≥1 for even stones, I verify it holds, then try to find the right relu formula for odd stones needing a true result when nEmpty is 0 or 1 but false otherwise — my first attempt using a 2·own − nEmpty expression doesn't quite give the right output at nEmpty=1.

Applying that trick costs extra units though, so I'm working out the per-point layer-1 budget: own/opp take 2 units, and each color's zero/one-empty conditions need roughly 3 units apiece depending on parity, landing around 8 units per point total.

I'm reconsidering the design instead of brute-forcing through parity-based ORs — maybe the two-liberty condition for a single same-parity stone can be pushed to layer 2, or I need an entirely different split of neighbor classes that distinguishes every pair, which seems hard to achieve cleanly.

I'm trying to define sources by position: an even stone counts as a source if it has at least one empty neighbor, while an odd stone counts if it has at least two. The even case works cleanly with one relu unit, but the odd case needs two since "nE ≤ 1" isn't a single threshold — so I'm exploring whether switching the source definition to emit from empty points rather than stones could simplify this.

I'm wondering if I should shift propagation to layer 2 onward and use layer 1 purely for richer local features, giving roughly 512 units total, about 6.3 per point. I'm exploring whether own/opp can be recovered linearly by splitting into "own with signal present" and "own without signal" components, since their sum reconstructs own without needing a nonlinear combination.

Checking the counts, this split still needs three units per color per axis, same as before, but now the propagation rule for the "with signal" unit becomes a subtraction in the next layer rather than something nonlinear, which could simplify the circuit even if it doesn't reduce unit count.

That layer-1 approach with 8 units per point overshoots the budget at 648 vs 512 available, so I need a cheaper encoding. I'm trying a self-term trick where odd stones get a value distinguishing isolated versus connected states, hoping to fold that into layer 2's AND computation more efficiently.

But if the self term can take the value 2, the AND output could also hit 2, which breaks the binary assumption and compounds error through later layers — I need to find a way to clamp this or restructure the derivation entirely.

Let me try instead defining sources by parity directly: even-stone-with-liberty for one color, odd-stone-with-liberty for the other, and treating the two-liberty detection as a local, non-propagated feature. The question is whether skipping propagation causes false atari readings when a group's liberties all share the same parity — checking the simple cases like single stones (always fine, since the liberty-parity detection is local) and two-stone groups of differing parity, where the liberties naturally differ in parity too.

For larger chains, like a three-stone line whose only liberties sit beside the middle stone, I realize the two-liberty signal does need to propagate outward so the end stones know they're safe. That pushes me toward folding the two-liberty condition directly into both the even and odd source definitions, so each point tracks whether it's an even stone with any liberty or any stone with two-plus liberties, and symmetrically for odd — meaning I need separate weight terms for "zero liberties" versus "at most one liberty," using different unit costs for each.

Working through the unit budget, I settle on a layer-one layout per point with six values: own, opponent, and then two derived quantities for each color capturing the zero-liberty and at-most-one-liberty signals via simple relu combinations, since the true "own" indicator can't be reconstructed once liberties reach two or more. That totals 486 units with 26 to spare, and layer two mirrors this structure to carry out the first propagation step, folding in the self-term from layer one as a linear combination of those same quantities.

Layers three and four handle the second and third propagation steps across neighborhoods. For layer five I need the safe2 signal per color plus own/opp state, since atari detection depends on comparing stone presence against the accumulated safe2 term, and capture detection (empty point adjacent to an opponent stone in atari) needs that safe2_opp value available one layer earlier so the final layer can compute it via a relu over neighbor differences.

I'm extending this to distance-4 reachability: extending the existence check for an X-source or Y-source within 4 steps as a negated AND over neighbors of the distance-3 result.

Then I'm considering whether CAP (capture condition) can be expressed linearly in the logit rather than through a nonlinear combination of safe2 features, since occupied points get masked out as illegal moves anyway — meaning the capture signal might just reduce to a weighted sum over neighbors' atari status without needing explicit emptiness checks.

SAVE and SELFATARI logic need more nonlinear combinations than what the safe2 layer gives — things like escaping only when liberties actually increase, or avoiding self-atari unless it's a capture, don't reduce cleanly to a sum of per-point terms. So I'm weighing whether to add another propagation layer for these interactions versus settling for a linear approximation that combines atari-neighbor indicators with liberty counts.

I'm trying to build a single relu unit that captures self-atari by combining escape-liberty count with neighbor safety terms, but I keep running into the problem that "safe" requires a conjunction of conditions that isn't naturally linear, so I'm testing whether subtracting own-minus-threat terms inside one relu can approximate it anyway.

I'm weighing whether richer final features beat one more propagation step, but I realize jumping two adjacency steps in a single layer isn't achievable since AND propagation only works along direct neighbors. The 26 spare units in propagation layers also aren't enough to encode anything per-point, so I'm stuck looking for another angle.

Both own and opp carries seem necessary since empty-space calculation only matters in the source layer, but propagation needs same-color ownership for liberty counts and opposite-color for opponent liberties. This makes me pivot toward thinking about strategy over pure tactics — since scoring is by area, a policy favoring reasonable shape, territorial expansion, and pressure near opponent stones might win more against weak nets than precise tactical reading would.

An influence-based heuristic could work well here: computing a per-point score from nearby own/opp stones weighted by distance, then valuing moves near zero-influence boundaries (contested points) while penalizing moves deep in either player's territory. But implementing this requires a convolution over the own/opp planes plus a nonlinearity to detect "near zero," which eats into the already tight unit budget across propagation layers.

I'm now trying to fit this into later layers instead — computing influence from the carried own/opp planes at layer 5 or 6, then letting the final layer apply the relu-based thresholds. Still mapping out exactly how many units each per-point feature (own, opp, safe-distance variants) would consume against the 324-unit budget.

For the final layer, I'm sketching per-point features like a "capture count" unit that sums neighboring opponent stones in atari, computed by subtracting their safe-count from their presence — this would feed directly into the logit as a linear weight.

I'm also considering self-atari penalties, eye-detection logic, and positional influence terms, though exact-two-liberty detection for atari-making moves seems hard to approximate cleanly, so I'm inclined to skip it. Ladders probably don't matter much here either, so I'm shifting focus to weighing which of these features actually matter most against a weak RL opponent.

Thinking about endgame dynamics: the game ends after 162 moves or two consecutive passes, and since opponents may pass unpredictably, I want the network to heavily penalize eye-filling and self-atari while treating moves inside our own territory as mostly harmless under area scoring, as long as they don't threaten eye safety.

I should also worry about superko losses — a move that recreates an earlier whole-board position ends the game immediately with that player losing. This can actually arise naturally: if someone plays a self-atari stone inside the opponent's territory and it gets captured, the board can cycle back to an identical position with the same player to move again, triggering the superko rule.

I'm trying to recall how pgx actually implements the positional superko check in its Go module — it tracks board state hashes across turns and compares the new state after each move to detect repetition.

I'm thinking about how other implementations compromise on strict PSK: OpenSpiel uses SSK for legal actions but ends games in a tie if PSK occurs (detecting it but without clear agent feedback), while PettingZoo uses SSK and simply ignores PSK violations entirely for simplicity. This helps frame why pgx's own approach of limiting the PSK check to recent board history is a reasonable middle ground.

I'm now writing the logic that checks whether the current board matches any of the last several board states (an approximate PSK detection), and if so, applies an immediate loss penalty to the current player's reward instead of the normal reward.

Actually, tracing through a concrete scenario: if a single-stone throw-in gets captured and restores the exact prior board position, that capture itself would trigger this approximate PSK rule and cause the capturing player to lose — which seems like a significant edge case worth double-checking against standard ko rules.

That's the ko rule's standard mask preventing immediate recapture, but longer cycles like triple ko or send-two-return-one can still arise, though rarely. Trying to construct other repetition scenarios in a two-point eye space, I find they mostly reduce to captures of whole groups rather than true positional repeats, so superko rarely triggers in practice.

I'm also confirming the game length cap of 162 moves matches pgx's default (size²×2 for a 9x9 board), and recalling that scoring uses Tromp-Taylor area rules—counting stones plus territory reachable by each color.

But crucially, dead stones aren't automatically removed: since the final position at the move limit is scored literally, any uncaptured enemy stones sitting inside our territory border both colors and turn that region neutral, costing us points. This means we actually need to capture invading dead stones (and avoid leaving our own dead stones in opponent territory) for the score to reflect real control.

Playing inside own territory is neutral, but playing inside opponent's territory and surviving spoils that whole region while the stone itself counts for us — and since the game caps at 162 moves, a late invasion near the end can't be captured before the game ends. Since black plays move 161 and white plays move 162, white gets the last unanswerable move, making late-game territorial invasion a powerful closing tactic.

Since the network only sees the board and color, not the move number, it can't directly track how close to move 162 we are, though board fullness gives a rough proxy. Under area scoring, captured stones cost nothing (no prisoners counted), so even a self-atari invasion into opponent territory is essentially free — it costs a tempo but spoils their point if they don't bother capturing it.

If the opponent does capture, they're just filling their own territory, which could eventually shrink their group to the bare minimum of eyes. So persistent throw-ins can be a genuinely effective endgame tactic against weak opponents.

The key distinction is self-atari of a lone unconnected stone versus self-atari that joins an existing group — the latter risks losing the whole group, so the penalty should scale with group size, not be uniform.

I'm also wondering whether repeated throw-in/capture sequences could trigger positional superko concerns, since the board could cycle back to prior states.

The RL opponents likely absorbed similar tactical patterns during training, which explains their high win rate against purely random players who never weigh group size or capture value. For the architecture, I need a way to estimate group size when scoring capture moves, since bigger captures should get stronger preference than the current weak proxy of counting adjacent atari stones.

I'm also checking whether the propagation depth correctly identifies groups in true atari—verifying that with three propagation steps, a group with exactly one liberty produces no false "safe" signal anywhere within it.

So true atari always gets detected correctly, with no false negatives. But I'm noticing false positives are possible: groups with two or more liberties can still appear to be in atari if the signals don't reach within range, causing the policy to treat an ordinary attachment as a capturing move—somewhat harmful for attacking, though mostly harmless when it triggers unnecessary defensive extensions on our own groups.

I'm trying to work out when this actually happens in practice—for a big wall of stones, liberties on the adjacent row alternate in parity right along with the stones, so both signals still show up locally and there's no false positive. It seems like false positives would require a region where stones are packed tightly with few nearby liberties, like a point deep inside a large blob of stones.

I'm reconsidering how to allocate propagation layers more efficiently to extend reach, maybe by combining steps. The issue is that L1 computes directly from inputs while source bits depend nonlinearly on threshold computations, so I can't just merge the first propagation step into L1 — the sources need to already exist before I can propagate them.

For the safe2 feature, I could compute it either directly from L4's propagated W values, or push propagation one layer further and compute safe2 at L6, but then I'd lose the ability to combine safe2 nonlinearly with other features downstream. I'm checking which final features actually need that nonlinear combination versus which can be approximated linearly — CAP seems fine as a straight sum, while SAVE technically wants a product term that I might need to approximate with weighted linear terms instead.

For the escape condition, I'm reasoning through when playing at point p actually rescues a group in atari: counting new liberties from empty neighbors, adjacent own group liberties, and any captures, then checking thresholds — two or more new liberties means real escape, zero or one means it's still atari or even a wasted self-atari move unless a capture is involved. I'm sketching this as a linear logit combining the own-liberty count with indicator terms for these escape thresholds.

Now I'm restructuring the feature layers: one layer holding raw own/opponent stone and liberty-reach counts for both colors, and a second layer computing derived features like safe-liberty counts, eye detection, and threshold indicators from those raw counts, with logits as linear combinations across both layers — though I'm noting that detecting self-atari properly requires nonlinear dependence on neighboring groups' safe-liberty counts.

I realize a neighbor group with exactly 2 liberties isn't actually safe after the merge, since one of those liberties is the point being played, so detecting 3 liberties would be needed but isn't feasible with this approach — it's an approximation at best. Given the architecture constraints, I'm leaning toward L1 local features, L2-L4 propagation with reach 3, L5 safe2 plus carried planes, and L6 nonlinear features, though I'm second-guessing whether reach 3 is sufficient since 9x9 groups can span further.

I'm trying to see if I can merge own/opp carry into a single unit to save budget, but the algebra doesn't work out cleanly since recovering both values from one combined signal requires extra information I don't have.

Let me consider instead reformulating the propagation so the neutral term isn't needed at all — maybe using an AND restricted only to own-neighbors, or switching to an OR-based clamped formulation to avoid that bookkeeping.

Right, without own(s) gating, empty or opponent points would wrongly pick up signal, so own/opp both need to be carried forward, doubling the per-point cost to roughly 6 units total. I'm now considering whether parity tricks relative to the adjacent liberty q could substitute for full signal propagation, since p sits at opposite parity from q and other same-parity liberties might serve as a cheaper proxy.

For the capture test at p (odd), q is even, and the group already has odd liberty p. Safety requires either an even liberty (caught by Y-signal) or another odd liberty besides p, but that second case isn't reliably caught by X since X is already true just from p itself, so testing "Y ∨ T2" alone seems insufficient.

Given that, I think combining Wx and Wy covers that missing edge case properly. Trying to merge both parity channels into one wouldn't work since the needed signal depends on p's own parity, so I'll keep them separate and accept propagation at roughly 486 units plus 26 spare, giving reach of 3 — deciding that capturing/escaping precision matters more here than extending reach or doing full liberty counting.

Now I'm defining the L5 layer features (own, opponent, atari status for each) and sketching how L6 derives neighbor-sum features like counts of owning/opponent neighbors and empty degree for each point, building toward precise logit definitions.

I realize I need capture and save counts from neighbor groups in atari, plus safe-own-neighbor counts, but linear logits can't easily carry per-group atari states through a single neighbor-sum unit — I'd need a relu unit per point to approximate the capture feature, which is getting expensive to represent cleanly.

Working through the escape condition for F_save, I'm trying to combine the save-neighbor, empty-neighbor, safe-own-neighbor, and capture counts into a single threshold expression that correctly fires only when a real escape or capture exists, but the linear combination keeps producing edge cases I need to verify.

And if captures happen, freed points add liberties too. So escape roughly requires nE plus safe-group presence plus capture contributions to sum to at least 2, which I can approximate as a combination of relu threshold units checking these counts jointly.

I realize the clamping doesn't quite work since a can exceed 1 and corrupt the logic — I need a needs to be bounded to {0,1} properly, since savN could count up to 4 adjacent atari stones. Without a true min operation in one layer, I'm testing edge cases like two adjacent own atari stones with no escape liberties, which would represent a suicide-like move.

Maybe instead of forcing exact thresholds with one combined unit, I should design several L6 units whose summed logits produce the right behavior across cases, since self-atari penalties might offset bad "save" triggers. Alternatively, I could push more precomputation into L5 so L6 only needs simple linear combinations, but that requires L5 to already know about atari-adjacency, which conflicts with where safe2 is actually computed.

Given 512 units at L6, I can allocate a handful per point to approximate AND/OR logic through summed ReLU ramps — one unit capturing capture counts, another combining save-related signals like opponent atari neighbors and own safety counts.

Testing K=1 still breaks things: with savN=0, large escape scores incorrectly trigger a positive save signal, which is wrong since there's no atari to save from. I need a cleaner separation—maybe a separate binary indicator for whether savN is actually nonzero, rather than relying on the clamped arithmetic to behave correctly at the boundary.

I'm counting unit budgets per point: adding separate atari-neighbor indicators for own and opponent plus the base own/opp channels pushes the total over 512, so I need a cheaper way to fold in the escape-safety signal without blowing the budget, maybe by summing neighbor signals directly instead of computing per-stone relus.

For the save logic with clamping, I'm trying to construct a difference of two relu units so that the capped save-count interacts properly with escape score and capture count, though I need to verify the thresholds actually produce correct logit behavior across cases.

Testing a case where savN=2, esc=1 shows the construction gives a wrong bonus, so I realize this difference-of-relus approach can't really express min(1, savN) since that's not linear in these quantities. I think I need to redefine the L5 feature to directly include clamped neighbor atari indicators instead, dropping the per-stone safe count.

I'm now considering what L6 actually requires: a capture indicator for adjacent opponent atari, an own-atari-adjacent indicator, indicators for adjacent safe own and opponent groups, and neighbor counts by color. At L5, I'm thinking about clamping these per-point indicators directly.

Working through the clamping arithmetic — using min(1, ·) via relu differences to bound the indicators to {0,1}, since a single stone's weighted contributions could otherwise exceed that range. The tricky part is that defining a "safe adjacent own group" indicator seems to require two layers of relu operations rather than a single threshold, since safety itself needs an initial relu before I can aggregate it across neighbors.

I'm now reconsidering the architecture: instead of trying to combine AND-like neighbor conditions at a hidden layer and then clamping, I think it's better to express the final logit directly as a sum of relu terms over linear combinations of layer-5 quantities near each point, designing each term so their combination naturally orders the desired cases without needing an extra nonlinear layer for the AND logic.

I'm defining variables counting adjacent opponent-atari stones, adjacent savable own stones, adjacent safe own/opponent stones, and empty neighbors, then working through priority cases—captures get the highest bonus scaled by stone count, escapes get a slightly lower bonus contingent on an escape condition involving empty neighbor count.

Now I'm reasoning through self-atari detection: counting empty neighbors to estimate whether a new group ends up in atari, and working out exceptions when connections or captures offset the penalty for low-liberty moves.

I'm also thinking through eye-filling logic—if a point and all its neighbors belong to the player's own group, filling it should be heavily penalized unless adjacent groups have other liberties, since genuine two-eye shapes must never be suggested as fills.

I should also add positional priors favoring center or third-fourth line points, plus influence near boundaries. Now I'm working out how each unit combines capture, atari-escape, and other features into a relu-based linear construction, trying to encode conditions like "atari and escape liberties ≥2" as thresholded sums.

I'm testing edge cases with the escape unit when a≥2, checking that the formula gives zero bonus for genuinely bad self-atari moves while still correctly rewarding true saves across different a values.

This gives a reasonable self-atari bonus formula, but I also need a separate penalty term for moves that extend a group into atari rather than out of it—adding stones to an already near-dead group. I'm working through the legality conditions for the zero-liberty case, checking when a move is only legal because it captures or connects to a friendly group with spare liberties.

If s=0, a=0, it's a lone throw-in stone with just one liberty, a minor wasted-stone penalty but not fatal. I'm working out threshold formulas for self-atari penalty units, checking when they trigger value 1 versus value 2 based on combinations of n, s, and c being zero.

Trying relu(a − K(n−1) − Ks − Kc) but I realize n=0,s=1 cases are tricky since playing there could actually save an atari group by connecting it to a safe one, which shouldn't be penalized. The linear relu approach isn't cleanly separating these cases, so I think I need to enumerate the integer combinations of (n, a, s, c, t) directly instead.

Since fitting weights via regression on example data isn't allowed, I can't just build a target table and least-squares it — that's basically labeling examples. Instead I should construct the units analytically, using indicator-style ReLUs for specific discrete combos since the feature space is small and enumerable.

A single ReLU indicator for an exact lattice point isn't straightforward with integer variables, so I'm considering whether pushing to binary indicators at an earlier layer would let later layers implement clean ANDs, but that conflicts with where the safe2 aggregation needs to happen given propagation reach.

I realize a clamp of a linear combination — relu(z) minus relu(z−1) — actually gives a threshold indicator [z≥1] when inputs and weights are integers, which is effectively a halfspace AND built directly into the logit sum rather than needing separate gating units.

Testing whether conditions like "a≥1 ∧ e≥2" can be captured as a single halfspace, I'm checking whether specific boundary points can be linearly separated by some weight vector and threshold.

Working through the algebra, I find a contradiction forcing β > 3α but also requiring the opposite relationship, confirming it's not a valid halfspace without clamping a first. I'm now considering whether the a ≥ 2 scenario is rare enough to approximate, and thinking through which board configurations actually produce two adjacent stones of an atari'd group.

I'm also considering whether changing what gets computed per-stone at that layer would help, but the atari/safe2 values are just linear transforms of each other, so that doesn't add expressive power. I'm exploring whether I can use the neighbor weights directly to avoid needing two layers, since the "exists a neighbor in atari" aggregation is an OR-of-ANDs, which normally requires more depth.

I realize that specific case actually collapses: since atari is a disjunction over the two binary neighbor weights, the existence of an own-neighbor-in-atari (and its opponent counterpart, capture availability) can be computed directly as a single OR layer. Safe2 aggregation doesn't simplify the same way though, since it's a true OR-of-ANDs, so I'm thinking about how to approximate it instead.

Trying to use unclamped sums per-point to count neighbors satisfying atari runs into trouble, since the next layer would need to both threshold and AND simultaneously, which a single linear-plus-ReLU unit can't do cleanly — so I'm stuck weighing unit budget against expressiveness here.

I need to figure out which ANDs are actually required at L6: save (atari-own AND escape), self-atari conditions, and eye detection (all neighbors own AND not atari-own). If L5 carries binary safe/atari flags plus own/enemy flags, that's six units per point times 486 points, and then L6 can sum over neighbors to count how many are safe-own versus atari-own for computing escape-related quantities.

I'm now weighing an alternative L5 encoding using per-stone safe-own, clamped atari-own, and unclamped atari-enemy counts, and working out how capture detection could use the unclamped atari-enemy count since atari stones carry exactly one liberty-parity signal.

Checking the math: for a stone adjacent to empty point p, the sum of its two parity-flag contributions equals exactly 1 if it's in atari and 0 otherwise, since being a source for its liberty class zeroes one flag while the other stays unsignaled. That means summing these flags across p's neighbors gives exactly the count of adjacent opponent stones in atari, and the same logic applies symmetrically to own stones.

Since these counts are linear combinations of layer-4 outputs, a single relu unit at layer 5 can carry each value directly without needing extra nonlinearity, and clamping to binary just needs two units (relu(a) minus relu(a−1)).

Alternatively, I could propagate through reach 4 and compute features at a single later layer as halfspace ramps of those linear counts, with logits as a linear combination of ramps — effectively a one-hidden-layer network over the 5-dimensional integer input (a, c, s, t, n). Since any function on a finite grid can be expressed as a sum of piecewise-linear ramps, this feels like a promising way to trade off reach versus feature-layer depth.

I'm leaning toward reach 4 over reach 3 with two-layer features, since 9x9 group sizes can be large and false ataris could be costly — a false capture just leads to an aggressive-but-harmless move, a false save wastes a move but is mostly harmless, though a false self-atari positive is trickier given reach limitations.

Working through the layer budget, I'm tallying how many units per point are needed at L2-L4 for own/opp/positional features, then figuring out exactly what L5 needs to carry forward for L6 to compute eye detection, influence, and the nonlinear combinations.

I'm defining the p-centric quantities: liberty count n, adjacent-own-stones-in-atari count a, and starting to define another count c for similar neighborhood properties, all derived from sums over neighboring occupancy and atari indicators.

I'm now working out how to encode these as clamped ramp features for L5, needing pairs of ReLU units (relu(c) and relu(c−1)) to represent min(1, c) type thresholds, and similarly for a, which adds up unit costs quickly.

I'm also mapping out binary indicator variables L6 would need — things like [c≥1], [a≥1], [s≥1], [n≥1], [n≥2], and noticing that the eye condition (no neighbors, no opponent stones, no empty count) collapses nicely into a single ReLU unit since it's just relu(1 − n − t − c). I'm checking whether other halfspace indicators for small-range variables can similarly be built with just one unit each.

I realize complements like [v=0] are cheap (single ReLU unit), so I can build the AND conditions needed at L6, like a "save" condition combining an atari flag with an escape flag made from combinations of these indicators.

I also want an "eye" indicator capturing when all neighbors are owned, and I'm trying to work out an AND-style combination at L6 for when an escape attempt actually fails — essentially no capture available and either zero liberties or exactly one liberty with no safe neighbor to connect to. The tricky case is when a group in atari connects to an already-safe group, since the resulting liberty count depends on that group's own liberties minus the connecting point, which isn't directly observable at this layer.

I settle on approximating the failure condition as capturable AND no safe neighbor AND liberties at most one, treating the ambiguous connect-to-safe-group case as a success. From there I build SAVE as not-atari AND not-fail, and treat SELFATARI as equivalent to this same fail condition since placing a stone that ends up with at most one liberty is exactly a self-atari situation.

Now I'm working out how to express the capture term and self-atari penalty as linear logit contributions from L6 units, since the loss size depends on how many stones get captured when joining atari'd neighboring groups.

For the fail/save logic, I'm constructing Fail as a single relu unit combining capture, self-atari, and liberty-count conditions, then deriving Save as the complement of (Atari AND Fail) using linear combinations of the L5 units, avoiding the need to compute u_fail in the same layer as a dependency.

Now I'm adding a self-atari penalty scaled by whether this is a throw-in versus a real atari, and an eye-protection penalty that fires when a point is an eye and the new stone is adjacent, combining Eye and Za into a single relu term.

I'm counting up the per-point units needed for layer L5 and realizing the naive tally of eight units per point blows the budget, so I'm trying to collapse the liberty-count indicator into fewer units — instead of separate flags for n≤0 and n≤1, I'm testing whether a single combined expression like relu(2 − n − 2s − 2c) can capture the "fails with at most one liberty" condition in one unit.

Checking the edge case where n=0, s=0, c=0: this corresponds to a genuine suicide move, which is correctly illegal, except when a neighboring atari classification is itself a misclassification — in that rare case F5 would wrongly output 2 instead of signaling failure. I'm deciding to treat F5 as an approximately binary fail indicator for legal moves and moving on to combine it with the other activation term to build the full A∧Fail expression.

Counting units per point at L5: o, x, Za, Zc, F5, Eye5 gives six features across the board, totaling 486 units, and Zs doesn't need separate tracking. I'm now questioning whether o and x are even necessary at this layer versus folding positional biases elsewhere, and weighing what L6 strategic features to prioritize given the limited per-point budget — capture, activation, failure, and eye features take up five slots, leaving one more to decide on.

Since Zc and Za both enter the logit linearly, I realize I can combine their weighted contributions into a single carried unit rather than tracking them separately, since the combination itself is always nonnegative and can be passed through one relu. This means any linear-in-L5 term can potentially be merged this way, saving units.

EyeF and AF each need their own units, so L6 holds Lin, AF, EyeF plus the influence units. I'm also wondering whether trickier tactical features like atari-making are detectable with this limited architecture — probably not exactly, since it'd need more propagation channels to track group liberties, so I should skip that and focus on simpler local checks like new-group liberties.

For strategic influence, I'm thinking through simple heuristics that work against weak bots on 9x9: favor center and star points at opening, prefer third and fourth line moves, avoid first-line edge plays unless tactically needed, and treat second-line as mildly favorable.

I'm weighing a frontier-based approach—favoring moves between my own and opponent's influence rather than deep in either territory—since playing inside my own area wastes moves and playing deep in the opponent's area risks dying, except as a late endgame invasion tactic. I'm also considering whether a basic heuristic set (extend stones, capture, save groups, avoid self-atari, don't fill eyes) suffices, or whether I need a proper influence function that sums decaying contributions from each stone to estimate territory and evaluate how much a candidate move shifts that balance.

I should also think about the pass logit — it needs to be kept very low relative to normal move logits so pass only happens when every other option is genuinely bad, since a premature pass (especially if the opponent then also passes) could end the game before territory is settled, which is harmful if we're behind.

I'm also wondering about throw-ins into opponent territory when no good moves remain besides passing — since area scoring gives no benefit to prisoners, a throw-in costs nothing even if captured, but could pay off if the opponent fails to respond, so it may be preferable to passing in those edge cases.

I want the pass logit set below throw-ins but above eye fills and group self-atari — since giving up a lone stone just costs a small tempo, not a real loss, so other moves should still dominate it. Now I'm considering edge cases, like a lone stone with one liberty placed inside our own territory rather than the opponent's.

Turning to eye detection, I'm thinking through false eyes where orthogonal neighbors are ours but diagonals belong to the opponent — filling these may be necessary to stay connected, and if left unfilled the opponent could legally play there only if it captures an adjacent group in atari, which shapes how the save logic should treat atari cases differently from safe ones.

I'm now working through multi-point eye spaces, like a two-point eye where filling one point might not trigger a penalty under the current rules even though it could leave the group vulnerable if no other eye exists elsewhere, which suggests the policy needs refinement for these cases.

Actually, filling becomes genuinely dangerous only near two-point eyes, so I could detect deep own-territory points by checking that neighbors and their neighbors are all own-or-empty, and penalize those below pass. That way we'd naturally pass rather than fill safe territory, while still leaving throw-ins into opponent territory as valid above-pass moves.

But I realize this "no opponent within distance 2" heuristic would also wrongly flag early-game extension moves near a lone stone on an empty board, since those points also lack nearby opponent stones — so I can't apply it uniformly from move one.

Instead I'm thinking about valuing moves by how much they shift the influence-based territory estimate: a move gains value from points that flip from contested or opponent-leaning to clearly owned, weighted by how close the influence margin is to zero at each point.

The issue is budgeting network layers for this — computing per-point contestedness is nonlinear, then summing over a neighborhood is linear, needing two layers, but those layers are already packed with tactical features. I'm weighing whether to shrink propagation reach to free up a layer, or fall back to a simpler proxy like counting nearby own stones within distance 2 as a cheaper substitute.

Thinking more about what weak 9x9 policy nets actually look like — given Tromp-Taylor scoring with no dead-stone removal and a 162-move cap, surviving stones on the board are what matters most, so a policy that aggressively captures and defends its own groups should dominate, since the cap equals exactly 81 moves per side, the size of the board.

Toward the endgame, eye-fill avoidance becomes decisive: a player that carelessly fills its own eyes self-destructs, letting the opponent capture freely, so I just need to make sure I never make that mistake while hoping the RL opponents also know to avoid it. I'm also factoring in that my network's large logit gaps make it play fairly deterministically, with randomness only breaking close ties, which should keep it consistent and strong against stochastic opponents.

For the positional influence budget, I want to compute for each point a weighted sum of nearby stone differences — full weight for adjacent points, half weight for points two steps away — to estimate territorial influence. Then I'd penalize moves that go too deep into my own secure territory (wasted moves) and separately penalize moves that invade too deep into opponent territory, since those are likely to die in the midgame even though the self-atari throw-in cases are already handled by the Fail feature.

I'm also weighing other heuristics: reducing opponent territory near the boundary is good, opening moves should favor distance from edges, and contact moves near stones tend to be where the action is, though these signals can conflict. I'm also considering whether I can cheaply extend the liberty test to three or more liberties to detect atari-setting moves and two-liberty group extensions, but that likely exceeds the budget, so I'm looking for a cheaper approximation instead.

Detecting a single opponent stone with exactly two liberties as an atari-adjacent pattern seems computationally expensive within the current unit budget, since I'd need to carry that info through multiple layers rather than just one. I'm also checking whether I really need to track both own and opponent liberty propagation symmetrically, since both self-atari avoidance and opponent-capture detection depend on it, and trying to see if dropping opponent tracking in some layers could free up computational units.

Trying to derive opponent presence purely from own and empty neighbor counts doesn't save anything since I still need two of the three quantities tracked explicitly. I also consider merging two weight channels into a single unit with a combined ternary value, but that fails since the AND propagation requires separate ReLU operations per bit.

So I'll settle for 486 units in layers two through four with 26 spare capacity, possibly reserved for a global feature like total empty-point count to signal game phase and modulate later decisions through additive ReLU interactions.

Given three propagation steps, I'm checking whether group-level atari status can be pushed one more hop earlier — computing an atari indicator over each point's full neighborhood as a single relu, then summing those into the capture-count feature, though I run into trouble trying to do a further threshold on that sum within the remaining layer budget.

I'm now trying a different decomposition: use the last propagation layer for a 4th-reach version of the liberty features plus ownership/stone indicators, then collapse everything into one final layer of ramp (clipped-linear) combinations feeding directly into the logit. I'm listing out the logit terms needed — a capture indicator built from clipped relu's of the capture count, and a failure condition requiring simultaneous zero-liberty-after-capture, self-atari, and a bounded neighbor count — trying to see if each can be expressed as a single linear ramp unit.

I'm now testing coefficients to isolate the "attack succeeds despite fail conditions" case, searching for a single ramp expression that fires only when the attacker stone is present, the neighbor count is low, and both self-atari and capture counts are zero, but running into cases where the edge conditions (n=0, a=0) don't zero out correctly.

Adding the constraint γ ≤ β from the zero case, I get γ > β − α too, so picking β = 4 narrows the range for γ and α.

So δ needs to be at least 5, confirming the self-atari penalty formula scales with group size, which acts as a nice proxy for how many atari stones get joined. Now I'm thinking about the save case: wanting a bonus when atari is averted (A ∧ ¬Fail), which I can express by subtracting the failure overlap from the atari indicator, or just use the raw count since bigger rescued groups touching p deserve bigger bonuses.

For the save logit I'm balancing two weighted terms so that failed saves net out to a pure penalty proportional to group size, while successful escapes keep the full bonus — all driven by the same linear count variable a, which a single unit can carry forward. Then for detecting an eye, I want a unit that fires only when every neighbor is owned and there's no threat, atari, capture, or opponent stone present — a single relu expression handles that. Finally I'm setting up a simple linear capture term weighted by the capture count.

Now I'm checking whether the whole tactical logic — saves, eyes, captures, atari counts — can be packed into layer 6 using only linear combinations of what layer 5 produces, which would free up layer 5 to serve as a fourth propagation step and reach the depth target of 4. I'm tallying the unit budget: one linear-packing unit plus three tactical units per point, plus two influence units, giving six units per point and 486 total — then I start re-verifying the fail-unit formula and how the threat/support counts are actually computed from the owner/liberty indicator values.

Now I'm laying out the layer plan: L1 computes local own/enemy liberty counts and derives parity signals Wx0/Wy0 per point using the relu combinations, costing 6 units per point across the board. I'm continuing to sketch out L2's propagation logic.

L2 performs an AND over each point's neighborhood of the L1 outputs, combining own-stone presence with neighbor terms minus degree and a constant penalty for non-own points, keeping the binary nature of W0 intact. Then L3 through L5 extend reachability outward by 2, 3, and 4 steps respectively, feeding into a features layer and a final logits output, while I double-check that the constant C needs only to be at least 1 to enforce the correct thresholding.

So reach values increase by one per layer (L2:1, L3:2, L4:3, L5:4), and L6's atari test for q relies on sources within distance 4 of q via neighbors' W5 values. I'm checking the remaining feature budget per layer (26 spare at L1 and L2-L5, 26 at L6 if using 6/point), and considering whether the atari-making feature—detecting an opponent group adjacent to p with exactly 2 liberties—can be derived from the W-based info already computed.

The issue is I only know whether a group has at least 2 liberties, not exactly 2, so I'm exploring a local pattern: identifying if q is a lone opponent stone with exactly 2 empty neighbors, but summing this per-neighbor quantity would require 4 units per point which exceeds budget since L5 is already full with o, x, and four W features. I'm considering dropping a separate "Fail" unit to free up space for this computation.

Listing out the L6 per-point components: a linear term, Fail(p) based on liberty/stone counts, AF(p) for atari-related conditions, Eye(p) for eye detection, and InfOwn/InfOpp for influence thresholds — six units totaling 486, leaving 26 spare. Now I'm checking whether Fail and AF can be merged into a single condition.

Working through whether a single unit of form relu(β + a − γ(n−1) − δs − δc) can replicate both penalties at n=1 versus n≥2, solving for the constraint on γ in terms of β.

Checking the s≥1, c≥1 case against a≤3 bound confirms it holds with smaller β. I'm reconsidering the penalty ratio — maybe a small flat cost for a lone throw-in versus a steeper cost scaling with group size makes more sense, settling on β=0.25, W=8 as a workable tuning. With the per-point feature budget of 405 plus 107 spare points for L6, I'm thinking about what else to add: atari-approximation features, extra influence shaping, or liberty-gain signals, and starting to consider an "own stone" related feature.

But the real concern is whether the parity-based 2-liberty test is tactically sound. I'm checking failure cases — a single stone or a two-stone group with properly offset liberties works fine, but a three-in-a-row group with liberties only at the two far ends turns out to have the same parity, since they're four apart, which breaks the simple parity check.

That means such a group would falsely register as atari rather than safe, though thinking it through, this mostly matters for already-weak 2-3 liberty groups where treating them as atari is roughly the right call anyway since they need defending regardless.

There's a subtler failure though: a false atari on our own group could incorrectly trigger the self-atari penalty on an otherwise fine move, since the move's single liberty gets misjudged as unsafe due to nearby stones being wrongly flagged as already in atari.

The riskier case is false atari on an opponent group — if we think we're capturing, the self-atari check might wrongly excuse our own stone's single liberty, leading us to place a stone that's actually vulnerable and gets captured back, costing a tempo and a stone, though this should only happen occasionally given the reach and parity constraints.

Separately, I need to think through ko legality — after a single-stone ko capture, the immediate retake needs to be masked out.

Beyond tactics, the bigger strategic question is how much a tactically sound, capture-aware, self-atari-avoiding policy with a reasonable positional prior would beat weak RL nets — probably 60-80% given that weak play mostly fails at basic group survival. For positional priors on 9x9 with komi 7.5, opening near tengen or the 3-3/3-4/4-4 points seems sound, and Black needs roughly 45+ of 81 points under area scoring.

I'm sketching an influence-based move valuation that penalizes points too deep in either player's territory while rewarding neutral points with positional bias — this naturally pushes early moves toward big open points, mid-game moves toward contested boundaries, and endgame moves toward dame and boundary plays once territories are mostly settled.

For deciding between passing and filling one's own territory, passing is safer since it risks nothing if both sides agree the position is final, while filling non-eye territory is only harmless under area scoring but could needlessly shrink eye space — though I need to be careful that influence alone can't distinguish genuinely safe territory from cutting points or defects that still require defensive moves.

But I realize passing too eagerly is risky against an opponent who keeps playing: if they invade with a stone that isn't in atari, we won't capture it, and if we just pass, they could keep building inside our territory and either live or at least neutralize that area under Tromp-Taylor scoring, so we actually need active responses to invasions rather than passing reflexively.

I'm thinking a contact bonus is a good proxy — moves adjacent to an invading stone in our own territory should score well since they reduce its liberties, while contact moves on the opponent's side carry risk that self-atari detection should catch. I also want to add a positional bias by line, giving first and second lines penalties and gradually favoring more central play.

Now I'm trying to size the logit weights: capture moves get a large bonus (roughly +12, with a small extra per additional adjacent stone), saving a group in atari gets a smaller but still strong bonus (~+9), and I'm weighing how to handle cases where a move both saves and captures. I'm also setting a self-atari penalty for throw-in style moves around -3.

For riskier extensions that create self-atari, I need the penalty to scale with adjacency so the net value turns negative even after the save bonus — working through the numbers, a weight around 13 with a small offset gives roughly -3.25 for a lone throw-in and about -7.25 net when one adjacent stone is involved, which feels right. I'm also setting eye-fill moves below pass value, adding modest influence penalties for playing too deep into either player's territory, a small per-contact bonus, and a liberty bonus to discourage self-filling, before moving on to how pass should be valued.

Now I'm checking whether pass should sit near -5 relative to neutral moves: with typical positional bias plus liberty bonus, neutral moves land around +1 to +3, so pass would rarely win when real options exist. But late in the game, filling one's own territory scores close to pass value too, so with many such nearly-equal points the probability mass spreads out and pass could lose out to territory fills, which is mostly fine except for the dangerous case of filling a two-point eye — those points have very high adjacency-to-own-stones, so the influence penalty needs to kick in hard there to prevent self-destructive eye fills.

Checking against a 3-point eye row: the middle point has two safe neighbors so it's unaffected, while filling it still leaves two separate one-point eyes—correct behavior. The endpoints get penalized as intended, though this unit might also discourage some legitimate solid connections like tiger's-mouth shapes, which I'm assuming the save/atari logic covers separately. I'm trying to merge the eye unit and this stronger penalty unit into one consistent formula.

Working through the weights, an eye point and a near-eye point both end up below pass value, which seems right for endgame behavior, and the atari exclusion correctly still allows saving or connecting groups even when a liberty point looks eye-like.

I'm weighing whether to keep a separate own-area penalty for wasted deep moves, deciding to retain it since it discourages midgame filling. For opponent territory, invasions usually die but weak opponents might not capture reliably, so I'll keep a mild penalty rather than dropping it entirely, and I'm starting to think about how dead stones left in territory at game end should factor in.

I'm weighing distance-2 contributions at half-weight, so a point surrounded by several own stones nearby reads as strongly positive while dame points touching both colors cancel out near zero — I need a threshold like θ≈1.5 to separate solid territory from contested boundary points.

But big empty regions far from any stone would also read as zero influence, making them look falsely neutral — a proper flood-fill ownership check would fix this but isn't cheap. I'll treat influence as a secondary feature and move on to the pass logit, which should only dominate when remaining moves are eye-fills or bad self-ataris.

Working through own-territory penalties: with pass fixed low, own-area fills still beat passing until regions shrink to near-eye shapes. Tracing through small region shapes, the fill process naturally reduces them down to single-point eyes, which keeps the group alive.

Really there's no need to fill in our own territory since it has zero value—the only reason to consider it is uncertainty about whether the opponent will keep passing. Passing is safe as long as the opponent keeps playing; the actual risk is if they pass right back and the game ends in a worse scoring state than if we'd kept playing useful moves like dame or responding to throw-ins. So the real question is whether any productive moves remain, not whether to fill territory.

There's no actual loss here, so that works. Deeper invasions (lone stones with more liberties) deserve a milder penalty since they're harder to capture and spoil territory more robustly, though they cost tempo in the midgame — I'll keep that penalty mild rather than harsh. Now I want to think about how deterministic to make the policy logits.

I'm also refining the weighting formula: capture bonus for each adjacent atari opponent stone, a save bonus for rescuing an own stone in atari, and a smaller bonus per liberty — working out edge cases like double-counting when a move ataris two separate groups at once.

I'm weighing a small contact bonus for touching safe opponent stones versus a near-neutral adjustment for touching own safe stones, since clumping isn't clearly better or worse than connecting. I'm setting aside ladder and snapback considerations as too complex to model directly, and focusing instead on how tactical sharpness should matter most since the opponent is sampled from a weaker policy.

Checking edge cases: when a capturing move is also self-atari, it's properly excluded; when an own atari group can't extend, the model should fall back to capturing an adjacent atari opponent group via the capture bonus, with overlapping bonuses stacking correctly rather than conflicting. I'm also refining how contact counts should subtract out capture moves to avoid double-counting, and starting to sketch the influence-layer computation that compares opponent and own stone density within a radius around the candidate point.

I'm reconsidering whether the spare units per layer should encode global phase information, like empty-point count, to modulate throw-in or pass behavior — but conclude that's unnecessary since the self-atari penalty already dominates early-game decisions without needing phase awareness. Then I'm working through what the very first move on an empty board should be: with all features zero, the logits reduce to the positional bias plus the liberty term, which favors higher-liberty points like the center over edges and corners, so the opening move naturally gravitates toward the center.

Thinking through subsequent moves, I want tengen to be the overwhelming favorite as the opening play, then have later moves balance the positional prior against an influence term — penalizing points too close to one's own stones above some threshold while mildly penalizing points too close to opponent stones, with a contact bonus layered in to encourage reasonable shape like playing toward the 3-3 to 4-4 region after the first exchange.

Honestly, opening theory matters little against weak bots since tactics dominate. A bigger concern is whether the policy properly defends groups down to two liberties, since bots may not reliably punish this, and whether it ever plays deliberate ataris beyond the contact bonus and false-atari capture logic on two-liberty groups. I'm wondering if I can cheaply detect when a lone opponent stone has exactly two liberties using only local neighbor information.

Working out the feature logic...

This gives atari-detection for lone opp stones with few liberties, letting capture bonuses handle the rest. But checking the L5 budget, I only have 26 spare units and this feature needs 81 — not nearly enough, so I need to look at restructuring L5's W propagation to free up capacity.

I'm also considering whether the 26 spare units in L1 could help, though edges and corners likely matter less. Another angle: maybe I can encode each stone's liberty-direction signals in 2 units instead of 3 by noting interior stones can share states, cutting down the per-point cost for carrying ownership and liberty info.

But working through it, two units only give two linear functionals, which can't independently recover three quantities (ownership, Wx, Wy) without relying on nonlinearity further downstream — so this compression doesn't actually work with purely linear decoding in the next layer's pre-activation.

Actually, defining Sx = own ∧ Xsignal, I realize the X-propagation and Y-propagation each only need their respective S-value plus ownership and the complementary W-term, which comes out to the same three values per point. So the storage budget stays at three per color regardless, and I'm now wondering whether the final propagation layer could instead compute neighbor counts some other way to avoid needing both ownership and color explicitly.

For the next layer's budget — Lin, SA, E2, InfOwn, InfOpp at five units per point gives 405 used with 107 left over. I'm trying to see if I can add atari-detection directly at this layer, but checking whether a lone opponent stone has few liberties requires looking at pairs of points, and expressing that as a per-point combination ends up needing a product of two signals rather than a linear sum, which the architecture can't represent directly.

So I'll set aside atari-making for now and consider other uses for the spare capacity — maybe shape features like first/second-line detection, empty-region info, or a phase-aware pass decision — but that can wait. I also want to double check the masking logic handles occupied and ko-illegal points correctly, and I'm looking at how the pass logit should be biased relative to the other outputs.

Now I'm thinking through throw-ins more carefully: since losing stones is free in area scoring, late-game invasions into opponent territory are essentially costless tempo plays as long as the lone stone isn't self-atari and has at least two liberties, since the opponent may simply ignore it. The same reasoning needs to apply symmetrically to the opponent's throw-ins into my own territory.

I need my policy to eventually capture enemy stones sitting in my territory, since leaving them uncaptured at game end would count against me, so contact moves against them should get a liberty-reduction bonus with the usual own-area penalty suspended once contact is made, building toward a capture reward once the stone is reduced to atari. I'm also checking that near-eye restrictions don't block these contact moves, and setting aside superko concerns as rare edge cases while I work out the right scaling constant for the ReLU term in the move-value function.

Now I'm defining SA with specific coefficients and checking edge cases: lone stones at n=1 give exactly the threshold for throw-ins, while n=0 with safe or captured neighbors forces the value negative as required for legality, and I'm now verifying the n=2 case against similar bounds.

I'm also worried about a gap: a move with no opponent neighbors but adjacent to safe groups could still create a self-atari that the current formula misses, so I'm considering adding a mild penalty for moves that reduce the player's own liberties without capturing anything.

I'll set w_lib to 0.7 and add the Z0 penalty as its own unit rather than folding it into SA, giving six total feature groups totaling 486. Then I want to sanity-check the relative weighting: capture at 12·c and save at 9·a should dominate smaller ±3 features, which seems reasonable, though there's another important case to consider.

Ladders are a real risk here — a weak bot might not detect that extending into atari just leads to a chase and eventual loss along the edge, so I'll skip proper ladder detection for now since it's genuinely hard to implement cleanly. I also want to double check that the save bonus a(p) correctly counts adjacent own-atari stones when our group's last liberty is at point p.

Now I'm trying to nail down the parity-based source definition for liberties: using board parity par(s) = (r+c) mod 2, I want an X-source signal that flags a stone as having either an odd-parity liberty or at least two liberties, based on whether its own parity is even and how many empty neighbors it has.

I'm defining the mirrored Y-source signal for the opposite parity case, and checking that safe2 (X-signal ∧ Y-signal) correctly implies at least two distinct liberties, while a single-liberty group only ever satisfies one signal type, confirming the logic holds.

Now I'm starting to define the per-color single-liberty condition Wx0, looking at the negation of the X-source for even-parity stones.

Now I'm mapping the two input planes (own stone, opponent stone) through the L1 units, which compute relu activations plus the a/b threshold units for each color, giving six units per point. Then I'm working out the L2 computation for WxO2, combining neighbor sums of Wx0O and occupancy terms minus degree to propagate the "liberty count" style feature across the board.

When own(s)=0, the bound collapses to at most -1, which the relu clips to 0, confirming the recurrence holds for the general layer update. I'm now generalizing this to the full L(k+1) relu formula using previous-layer weights, and noting layers L3-L5 follow the same pattern with direct units. For L6, I'm setting up per-point features from L5 (O, X, and their weighted directional sums) and defining neighbor counts nO, nX, n to build the next aggregation step.

I'm verifying that the parity weight guarantee holds for neighbors of empty points - since any liberty neighbor has nE≥1, its weight is forced to zero at every later layer. Now I'm defining the L6 unit computations (Lin, SA, E2, Z0) as relu expressions combining the aggregated counts a, c, n, s, t with specific coefficients.

I'm working out the influence-based penalty terms, trying to figure out which contact and save conditions should exclude an area from the own-penalty, then combining everything into a final logit formula with weighted penalty terms for shortage, eyes, and opponent/own influence.

Now I'm sanity-checking the scoring against typical endgame scenarios — dame points, own territory interior points, eye-fill moves — comparing the computed logit values against the pass baseline to see if the formula produces sensible preferences.

If the diagonals inside filled territory are also our own stones, those isolated eyes stay real rather than false, so filling is safe even if pointless and should just be deprioritized relative to genuinely useful moves. Now I'm thinking about throw-ins into lone opponent territory points, assigning them a smaller weight like 0.7.

Adding positional bias terms (first line penalty, second line penalty, center bonus) helps during the opening, but comparing midgame extensions versus attachments shows similar scores since positional bias dominates the differences. The influence penalty structure seems to be working reasonably, though I'm wondering if there's a cheaper way to capture things like extending weak groups.

I'm also considering that the RL opponents are likely weak nets with some exploitable tactical gaps, though hard to pin down without more info. I'm double-checking leftover capacity in the later layers and confirming the influence function stays simple and linear over a small radius.

Working through the influence calculation: I'm tallying contributions from adjacent versus diagonal own stones to see how territory points accumulate influence scores, then setting up a threshold-based term that subtracts for nearby threats before applying a negative weight.

Thinking about the opening move on tengen, I realize the influence penalty shouldn't start too late — if adjacent points get zero penalty, the formula would encourage clumping near the center, which is slow shape. So I'm testing a lower threshold that starts penalizing even a single adjacent own stone, to discourage that tendency unless an opponent stone is also nearby for contact play.

I also need the formula to relax this penalty when opponent stones are close but not directly adjacent, since that's when extending or haneing next to our own stones makes sense tactically. For the opening bias itself, I'm working out a distance-to-edge function that rewards center and near-center points while penalizing the very edge, to be combined with the existing positional weighting.

Running the actual probability for tengen under this scheme shows it's still too low given how many points share the d=2/d=3 bonus, so even though tengen isn't essential against weak opponents, I should just push the center bonus higher (making d=4 significantly larger than d=3 and d=2) so the first move naturally favors tengen without needing a special-cased rule.

Weak policies mostly lose groups by missing ataris or two-liberty danger, but our policy already handles atari responses. On the PSK front, ko fights and snapback-type captures naturally avoid repetition since intervening moves change the board state, so those scenarios seem safe.

I also want to check suicide and ko handling, both of which are masked out. Thinking about the move limit, a white throw-in on the final move could spoil black territory if unanswered, though the exact move count is uncertain. The pass logit of -4 versus move logits of +1 to +3 keeps pass probability negligible except near the very end, where throw-ins and fills dominate the remaining options.

If the opponent passes early, maybe out of overconfidence, I should just keep playing normally rather than worrying about it. I'm also working through the eye logic for opponent eyes: playing into a single eye is treated as suicide unless it captures, which is masked, while a lone throw-in into a two-point eye carries a penalty score of about -3.25 if unanswered.

Tracing through the two-point eye case, throwing into one point leaves the opponent's group with only the other point as a liberty, so following up there captures the whole group since capture resolves before our own stone's liberties matter. The penalty makes throw-ins unattractive compared to regular moves, but they'll still happen naturally in the endgame when needed.

But this cuts both ways: if the opponent throws into our own two-point eye, our group was already dead regardless since a two-point eye only counts as one real eye, so capturing the throw-in stone doesn't actually save us unless we have a second eye elsewhere.

Beyond that tactical core, I'm fairly satisfied, though I briefly wonder about cheap ways to represent liberty counts of three or more, or using spare capacity to track global atari conditions, but these feel unnecessary for now. I also want to make sure the bias values for each unit are set sensibly before moving on.

Checking that L6's influence calculation correctly carries O and X values through the identity layers from L1-L5. Then I trace through a specific safety scenario: a stone played adjacent to a two-liberty group, working out whether the resulting merged liberties end up at one or two depending on whether the empty neighbor coincides with the group's other liberty point.

I verify several more cases numerically: connecting to a safe zero-liberty neighbor still gives the correct bonus with both E2 and Z0 terms vanishing, extensions onto groups with two or more liberties behave properly, and a failed single-liberty extension without safety or capture correctly comes out negative. I'm now checking the capture case, confirming SA and E2 terms are properly excluded there.

Then I move on to eye-detection edge cases—stones adjacent to a mix of atari and safe own groups, and connecting versus suicide when all neighbors are in atari—before starting to work through the Z0 term.

For n=0 with no capture or save, Z0 gives −1.5, but if t=0 too, E2 also triggers an extra −30 penalty. I'm now considering the dame-filling case: a neutral point touching both colors with n=0 — under area scoring filling it should net +1 for us, unless doing so drops our own group's liberties dangerously.

This risks self-atari since we'd be filling our own liberty, so Z0's penalty (around −1, still better than passing or filling our own territory) may actually be a useful signal flagging that risk. I also need to factor in the contact weight of 0.5 per opponent neighbor, and remember that first-line dame points still carry +1 value even with the edge bias penalty, just played later in the sequence.

Now I want to work out the L6 Lin terms more concretely and check whether I can fit additional features, like a cheap atari-detection heuristic, within the remaining 26 spare units.

Detecting whether a lone opponent stone sits at exactly 2 liberties needs per-neighbor nonlinearity that L6 can't handle without a dedicated L5 feature, and there isn't spare capacity at L5 for that. An alternative would be shrinking propagation reach to 3 layers and using L5 for richer per-point features—carrying both stone color and computed nonlinear values—so L6 could perform two-level logic, enabling things like atari-making on lone stones or protecting stones with two liberties. But that tradeoff seems marginal compared to just keeping the reach at 4, so I'm leaning toward preserving reach over adding this logic depth, and now I'm reconsidering what reach 4 versus reach 3 actually means for the atari test in practice.

For large groups spanning distances greater than 4, I'm worried about false atari signals when the local neighborhood within reach 4 happens to show only one liberty parity and no second-order evidence—though with many liberties scattered around a big group, there's usually a nearby stone with 2+ liberties or a mixed-parity liberty, so false positives from this seem rare. I also check whether L1's unused capacity could help here, but there's nothing to repurpose there, and I consider whether L6's spare 26 units could support some global stone-count phase feature for pass decisions, though I decide to set that aside for now.

Turning back to strategy, I think about whether to add a feature capturing liberty-shortage of opponent groups to help the policy target weak groups more aggressively, but counting adjacent W-signal stones doesn't translate directly into a count. I then run through the tactical logic around atari: when our move ataris an opponent group, a competent opponent escapes, otherwise we capture next turn; symmetrically, if the opponent ataris us we escape when possible, capture when they're in atari, and avoid wasting moves when we're in atari with no escape.

I decide this plan is reasonable and move on to defining the indexing scheme for inputs and outputs, along with the layout of feature planes per layer, covering own/opponent stone indicators and directional weight signals.

For layer L1, I derive the neuron equations: O and X are direct pass-throughs of the board input, while the adjacency-based features like aO combine a cell's own value with its neighbors' values minus its degree to express a relu-based threshold, and I work out the corresponding weights and biases for each term.

I verify these formulas in the degenerate case where a cell is unoccupied, confirming the relu output stays correctly at zero whether all neighbors are filled or not, then I move on to defining the linear combinations Wx0O and Wy0O based on each cell's parity, distinguishing even and odd cases.

Continuing through the layer structure, L2 applies relu to the first-layer outputs to produce O2, X2, and the weighted versions, and I note that L3 through L5 follow the same pattern using direct weight units. For L6, I'm working out how to aggregate neighbor counts of occupied, X-marked, and empty points around each candidate position, along with sums of the directional weight terms.

Now I'm deriving the interestingness score I(p) from differences between O and X counts at distance 1 and 2, then starting to define the specific L6 units like "Lin" and "SA" by working out their relu expressions and coefficients in terms of the aggregated L5 quantities.

I'm double-checking that each point's logit only depends on its own unit values, so occupied points' large unit outputs don't leak into neighbors. Then I'm working through how InfOwn should scale with the number of adjacent own stones — roughly −1 per stone up to deeper penalties for larger clusters — and thinking about how nearby (non-adjacent) opponent stones should also discount influence for hane or extension moves.

I'm also reasoning that since contact bonuses plus central positioning make attaching directly to a lone White stone attractive, this could actually be a good strategic bias — fights favor us if we're tactically strong against a weaker opponent.

I'm also weighing the liberty-count weight against positional scores, and confirming that when both capture and save moves are available, prioritizing capture makes sense since captures often double as saves — including checking that a stone made by capturing always ends up with at least one liberty from the freed point.

Thinking through edge cases: capturing even a worthless opponent stone is still good since it secures territory, and extending our own dead stones inside enemy territory just to avoid atari is often wasted effort worth guarding against.

I'm sketching a positional bias function based on distance from the board edge, ranging from a penalty near the edge to a bonus toward the center, while noting this overlaps somewhat with the existing edge penalty term. I also want to confirm the scoring naturally creates near-ties among reasonable strategic moves so play doesn't become too deterministic, and I'm checking that contact plays near opponent stones aren't unfairly penalized, along with reconsidering how lone throw-in penalties should behave in midgame situations.

Now I'm catching an edge case: a point with one empty neighbor surrounded otherwise by safe own stones triggers a heavy near-eye penalty, since filling in one's own territory should be discouraged — but I need to verify this logic doesn't misfire on legitimate edge points that happen to share the same neighbor pattern.

These rare edge-fill cases are probably fine to accept as-is. I'm also double-checking that the Z0 rule correctly handles n=0 with no capture/save, giving a small penalty, which matches E2's coverage of the t=0 case. Now I want to revisit how influence should be computed for other points.

Since the RL opponents use fixed policies without search, I don't need to worry about them exploiting any deterministic tendencies in our approach. Edge-of-board shape adjustments for small neighbor counts look consistent too, and negative evaluations for occupied points simply get zeroed out via relu, so they don't matter. I still need to verify the endgame scenario where only eye-filling moves remain legal.

Working through the probabilities, the pass option ends up with roughly an 8% chance compared to filling own territory with two or more liberties, which still seems mostly harmless. Though I realize there's a subtler danger: filling territory near diagonal contact with opponent stones could create false eyes, and the simple liberty-count rule doesn't actually guarantee two real eyes survive in every territory shape.

Checking various shapes confirms the pattern holds though — 2x2 blocks, L-shapes, lines of 3 or 4, all reduce correctly to two eyes when I trace through which points get filled versus left as near-eyes.

Given this, maybe the safer fix is raising the penalty for filling into own territory so passing becomes preferable when uncertain, as long as it doesn't produce missed useful moves. The risk is that if the territory detection misjudges a point, passing could cost us that point, but with an influence threshold around 2.5 and a penalty of 4 or more, I'm fairly confident the ownership call is correct.

I should check that attacking adjacent dead stones doesn't self-atari our own stone, but that's handled fine for lone attackers. If an invading group inside our territory is actually alive with two eyes, we'd need to think about whether continued attacks risk our own liberties in a capturing race, though that's an edge case. For dead stones left in the opponent's area, if they don't bother capturing them it just hurts their own score, which is fine.

I'm also reconsidering the opening strategy given 9x9 area scoring with 7.5 komi—Black needs 41+ points, White 37+, so territorial balance in the early game matters against well-trained opponents. I'm weighing whether a better territory-value feature would help, and whether there's room in the network's sixth layer (only 26 units) to free up capacity by merging or dropping components.

A promising idea is counting empty space near a point—moves into open areas are valuable in the opening and midgame—and since that's linear in stone positions, it can be folded directly into the linear term for free rather than costing extra units. Similarly, nearby own or opponent stone density can be captured as free linear terms, leaving only a couple of nonlinear interactions needing dedicated capacity, so I can treat the strategic value mostly as a linear influence convolution plus a few ReLU nonlinearities for own/opponent influence.

On an empty board, points differ mainly by position and open-space value (edges have less), which naturally favors tengen as the strongest opening move—this seems like a reasonable design.

I'm double-checking the bias constant K in the capture/safety term so the logits stay well-behaved even if I later add negative-weighted terms, probably setting K=2 and offsetting it elsewhere. I'm also confirming the capture weight dominates appropriately when a group touches multiple liberties, though I'm now worried about false positives in the atari-detection term.

So the false-positive flags really correspond to groups genuinely low on liberties, meaning the "save" response of extending at that point isn't actually distorting play much — it's a reasonable move anyway. For the capture-side false positives, I need to check whether playing there puts our own stone in self-atari, which the exclusion condition c≥1 should rule out.

I'm now tracing a mutual-atari scenario: both groups end up with one liberty, opponent moves first and captures our stone, but that's a minor tempo loss and seems acceptable.

Let me reconsider whether a simpler parity-based classification scheme could work better than this one.

I realize I'm barred from running the network itself in the sandbox, so I can't verify via forward passes. But writing separate plain-Python logic that mirrors my parity-propagation formulas and checking it against true liberty counts on chosen boards should be acceptable, since that's testing my design logic rather than executing the actual network.

I decide inspecting weight values directly against expected formulas is safer than emulating activations, so I'll write careful code with assertions checking weight structure. Setting up grid helper functions for coordinates and neighbors...

Building the first weight matrix for each cell, encoding occupancy and neighbor relations...

Defining assignments for even and odd positions, then designing the L2 layer with units for stone type and color signals combining the L1 outputs, and starting to sketch out the L3 layer logic.

For L6, I'm setting up unit types for linearity, structural analysis, and ownership inference, then defining per-point expressions aggregating neighbor stone counts and weighted signals across the board.

I then build the final logit formula combining linearity, positional, and penalty terms, and work out influence calculations using distance-weighted stone differences with atari-adjusted thresholds for ownership and opponent influence.

With the remaining budget, I want to think more carefully about higher-level Go strategy against weak opponents: favoring open-area moves early, attaching to fight since tactics are reliable, avoiding overplay deep in one's own territory, and reacting to contact with hane or extension. I also want to account for attacking weak opponent groups and defending one's own groups with few liberties through reactive responses.

Weak groups risk ataris, ladders, and double-ataris, so I'm weighing a heuristic that rewards extending a stone next to a friendly group with few local liberties, though capturing that nonlinearity with purely linear terms is tricky. I'm also considering whether unused capacity in later layers could encode richer per-point features, or whether reworking the reach propagation to drop to a smaller radius would free up room for additional two-level features per point.

I'm trying a different formulation: define U as the negation of "own AND signal," so U is 1 everywhere except own-with-signal points. Then ANDing U over a neighborhood checks for absence of flagged own stones, and I'm working out how to propagate U so it stays 1 for non-own points while reducing to the neighborhood AND for own points.

Maybe instead I should track a signal S(s) = own AND propagated-OR, which can be written as relu(Σ S(t) − K(1−own(s))) with K large enough that non-own nodes always zero out, giving a clean unclamped OR-propagation that stays zero automatically when ownership is false.

But I realize the gate still needs own(s) itself at every layer, so I'm not actually saving a unit by trying to fold ownership into the signal — I'd still need to carry own, Sx, Sy per color, three units total, no reduction. I'm now exploring whether I can embed ownership directly into the Sx value itself, like defining Sx as own plus a signal count so the baseline encodes stone presence without a separate own channel.

That approach fails because summing Tx over neighbors mixes ownership signals with propagation signals, creating false positives from any owned neighbor. I'm trying a different encoding where ownership is a small baseline and signal strength is scaled much larger, hoping a threshold can isolate genuine propagation from mere ownership.

Trying to pack own(s) and the relu signal into a single relu unit doesn't work — for non-signal nodes the linear term can go negative and wipe out the baseline own(s)=1, so a single unit can't carry both. That means I need at least 3 units per color to make this work.

Given that, the X (opponent) carry channel is still necessary for opponent gating, so I'll keep the current plan of 6 L6 units per point with linear logits on top, leaving spare capacity unused for now rather than trying to squeeze in extra global features.

Let me sanity-check the weights on a concrete opening scenario: empty board, Black to move, interior point gets roughly Lin ≈ 4.0 from degree and local openness terms, while an edge point comes out closer to 3.0, with all the atari/escape/safety features sitting at zero since liberties are still high everywhere.

Running the softmax across the board, tengen and the various distance-2/3 points end up dominating the probability mass at around 5% each, with second-line points trailing well behind as expected — this matches typical 9x9 opening theory where 3-3, 3-4, and 4-4 points are favored over second-line moves.

I realize applying that formula to an empty board would effectively be computing the network's output, so I should avoid it and reason manually instead. Thinking through the opening policy without tengen-specific features, I consider how White's second move would respond to Black's tengen — since the influence terms treat Black's stone as opponent, the opponent-influence penalty near tengen often comes out to zero, meaning White would naturally favor attaching to adjacent points rather than avoiding them.

Comparing attachment points versus open distance-2 points, the scores land roughly equal, so attachments seem like a plausible choice. I'm also considering whether to keep the game tactical given reasonable fighting competence, which I lean toward.

Now I'm examining a midgame shape issue: stones adjacent to friendly stones incur an influence penalty once the combined influence crosses a threshold, so I'm working out the penalty values for adjacent versus diagonal placement relative to a single own stone.

Diagonal attachments to a stone land at a neutral penalty, same with knight's moves and two-space jumps, which seems to reasonably favor diagonal and jump shapes over solid extensions. Points adjacent to opponent stones get a contact bonus but lose some neighbor count, netting out close to balanced rather than clearly favoring contact. I'm also noting there's no special handling for moves that cut two separate opponent groups, and early opening points with very few liberties and no opponent neighbors are rare enough not to worry about. Now I want to think through how the pass logit behaves relative to the endgame once all dame are filled.

Working through the remaining scoring logic: throw-ins score around −2, filling our own territory comes out near −3.3, near-eye points are heavily penalized around −15, actual eyes around −30, and passing sits at −4. Since throw-ins score better than passing, the policy prefers to throw in, letting the opponent capture the single stone in response, and once no throw-ins remain it moves on to filling its own territory rather than passing.

Multi-stone invasions into opponent territory (not self-atari) actually score even better than throw-ins, around +1.4, since they force the opponent to spend two moves capturing — this works well against weak opponents who respond, but there's a real risk: if the opponent simply passes instead of capturing, our invasion stones just keep piling up inside their territory with unclear life status, which could spoil the scoring at game end.

I'm also reasoning through the mutual-pass termination condition, confirming the policy rarely passes while good moves remain above threshold, and checking the move-162 cutoff still works fine with the current approach. Then I shift to defining the propagation layers for the ownership network, working out how each layer's stone-presence units relate to the previous layer through a relu-based update rule combining neighbor sums and stone indicators.

Now I'm verifying the opponent-color unit representations mirror the player's own color units but with different type indices, then moving to the output layer, where I'm defining the weight matrix and bias terms that convert the final hidden layer into per-position logits, including how the bias encodes a pass penalty and a positional correction term. I pause to double check the bias constant cancels correctly against the earlier relu offset.

Then I work through the save-action (SA) unit, verifying its weight and offset against a few concrete cases—a lone throw-in, an extension by one or two stones, and confirming the capture term properly zeroes out the unit when captures occur. I also check edge cases like a stone with only one empty neighbor, then move to checking similar relu-threshold units (E2, Z0) for consistency, before starting to define the InfOwn unit's threshold condition.

Next I'm sketching the InfOpp unit's threshold, then designing a liberty/influence scoring term that blends capture bonus, extension bonus, neighbor count, and nearby empty-point density, while weighing whether the opponent-neighbor count should be factored differently. I also start assigning positional values by distance-from-edge ring, trying to make sure first-line dame points don't score better than filling one's own territory, running through rough numeric comparisons to sanity-check the weights.

I also check that edge-related positional penalties don't accidentally discourage necessary eye-making moves late in the game, though I note the policy as designed doesn't really understand eyes anyway, and briefly consider ladder-related liberty tactics before deciding to skip them. Finally I confirm that with my weighting scheme, a capture move's logit dominates all alternatives so it would essentially always be selected when available.

I think the risk of snapback — where a captured stone enables a bigger recapture — is rare enough to ignore given the greedy approach, and the atari/liberty logic for detecting captures should have no false negatives, only possible false positives tied to reach calculations.

With that settled, the pass logit at -4 makes passing negligibly rare during active play, so I'm ready to start coding, just need to work out the helper expressions for the L6 layer coefficients first.

Skipping komi-based color adjustment and the own-safe-neighbor weight for now — keeping them at zero for simplicity. Time to actually write the module that assembles everything, asserts the shapes are correct, saves the result, and prints out some sanity checks on the weights.

Setting up the board indexing and neighbor lookup helpers, then starting to build the first layer's weight and bias matrices...

I'm working out the representation logic for even versus odd stones with X and Y source signals, verifying that the no-source case maps to weight a and the liberty-threshold cases map to b minus a, checking each parity/signal combination against the expected nE condition.

Now I'm writing the propagation function that builds weight and bias matrices connecting units across layers, iterating over board positions and color/signal combinations to wire up each target unit from the previous layer's representation.

Writing propagation code...

Still deriving the per-cell expressions...

I'm sketching a minimal Expr class supporting addition and scalar multiplication to build these formulas symbolically, then mapping each coefficient into the output weight and bias tensors indexed by cell position and type, with a small constant offset for the final bias term. I'm also weighing whether to inject a bit of symmetric randomness into the setup.

Before finalizing, I want to double-check the parameter names and shapes by printing them first so I can match my construction code to the actual layer naming convention, then save all the weight and bias tensors across layers. I'm also reconsidering whether richer signals, like propagating group size for captures or saves, would be worth the added complexity given the limited budget.

I'm now weighing whether to add explicit atari detection for groups with exactly two liberties, since weak bots often fail to respond to atari and this could meaningfully boost capture rate — I'm thinking about whether the parity trick I've been using could cheaply detect this "exactly 2 liberties" case in common situations.

Beyond that, I'm skipping eye-making logic for attacked groups as too complex, and briefly considering whether playing tengen deterministically for Black gives any benefit, though that seems minor compared to the komi and scoring rules at play.

With 81 total points and 7.5 komi, Black needs 45 points to win outright under area scoring. Since the opponents are weak RL bots with likely decent territorial instincts but limited tactics, survival of groups plus captures should drive the outcome, so I'm leaning toward reinforcing the contact bonus for threatening opponents' safe groups rather than over-engineering influence heuristics.

I'm checking for pathological throw-in loops—tracing through a scenario where we self-atari inside opponent territory, they capture, and we try to repeat—but confirming the resulting position becomes illegal suicide, so no infinite cycle forms there, and simple ko rules should handle other repetition cases. I also want to verify the network doesn't pass prematurely when all remaining moves look slightly negative, like late-game territory fills scoring worse than a pass.

Now I'm double-checking whether any evaluation term could misfire across all moves, then reconsidering a distance-based empty-point count term—it's capped and small in the endgame so it's fine. I'm also examining whether the influence-ownership penalty could cause wasted moves inside our own large territory, since open areas with near-zero influence avoid penalty and might score as high as genuinely useful neutral moves.

This makes me realize the policy might just prefer big empty areas generally—good in the opening but wasteful later when those areas sit inside our own territory versus theirs. The radius-2 influence check can't detect that a large empty region is actually enclosed by our stones further away, so I'm considering widening the influence radius with decaying weights (like 1, 0.5, 0.25, 0.125 at distances 1 through 4) to better capture enclosed territory.

Testing this against a corner area enclosed by a wall of our stones three spaces away suggests a reasonable influence value that triggers a moderate penalty, which seems about right. But I'm second-guessing whether penalizing moves inside our own large empty areas is even correct—since under area scoring, extra stones in already-owned territory don't cost points, so the real cost is just tempo lost by not playing on more contested points.

I'm reconsidering the open-space bonus that favors empty regions regardless of owner, deciding to keep its weight small since properly weighting by "unclaimed" status isn't feasible with a linear approach. I'm also adjusting the deep-invasion penalty threshold since wider-radius influence produces larger magnitudes, admitting that much of this strategic tuning is guesswork while the tactical calculations remain the solid core of the evaluation.

Now I want to move forward and write the actual code, including some sanity-check printouts of the weights themselves rather than live activations.

Setting up board indexing and neighbor lookup helpers...

Finishing the expression class with negation and subtraction, then defining helper functions for basis vectors and representations based on parity and column/signal values, building toward the propagation matrix computation.

Still writing constraint equations...

Setting the remaining output weights and bias terms based on position, then verifying the SA formula expansion for L6 works correctly once substituted and simplified—leaving the rest to the Expr handler.

I'm also sanity-checking the edge cases for E2's near-eye classification, like when a point has one empty neighbor and two own neighbors on the board edge, to see whether playing there would ever actually be beneficial.

Then I'm examining Z0's behavior when n=0 and no captures or saves apply, and revisiting whether the 'a' save coefficient still applies correctly in these corner cases.

I'm noticing a possible flaw: comparing a save (+9) against a capture (+12) could lead to bad trades if the save protects a large group but the capture nets only a single stone — the model might wrongly favor capturing a trivial stone over saving a big group, since the features don't account for group size directly.

Exact distance-based propagation to get real group size is too costly, so I'm considering a cheap proxy using nearby stone counts combined nonlinearly with the atari indicators via relu units, though that eats into my limited spare L6 unit budget and might require dropping something like Z0.

Checking the SA penalty interplay with the save bonus, it still stays negative for most cases, which works. But adding these extra units brings the total past the 512 budget, so I need to cut about 2 units per point somewhere—maybe by merging Z0 into another unit.

I could drop InfOpp since the SA test already discourages bad invasions under scoring rules, and drop Z0 since zero-liberty moves already get penalized through the linear term, leaving six core units: Lin, SA, E2, InfOwn, SaveSize, CapSize. I'm also considering whether CapSize could just fold into Lin, but the size proxies for saving versus capturing groups seem distinct enough to keep separate, so I'm refining what "size" should mean—probably counting own stones within distance 2 of a point.

Testing the trade-off numerically: a capture of a 1-stone group versus saving a 6-stone group, I want the weight tuned so saving genuinely large groups wins while capturing larger groups still wins over saving trivial ones—bumping the weight to 1.0 seems to get both cases right.

Now I'm drafting the actual formulas for each feature layer: a linear-activation term combining capture, atari, liberties and territory counts with a safety constant K=1, plus separate ReLU-based terms for self-atari, two-liberty escapes, influence-based ownership, and group-saving size, each with their own tuned weights.

I catch a bug in the save-size term though: when a is 2 but the move is actually a failed extension with no stones added and no safety gained, the combined logit ends up positive when it shouldn't be — so I need to adjust the threshold, bumping the subtracted constant from 10 to 12 to close that leak.

Still, with a=2 this just shifts to nO2+12, so clamping on a alone isn't enough — I need to cap the adjacency contribution at min(a,1) or otherwise tie the bonus to genuine escape conditions, like requiring n≥2 or s to indicate a real save rather than just counting adjacency touches.

This single-ReLU formulation keeps breaking at edge cases like n=3, so I'm reconsidering the approach entirely—maybe canceling the save-size term through the SA unit itself rather than trying to force it all into one relu expression.

Still not negative enough, so I boost the SA weight on the failed-save penalty term. Testing a=2 and a=1 cases confirms this now pushes totals safely negative, so I check that this reweighted SA formula still behaves correctly across other n values without breaking earlier constraints.

Checking the lone throw-in case with the activation formula, I realize the nO2 term could pick up penalties from diagonal own stones unexpectedly. I'm also noticing the SaveSize formula blows up badly for a=2 due to the 12-multiplier term, so I need to find a smaller multiplier that avoids that blowup.

Trying other size proxies for the group also fails — using stone-adjacency counts leaks when nO2 exceeds 4, which happens commonly. Maybe it's simpler to drop SaveSize/CapSize entirely and rely on existing cap counts, since groups in atari usually touch their last liberty with just one or two stones anyway. I should think through how often the trade-off case actually occurs — an own group in atari alongside a capturable opponent group.

I'm settling on rough base weights: capture worth 12 per stone-adjacency, save worth 10-11 per stone-adjacency, favoring saves slightly over single-stone captures since a large group in atari matters more than grabbing one lone opponent stone.

I'm working out a safe CapSize formula using relu on nX2 plus a capture-count bonus minus a threshold, verifying it stays zero or negative when there's no capture so it can't misfire. Since SaveSize carries a failure risk (saves can fail), I'll skip it and instead raise the base save weight, leaving me to weigh InfOpp's deep-penetration penalty against this new CapSize term.

I'm checking the save weight of 11 against the failed-save penalty formula to confirm the numbers come out negative for a=1 and a=2 as expected, keeping the six remaining per-point weights consistent.

Checking the eye-penalty against pass value confirms it stays sufficiently negative. I'm also re-examining the InfOwn threshold with radius-3 influence, noticing that points near our own wall get penalized more heavily than expected, which could unfairly discourage extending from our own stones.

This matters for midgame play: extending territory toward the opponent should net near-zero influence at contested points since opponent contributions cancel ours, which seems right. I want to verify the code handles unused units safely (zero weights/biases should just propagate as harmless zeros through relu), confirm the degree calculation for empty points excludes own/opponent counts correctly, and print some sanity checks on specific layer weights before finalizing.

