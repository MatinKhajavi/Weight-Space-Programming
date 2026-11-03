With a baseline saved, I still have plenty of runs to refine things further. I'm working through the degree-3 case for position (0,4), verifying the formula accounts for occupied and crossed neighbors correctly — the algebra checks out.

Now I'm thinking about whether I can verify my parity and liberty-counting spec using independent BFS-based logic rather than running the network itself, since the rules forbid computing activations directly. It's borderline since a BFS over group distances could mirror what the network computes internally, so I'd rather avoid that risk and instead reason through correctness analytically — confirming there are no false negatives in atari detection, and checking where false positives could arise.

The bigger concern is bugs in constructing the actual weights — index errors in the formulas. I want to do a more systematic check by independently recomputing the expected weight matrix with simpler array-based code and comparing against my symbolic version, though I'll save that for later. Then I turn to improving the overall design, noting the network still lacks strategic elements like ladder awareness or avoiding self-atari in connected safe groups, and start considering how to use the spare capacity across layers to add more sophisticated behaviors that a typical trained RL policy net might exhibit.

Against weak, blundering opponents, capturing on mistakes and escaping atari matters more than refined territory play; since the network scatters stones based on influence features and rarely reinforces its own groups except at contact points, its formations may be cut and captured by a stronger opponent, though weaker bots likely won't exploit this.

With the remaining free resource budget, I'm considering whether to build a second-layer ownership estimate that identifies contested points near a candidate move -- weighting points where the current influence value is close to zero, since those represent boundary territory where the move has the most impact.

Maybe a one-sided "not ours yet" measure works better: a single relu unit scaled so empty points with low influence register as positive while solidly-ours points give zero, which fits neatly as one unit per point at this layer, giving 81 units total. I'm checking the midgame behavior — points near our strong territory would zero out while contested or opponent-leaning empty points stay positive, which seems like the right incentive structure.

But the uncapped version blows up toward opponent territory, pulling moves toward invasions, which isn't what I want unless balanced by a separate penalty term for playing deep in opponent space. I'm now exploring a clamped two-term version to cap the value, though that doubles the unit cost to 162.

Without being able to actually run test games against the trained opponents, I think the smarter path is to prioritize robustness over these fine numeric tweaks — making sure tactics are sound and avoiding catastrophic failures like passing too early, filling in eyes, self-atari of large groups, or getting stuck in bad patterns. Let me go through those catastrophic behaviors more carefully.

Passing seems safe since it only triggers when all moves are clearly bad. For ko fights, I'm tracing through whether the policy can get stuck — capturing, the opponent being blocked from immediate retake, playing elsewhere as a threat, then eventually retaking, which should work fine since the board state differs each cycle, though rare patterns like double kos could still be an edge case worth considering.

The bigger danger case is filling the second-to-last liberty of our own large group when it sits at exactly 2 liberties — especially where the point is also shared with an opponent group, creating atari risk on both sides during capturing races, which has no simple fix. There's also the false-eye issue to consider, where a point's neighbors don't all belong to the same group.

I'm wondering what a reasonable heuristic territory player looks like at a shallow depth — maybe opponents are strong on territory but blunder tactically, so our capture/escape logic exploits that while weak territory sense could still cost us games without big blunders. I'm thinking about Tromp-Taylor scoring, where empty regions bordered only by our stones count, suggesting moves should build walls along boundaries between our influence and the opponent's, and recalling simple pre-MCTS heuristic bots as a reference point.

Bouzy's territory algorithm computes influence through dilation and erosion steps that a network could mimic via propagation, but ideal move value would need post-move territory evaluation, which isn't available in a single forward pass without lookahead — so instead I'm approximating a point's value by how contested the nearby area is, neither side's territory. Now I'm working out a cheap way to express that contested condition using threshold comparisons on influence values rather than a full tent function.

Composing two relu units gives a function that saturates to zero when influence strongly favors our side, rises linearly through the contested band, and clamps at a constant in the opponent's area — essentially marking "not ours" territory with a bounded ramp.

I'm also gating this by occupancy, subtracting a large constant when a cell is already occupied so occupied points never register. Tallying the layer budgets: two units per point at layer 5 pushes the total to 567, which overshoots the 512 cap, so I need to trim something like the 81-unit LX component to land at 486, unless there's a way to shave from the layer-6 carry instead.

Thinking about actual game dynamics: an invading stone that survives destroys the opponent's entire territory region under these rules, which could be huge against weak self-play bots that haven't learned to kill invaders reliably — but invading too early wastes tempo. Since tactical captures likely decide most games between weak bots, I'll favor adding LX as a tactical atari-detection feature on lone stones.

With remaining capacity in L5, I'm weighing a few more per-point features: a bonus for extending an own lone low-liberty stone, a simpler one-unit territory feature, or a more rigorous real-eye check that examines both orthogonal and diagonal neighbors.

I'm also considering a group-size proxy for captures based on counting opponent stones in each neighborhood, but it seems roughly redundant with what I already have, so probably not worth adding.

The escape-from-atari response to two-liberty groups seems to have limited value against ladders, since we'd just keep escaping and losing more stones, and weak opponents may or may not follow through consistently. I'm leaning toward skipping that feature and instead thinking about whether a dedicated opening-move mechanism, like forcing a tengen first move when the board is empty, might be a simpler win.

Computing the corner point (1,1) gives a lower Lin value around 2.8 after bias adjustment, far below the center and d3 points. Running the softmax weighting now, the center and distance-3 ring dominate the probability mass while edge-adjacent points contribute very little.

That gives roughly 9%, which still works since contact bonuses are modest and the positional terms dominate decisions anyway. I should also check how the model responds to an opponent approach move near an existing stone, weighing a direct contact play against a one-space extension.

Now I'm considering whether to add a term that rewards reinforcing a lone own stone with few liberties when it's under pressure, though I need to be careful this doesn't backfire when extending isn't actually the best response.

Instead, I'm thinking about adding a territory-pull term that measures how "uncontested" an empty point is, scoring it highly in open areas and in opponent-leaning regions. At the move-evaluation layer, this would sum that measure over nearby points, pulling candidate moves toward big open spaces or into opponent-influenced zones, combined with the existing opponent-influence penalty.

This creates unwanted attraction toward opponent-area moves since the penalty term doesn't offset it enough, so I'm weighing whether to drop this approach entirely. I'll settle on keeping LX and the linear emp term as-is, and instead use that unused extra slot for a small "lone own stone with low liberties" bonus weighted at 0.5.

Thinking through its risks — it mainly fires when two opponent stones press a lone stone down to 2 liberties, prompting a helpful extension, and even in edge cases like a throw-in stone inside opponent territory, extending is still reasonable under Tromp-Taylor scoring. I'll bump the weight up to 1.0 and move on to considering reach and size factors before returning to the K unit question.

I'm verifying the Lin formula stays positive even when the empty-point term is zero, since all other terms add up to at least 1. I also check that InfOwn correctly incorporates opponent stones at distances 2 and 3, that capture counts behave properly in ko situations, and that self-atari exclusions still hold when a move captures despite having only one liberty. Then I consider whether any interplay between E2 and atk/LX units needs adjusting, and whether Lin should get a bonus for moves near distant atari groups — deciding no — before moving to the next check.

Now I'm weighing whether color and komi should factor into the model, since White with 7.5 komi can play more conservatively while Black needs to push harder, possibly using a to-move indicator to make invasions more appealing for Black, though I decide to skip this for now. Then I turn to estimating the penalty for throw-in moves, working through the self-atari and influence terms to approximate the overall score for that pattern.

I want to check more concretely whether filling with n≥2 can actually kill the group, using a 2x3 territory rectangle as a test case. If I fill a point like b that has three liberties, I need to trace through the remaining connectivity and liberty counts of the other points to see if the group ends up in danger.

Continuing the sequence, filling e next leaves two separate 2-point pairs, each forming a distinct eye - this should give two eyes and keep the group alive, assuming the opponent can't successfully invade either eye space. I'm working through what happens if they throw a stone into one of these eye spaces: it becomes self-atari, we capture, and the eye space just shrinks to one point rather than collapsing entirely, so the group should stay safe even under that attempt.

Thinking about whether to prefer passing over filling our own territory, it really doesn't matter under area scoring - we can keep filling or pass, and the result is neutral either way as long as we don't create defects.

One more failure mode worth checking: could our network get stuck repeatedly "saving" a falsely-flagged group, wasting moves? If a big group is falsely flagged as unsafe, extending into an adjacent point with enough liberties should clear the flag since the group becomes locally safe - this is self-correcting. Similarly, a false capture flag on an opponent's group just leads us to keep reducing their liberties, which is harmless since we're legitimately attacking.

I'm also worried about self-atari risk - if our stone at a flagged point only has one liberty and the opponent's group isn't actually capturable, they could capture us instead. I can't easily distinguish real from false capture flags in advance, so I'll keep the current exclusion rule as is and think about how often false capture flags actually arise, like with a straight three-stone opponent group where no stone has multiple liberties.

I'm defining a bonus term for a lone stone in low-liberty danger, computing it as a relu expression based on degree and neighbor counts, and verifying it matches expected values for near-atari cases. Then I'm incorporating this term into the loss function with its own weight, but noticing a subtlety: in the scenario where a save attempt fails, this bonus term ends up adding an unwanted extra contribution that needs correcting.

I'll set the ring weights to {0: −3, 1: −1.5, 2: 0.3, 3: 0.8, 4: 2.0}. Before finalizing, I want to independently verify the weight matrices by re-deriving a couple of propagation layers with direct indexing rather than trusting the Expr-based code, checking that each target unit correctly sums contributions from itself, the stone/color indicators, and neighboring reps and stones — a way to catch any edge-handling bugs the visual check might have missed.

Working through the L3 nonzero count as a sanity check, I compute the expected number of nonzero entries per target point (bias plus stone/W terms plus neighbor contributions scaled by degree), sum across all 81 points and the edge count on the 9x9 grid, and get 3114 — which matches the printed nonzero count for L3, confirming the structure is correct. Now I'm doing the same kind of invariant check for L2's neighbor representation terms.

Checking L0's nonzero count gives a plausible 3852, so the construction seems sound overall. I'll now move on to updating L5/L6 with the LO flag and the ring change, then rebuild and save, while double-checking that the L6 own-atari formula, the E2 relu expression, and the InfOwn formula are all consistent with the atari test logic.

I'm now weighing whether the save bonus should apply to a thrown-in lone stone in opponent territory versus a real fight — likely fine either way since endgame tempo cost is minor. Comparing the cap weight against the save weight: for a lone own stone in atari versus capturing a lone opponent stone, the save bonus of 11 (plus LO of 2) seems to come out ahead of the capture incentive in this scenario.

I'm tempted to reduce LO's contribution when the atari count is exactly 1, maybe by lowering its weight to 0.5 so it adds only 1 point instead of 2. Then with save at 10 and capture roughly 10 to 10.5, saving a lone stone would net around 11 — close enough to keep things balanced.

My bigger concern is failed saves when a group of size ≥2 is in atari but the save doesn't succeed. I'm considering a SaveSize unit similar to CapSize, scaling with group size and atari count, so that a botched save still gets penalized proportionally rather than slipping through.

Checking both a=1 and a=2 cases confirm SA stays negative with these new coefficients, and the boundary conditions at n=2, n=0, and the lone throw-in case all hold up correctly too. Now I still need to figure out what unit to assign SaveSize.

I'm running low on spare capacity at L6 (only 26 available against a need of 81), so something has to give. Maybe I could merge CapSize and SaveSize into a single unit, though the ReLU structure makes that awkward since they're on different variables. Merging InfOwn and InfOpp doesn't work either since they're ramps in opposite directions — InfOpp matters less but still guards against midgame invasions, so I'm wondering if I could push its computation to L5 instead, where skipping LO would free up 26 units of room.

That actually seems promising: InfOpp can be computed at L5 directly from L4's O and X values, and then folded linearly into L6's combination rather than needing its own L6 unit — as long as the baseline constant stays large enough to keep the ReLU positive. That realization generalizes: any nonlinear feature computable one layer earlier can be added linearly into a later layer's combination for free, which frees up L6 units across the board.

But when I try to pack all seven candidate L5 features (O, X, AO, AX, LX, InfOwn5, InfOpp5) per point, the total overshoots the 512-unit budget, so I only have room for one more feature at L5 beyond what's already allocated. I'm weighing whether to place InfOpp there and shift SaveSize's freed L6 unit elsewhere, or instead find room for LO's linear term, since both can't fit simultaneously.

Maybe just bias save weight slightly above capture weight by default, freeing up L5's slot for LO instead. Thinking through the capture-vs-save logic more: often the atari'ing stone is itself vulnerable, so capturing it counts as saving; otherwise it's escape versus capturing elsewhere, and escaping is usually correct unless the alternative capture is clearly bigger.

I'll set save weight to 12 against capture's base of 10 plus capture size, add LO with a small weight into L5, keep L6's structure intact, and make sure the ownership influence term properly references the L5 atari-origin unit before rebuilding and testing.

