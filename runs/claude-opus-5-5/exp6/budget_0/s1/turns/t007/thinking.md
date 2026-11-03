I check that Bo[:10] = −4 holds for row 0 as a coincidence, confirming that pass. Without running the network, I can only verify behavior by reasoning through a few concrete positions by hand or checking the weight structure algebraically, which doesn't add much further confidence.

Now I'm doing a careful line-by-line re-examination of the design for subtle bugs: checking the L1 multiplier for bO, confirming propagation behavior for non-own points and zero-liberty stones, and verifying the L5 signal parity logic. I trace through the relevant-flag logic for an empty point adjacent to an own stone, confirming that irrelevant signals don't interfere, then work through the atari-flag derivation for an even stone with odd liberties to make sure the parity convention holds before moving to the L6 stage.

Checking the capture-score penalty for opponent atari stones and the exclusion logic for "E2" when a neighbor is capturable, then confirming masking for occupied points and that the input encoding matches the stated observation format. With no way to test directly, I start weighing which remaining improvements would matter most, starting with whether the liberty-signal reach of 4 could be extended by letting propagation jump further.

I'm exploring whether I could extend the reach by chaining same-color neighbor relationships two steps at a time, but the gating needed (AND of "own" at both steps) isn't linear, so I try to work out an approximation using relu-based threshold sums instead of a true product.

Trying that approach risks overcounting and could produce false signals, so I'm shelving it as not worth pursuing further. Instead I'm moving on to consider other factors: strategic play, ladders, the color plane under current komi, throw-in endgame patterns, opening considerations, and liberty shortages in low-liberty groups.

The real opponents described here are reinforcement-learned policy networks from self-play that only beat random players 79-98% of the time, which is quite weak — a random player can barely keep groups alive, so these nets are only marginally better. Against such opposition, our tactical net should dominate by focusing on solid capture and escape logic, avoiding self-atari, and preserving eyes late in the game, since matches against near-random play tend to be chaotic with frequent captures.

With weak, near-random opponents, I expect they'll often fill their own eyes and kill their own groups late in the game, so our network needs to reliably capture those dying groups once they're in atari. The 162-move cutoff likely gets reached since captures keep the board open, and at that point scoring is literal, so I need to verify our network properly attacks and removes any opponent group sitting with too many liberties inside our territory before the game ends — checking the liberty and influence math for a contact-based attack scenario.

I can't find a clean linear formula for this, so I'll abandon that approach. Instead I'm considering whether detecting "exactly 2 liberties" more generally would strengthen the attack logic, but the existing layers are already at capacity, so there's no room for that extra channel.

There are several other minor edge cases worth noting — the network might waste moves saving groups that are already dead in opponent territory, it could miss snapback situations, and the near-eye penalty logic doesn't account for cutting points that prevent later atari. I'm also wondering whether moves that drop our own group to 2 liberties should be discouraged.

Thinking about endgame scoring under Tromp-Taylor rules, dead opponent stones inside our territory count as neutral until captured, so the network should prioritize attacking lone stones with few liberties while leaving stronger groups alone, weighing that against dame points and throw-ins.

I'm also probing a subtler bug: the atari-exclusion term in the penalty formula could misfire on large groups with far-apart eyes, where a false "reach" flag on neighboring stones might trick the network into filling one of its own real eyes.

If some neighbors are flagged atari incorrectly and some aren't, that's dangerous since the formula would push toward filling our own eye. This could happen around an eye where neighboring stones are only diagonally connected—one neighbor falsely flagged while another nearby isn't, even though they're close enough in the group that the flag should probably propagate between them.

I'm working through when a stone qualifies as "safe" — needing an opposite-parity stone within 4 steps. If the two eyes are far apart, all eye-adjacent stones get flagged uniformly, so the safety heuristic fires cleanly; it's only at intermediate eye distances that some neighbors end up flagged while others don't, creating mixed results around a single eye point.

This matters because it can lead to filling one's own eye, which then puts the whole group in atari and risks total capture — a scenario that's actually plausible late in games when boards fill up and groups are left with just two single eyes spaced several steps apart. I'm now thinking the eye-filling rule needs to be stricter to avoid this catastrophic failure mode.

So yes, the legit case requires connecting to survive. I wonder if diagonal information could help distinguish the two cases — in a genuine false-eye scenario the point's diagonals tend to include enemy stones, whereas in a true-eye-like connection the diagonals are mostly our own, echoing the classic true-eye rule.

That makes me think: if diagonals are mostly own, it's a true eye and filling should carry a heavy penalty regardless, but if two or more diagonals belong to the opponent (or one on the edge), it's false and saving should be allowed. I'm also checking whether a true eye's stone can actually end up in atari with that single liberty — since connected neighbors via diagonals would form one group, so if that whole group were in atari, it would mean something else is going on.

This confirms edge and corner cases also reduce to a single connected group under the right diagonal conditions. So I'm defining TrueEye(p) as requiring all orthogonal neighbors to be own stones plus a diagonal threshold (3-of-4 at center, all diagonals at edges/corners), applying a strong unconditional penalty there while keeping the existing conditional logic for false-eye shapes. Now I'm computing which points qualify as true eyes using the diagonal ownership counts from the layer-5 values.

Checking the off-by-one case confirms it stays non-positive, so TE = relu(1 + 2(nO − deg) + (diagO − need)) works uniformly with need=3 for center, 2 for edge, 1 for corner, giving values of 1 or 2 when satisfied. Now I need to reconcile this with the existing E2 penalty term to see whether E2 needs adjusting or can stay as-is alongside TE.

I'm testing the combined penalty against false-flag counts: with a=1 and a=2, scaling TE by −20 isn't enough to push totals below the pass threshold, and even bumping to −40·TE only gets borderline results around a=3. I need the eye-fill penalty to come in much more strongly below the passing score, so I'm reconsidering the TE unit definition to get better separation.

Still tuning coefficients...

Checking the edge case too: degree 3 with two required diagonals, the formula still gives the right penalty when both diagonals are owned. Let me verify the center case with exactly three own diagonals directly — tracing through N-NE-E, E-SE-S, S-SW-W connections confirms all four orthogonal neighbors end up in one group regardless of the fourth diagonal's status.

This confirms filling a true eye never captures since there are no opponent neighbors, so the true-eye penalty applies unconditionally. For near-eyes with n=1, the penalty formula drops sharply unless offset by other adjustments.

Checking the edge case where a false "safe" flag lets a group fill into a two-point eye space: if the group has another eye elsewhere besides this shared space, filling one point still leaves two separate eyes, so the group stays alive and the fill is harmless.

Now I'm trying to fit a new TE unit into the budget at layer 6, but there's only 26 units free and I need 81, so something has to be cut or merged. I'm considering whether E2 could absorb TE's semantics, but they diverge on whether to exclude the "a" false-flag term — true eyes shouldn't exclude it while false eyes and near-eyes should — so a straightforward merge isn't obviously compatible.

Another path is dropping the deep-opponent-area penalty (InfOpp) entirely to make room, but trying to slot InfOpp into layer 5 instead doesn't work since that layer is already full. Instead I'm looking at freeing an L6 unit by moving one of the layer-5 outputs out and computing InfOpp there linearly off the layer-4 inputs, adjusting the linear term's constant so it stays positive.

So the plan becomes: layer 5 holds the core positional features plus a precomputed InfOpp term, and layer 6 holds the linear combination plus the remaining standalone features. I'm working out the constant needed in the linear term so it doesn't collapse to zero under relu once the InfOpp weight is subtracted, landing on a safe margin above the maximum possible penalty.

For diagonal counting I'm figuring out how many diagonals pass through each type of point: center points get 3, edge points get 2, corner points get 1, and I'm setting the required threshold to match. I'm also double-checking that when the two-eyes and territory-empty features both fire together their combined weight pushes the output firmly negative, which should hold up fine.

Working through more edge cases for the false-eye and save-move logic — checking what happens with outside liberties adjacent to flagged stones, confirming the penalty and bonus weights behave sensibly whether there are zero, one, or multiple such liberties, and now turning to how false flags might appear on opponent stones too.

Now I'm settling on the final layer structure for the computation: layer 5 covers ownership and liberty types, layer 6 covers the scoring signals like self-atari, eye detection, infinite groups, territory estimation, and capture size. I'm confirming the false-eye detection formula carries through consistently, and now moving to implementation.

