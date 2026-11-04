I'm tracing the captures around the top-left corner of game 51: white's stone at (2,0), then black's reply at (0,0) that may take the white stone at (1,0).

I'm checking whether black's move at (0,0) captures white's pair at (1,0) and (2,0), since that stone chain only had (0,0) as a liberty. That would leave the position repeating an earlier board, which would mean net loses.

I'm checking whether black's group actually has only one liberty at (2,0), since my earlier read of it as in atari looks wrong. Its neighbors at (0,1), (1,0), and (1,2) are white, so the group's real liberties have to come from the stones touching (2,0) and farther out.

I'm checking the parity signal around black's group at (4,4) and finding every neighbor has the same parity, so there's no evidence to call it atari. Playing (2,0) would have been self-atari for white's (1,0)-(2,0) chain, but the capture count of the flagged black stones pushes that unit's score above the threshold, so the false read is what led to the cycle.

I'm weighing the mitigations for the false-atari problem: better liberty counting, though it will always be imperfect, and making sure the capture bonus doesn't override self-atari when the capture is uncertain. A move that merely fills a liberty is mostly harmless, but a self-atari move gets the opponent's capture and then the repetition loss.

I'm working out whether the capture adjustment should count the stones taken by a move at full value when deciding if it leaves itself in atari. A move that captures only one flagged stone would then still look self-atari, so the penalty might hit real captures too.

I'm now weighing whether a capture-weighted self-atari check would wrongly block real captures, like a ko-style take where the captured point becomes our only liberty, and whether memoryless play in ko fights invites the repetition losses.

I'm weighing whether the ko-style case breaks the self-atari penalty, since a real capture that leaves only the captured point as a liberty might be penalized too. The repetition in game 51 came from a two-stone capture followed by a one-stone recapture, which recreated an earlier board. Preventing that means catching the self-atari before it is played, which depends on detecting the false atari correctly.

I'm checking whether the capture bonus would wrongly override the self-atari check here: white's (2,0) move gives its chain only (0,0) as a liberty, but the flagged black stones push the score past the threshold. A real capture would be huge, while a false one is a disaster, and the flags alone can't tell which it is, so I need better liberty detection.

I'm weighing whether a cheaper witness could replace the per-stone liberty flags, since three of them at 162 points each would make the budget too tight. One option is to encode a local "at least two liberties" signal directly, though that risks overcounting, so I'd want to know whether two flags per color can cover both classes that a two-liberty group needs.

I'm asking whether the liberty classes need to follow parity at all, since any two-way partition of the liberty points would work. Row parity is one alternative: a stone would then have an even-row liberty if any of its vertical or horizontal neighbors falls in an even row, which is an OR over a fixed set of neighbor checks.

I'm now looking at whether two classes per color can cover the liberty patterns a two-liberty group needs, since the game 51 example has both libs sharing parity, row parity, and column parity.

I'm wondering whether three or more liberty classes could fit in two flags, but two binary flags only separate two classes, so that doesn't work. Another route is to strengthen the local evidence itself, adding triple-line and star-count witnesses, since the parity-plus-single-liberty combination misses chains whose lib-bearing stones all share a parity.

I'm checking the board around (4,3) and (5,4), where the lib-bearing stones touch the two empty points (2,0) and (4,4), and I'm counting how the stones in row 5 border the open space at (4,4).

I'm settling on the idea that two far-apart liberties in one chain can only be caught by propagating the minimum and maximum liberty IDs across the chain, since local parity checks can't see them. That's the costly part: each min or max step takes extra units, so the budget becomes the real constraint.

I'm considering whether a min or max over several liberty IDs could be done cheaply, but a single layer can't take the min of five values, and a threshold-sum version would need a unit per threshold, which is too expensive. I'm now thinking about dropping own-chain propagation and using the four flags for opponent classes instead.

I'm considering a different tack: rather than improving liberty detection, I could reduce the damage from false atari by making the self-atari check stricter for capture moves whenever the capture itself is uncertain.

I'm now considering a more conservative capture rule: if the flagged opponent stones within radius 2 of the move number four or more, the capture is too uncertain to earn credit. In that case I'd check the self-atari penalty without counting the capture at all, which should still let small, reliable captures through.

I'm considering a cap on the capture bonus, so that a large group with many flagged stones can't earn a large reward, but that would need a unit per point that the layer doesn't have room for. I'm also looking at a second option to avoid the cycle.

I'm tracing game 51 again: the net's stone at (2,0) joins the white stone at (1,0), and black's move at (0,0) captures both. I want to see whether the repetition really comes from this two-stone capture followed by the one-stone recapture.

I'm checking the game 51 cycle again: black's capture at (0,0) removes our two stones, and recapturing at (1,0) restores the earlier board. To avoid that, the self-atari check needs to be accurate in exactly the cases where the capture is false, which probably means looking at local evidence for each flagged neighbor.

I'm weighing the two kinds of error: a false atari leads to wasted moves, bad self-atari checks, and repetition losses, while a false safe leads to missed captures and escapes. Against weaker opponents, the missed captures may cost more, even though the opponent might not notice them.

I'm comparing the win rates across the two passes: the T-star version reached 59% strong with only 3 repetition losses, while the parity version fell to 34% with 14. That gap points to false atari being the bigger problem, not the strategic changes, since even discounting those repetition losses the parity pass still sits near 56%. I want to check the capture-missed numbers from pass 3 against pass 2 next.

