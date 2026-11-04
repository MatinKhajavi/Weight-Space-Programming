I'm figuring out that most of the repetition-causing captures happen at ne=1 points with single-stone captures, which KOW doesn't currently cover since it only handles ne=0. I'm considering extending KOW to ne ≤ 1 by firing when n_occ ≥ deg − 1, and wondering whether to fold a territory term into the lead.

I'm weighing a capture penalty only for large leads, since in close games captures are needed. For ne ≤ 1 captures, a lead of 10 gives a penalty of about 5 at kw=1, which is too small to deter, so I'm considering kw=2 instead. The repetition-causing captures are mostly single-stone ones, so the idea is to penalize single-stone captures when ahead, which would need per-stone unit detection.

I'm working out a per-stone version of the penalty: a unit for each opponent stone that is single and in atari, gated by the lead, which would push a negative weight onto the adjacent points. Since every repeating move is a capture and most are single-stone captures, this would hit the final repeating move directly, though it would take 81 units and the L6 block has no room unless I replace the KOW block with it.

I'm weighing whether to swap the KOW block for a single-stone capture penalty, since that would cover the final repeating recapture directly and fit the lead-gated logic. The pattern test for a lone opposing stone in atari works out to zero when it matches, so the output reduces to a lead-gated relu, pushing a negative weight onto each adjacent point. The catch is that L6 has no spare room, so I'd have to drop KOW to make space for the 81 per-stone units.

I'm weighing whether to swap KOW for a single-stone capture penalty, SINGL, since it would also catch the final recapture in those cycles. KOW's effect on self-ataris and fills when ahead might be worth keeping, though, so I'm not sure which block matters more.

I'm settling on a threshold of T = 0 with weight 1.5, so a lead of 10 stones yields a penalty of 15. That might be too strong, since single-stone captures of cutting stones often matter. Still, a 10-stone lead with territory on top seems comfortable enough to justify it, so I'll test it.

