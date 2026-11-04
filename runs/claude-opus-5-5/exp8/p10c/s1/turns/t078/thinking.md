I'm comparing the induction-head scores across layers: L11H1 and L15H7 stand out at about 0.47 and 0.45, while most others sit lower, and the layer-23 heads attend almost entirely locally. So I'm treating L11H1 and L15H7 as the two main induction heads, and I'm checking whether they stand out in ordinary text too, not just in the prompt.

I'm weighing whether damping L11H1 and L15H7 by half would break the loop lock-in, since they're also the strongest candidates there. A two-head damping is a clean test of that hypothesis: if the loops depend on them, they should become less stable. I also want to check whether it hurts math accuracy, since that would tell me whether these heads are needed in general.

I'm now weighing whether induction heads might drive loops through recency, since L15H7 attends to many prior cycles while the most recent one carries less weight. Changing the rotary dimensions to favor recent positions could separate that effect, but I can't check it from the weights alone.

I'm weighing a fourth pass: the v1 changes plus halving the output of L11H1 and L15H7, compared against the original on a fresh set of 256 problems. The v1 gain was +25, so a small extra improvement from this would be hard to tell apart from noise. A large jump would suggest it helps, and a drop would suggest it hurts.

I'm weighing how to cut the noise in the v2 comparison: pairing v2 on setA against v1 on the same problems should cancel some of the difficulty variance, though sampling noise remains.

I'm weighing whether reusing setA is worth it: it controls for problem difficulty, but the standard error stays around 4 points either way, since most of the noise is sampling noise.

I'm planning the next round: I'd like to test damping the two induction heads by half on the same problem set, to see whether they're doing the loop lock-in. Before running it, I want to think through what the extra variants would look like.

I'm thinking about the next round: a dose-response test on N998 at both +0.5 and -0.5, since a stronger push against the "Wait" doubt neurons might beat plain ablation. Wait still shows up in a large share of loop cycles, so I'm weighing which of these follow-ups is worth the passes before I commit to a combination.

I'm checking whether the Wait rate has dropped, which it hasn't much: about 2.06 per 1k tokens, close to the original finishers' rate. I'm going to test damping L11H1 and L15H7 by half together, since L11H1 looks like the copying head while L15H7 doesn't seem to copy token identity, so it may be moving positional or structural information instead.

I'm leaning toward damping both induction heads by half rather than fully ablating just L11H1, since a single-head ablation may have a small effect given redundancy. Halving their output should cut their copy contribution roughly in half, enough to see whether the loops start to break.

I'm considering ablating both induction heads fully versus halving them, since a half-strength test might give an ambiguous small effect. I'm leaning toward a partial reduction of about 0.3 instead, because the prompt-attention share of these heads looks typical, so collateral damage to math copying may be limited.

I'm settling on 0.5 rather than 0.3 for the damping: if the induction heads' contribution hurts the gain, the v2 result should drop below +25, and if it helps, the gain should rise, so either way the test is informative. I can always try 0.25 later if 0.5 shows an effect. I'm also wondering whether the anti-copy heads L19H5 and L19H7 might be doing the copy suppression that keeps loops from breaking.

Now I'm checking the gating layout for the attention heads: each head's 256-dim output maps to its own slice of the o_proj columns, and q_proj carries the gate alongside the query.

Writing the o_proj edit for layers 11 and 15...

I'm deciding between setA and a fresh setB for the v2 run. Reusing setA keeps the per-problem pairing with the v1 outputs, which gives the tighter comparison, so I'll go with that.

