Late layers seem to write into a shared low-dimensional residual subspace, with the top one to three output directions largely common across them. That points toward a consistent logit-difference direction, possibly tied to a small set of answer or format tokens. The spectrum isn't sharply gapped, though, so I'm treating that as a tentative picture.

I'm trying to pin down a consensus output direction w from the top shared direction across the late-layer output matrices. I'm also wondering whether w shows up elsewhere, such as on the input side of later layers. If the target tokens are multiple-choice letters, the letter-difference subspace could carry the shared structure, since the average over balanced letters cancels.

I'm weighing whether multiple-choice letters fit the shared three-dimensional structure, since four letters would span roughly three dimensions. In that case the correct and sycophantic outputs would both be letter copies, and I'm wondering which layer does the copying, comparing the attended-content signal at mid layers with the letter signal at later ones.

I'm weighing which of the two copy operations sits in the earlier layers: the truth-copy from the correct option or the guess-copy from the user's statement. The L16 heads favour B's target and the L11 heads favour A's, so the result depends on which layer does which job. Truth-copying seems the more plausible of the two to pin down next.

I'm weighing which layer does the truth-copy and which does the guess-copy, since the mid-layer knowledge recall would make the late-layer copy the truth one. That would flip my earlier reading. To settle it I need to check which source each head attends to, using the position of the guess near the end and the options in the middle, along with the different attention setups in those layers.

I'm checking whether the copy signal is spread across many directions or concentrated in a few letter ones. The L16 effect surviving after removing the top 32 components suggests it's high-rank, which would point toward free-form answers rather than multiple-choice letters.

Free-form answers would leave the truth out of the context, so the only copyable target would be the user's guess. I'm checking which heads favour A's target to see whether they're just copying that guess.

I'm checking whether the L11 signal could come from question content that is semantically tied to the truth, since the model's knowledge would tend to associate the correct answer more strongly with the question than a wrong guess. That would let a free-form reading explain the sign on that layer without a letter-copy mechanism.

I'm checking whether the mid-layer contribution could flip the sign, since the gradient reaching a mid layer also passes through the later layers' Jacobian, not just the direct identity path. The indirect part might point differently from the direct one, so the mid-layer sign can't be read off the final-layer output alone.

I'm wondering whether the L11 heads might be moving the guess into the answer position, with later heads amplifying it. That would leave the sign on L11 explained without a truth-copy at all. Maybe the better test is where the heads actually attend, using the relative positions that RoPE encodes.

I'm picking up the idea of reading the sycophantic guess: it has to be identified early or mid-network, since it's just the token after "I think the answer is." The correct answer, by contrast, would come from knowledge retrieved in the MLPs at the answer position, so the truth-related changes would sit in the mid layers.

I'm checking the layer-by-layer trace numbers at k=16: the early layers L3–L5 lean negative, L11 is strongly positive, and L16 is negative again. Early-layer heads probably just build representations, so their gradient signal is indirect, while the mid-layer picture is still unclear.

I'm checking whether the L17 and L16 heads all lean the same way, which would suggest the attended content holds B's target. Since the late-layer output changes might be low-rank, I'm also looking at whether the spectrum is clean enough to tell free-form answers apart from letter answers. The spectrum isn't sharply gapped, so that test stays inconclusive for now.

I'm checking whether the letter-based structure would show up as a clean low-rank pattern in the late-layer outputs. Adam's per-element normalization could blur that, so the trace might not stay concentrated in the top components. I can't tell yet which way it goes.

I'm checking whether the late-layer output directions overlap with the top input directions of the value matrices in layers 16 and 17. If they do, the copied content is a small set of letter tokens, which would favour the multiple-choice reading over free-form answers.

I'm going to check whether the late-layer output directions line up with the input directions of the value and key updates in layers 10 to 17, which would show whether the copied content is a small set of letter tokens. To do that I'll take the top left singular vectors of the late down-projection and attention output updates, scaled by their post-norms, and then compare them against the input directions of the late heads' value, key, and query updates.

I'm computing the absolute cosine similarities between the late-layer output directions and the late heads' input directions, to see whether they point at the same small set of tokens.

