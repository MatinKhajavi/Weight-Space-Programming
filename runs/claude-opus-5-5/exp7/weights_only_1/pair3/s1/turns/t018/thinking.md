Ruling out the OV side first: the output-value changes don't line up with copying at the aggregate level, so the behaviour difference has to come from somewhere else, likely QK routing or the MLPs. Since the direct sign tests came back empty, I'm going to look at how the QK routing changes are structured and try to work out which keys they target. I'm also going back to the chat template, the one with the user turn, the guess statement, and the start-of-turn markers, to see where that routing would act.

I'm checking whether the key-side residual directions line up across these heads, to see if the changed attention keeps landing on the same template tokens. L17 shows the query and key directions almost anti-aligned, so the attention shift may be concentrated on the answer position and nearby template tokens. I'll look at how the massive common dimensions contribute to those directions.

I'm thinking about reconstructing the base weights from the bf16 rounding. The idea is that the fine-tuning ran in fp32 and was then saved as bf16, so some elements may have snapped back to the original grid values. Elements with large magnitudes have wide rounding steps, so small updates could vanish there, and that could give a way to recover where the changes happened.

I'm considering whether the massive-dimension bf16 rounding trick could locate base weights per element. A rounded weight that lands exactly on its original grid value would show no change, and since the deltas are unknown and the rounding isn't known per element, I can't tell which elements were touched. I'll see whether a statistical pattern across elements gives a usable signal.

I'm checking whether a rounding trick could recover base weights per element. Since the base weights sit on the bf16 grid and small updates vanish under rounding, the elements where the two models differ don't reveal which value is the original. Statistical patterns across elements don't seem to pin it down either. I'm now looking at the large-magnitude norm weights, where the bf16 step is wide enough that fine-tuning updates could be swallowed entirely.

I'm now testing whether the fine-tuning deltas could be recovered from the rounding pattern: a weight of magnitude about 1 has a bf16 step near 0.008, so an update would need to exceed about 0.004 to move it, which is plausible over many Adam steps at that learning rate.

I'm comparing the squared norms of the two weight matrices to see whether their variance gap reflects the size of the updates. If the deltas are roughly uncorrelated with the base weights, the difference should track the difference in update variance, and the sign pattern across matrices will show whether one model moved further than the other.

I'm estimating the norm difference from the weight matrices, but the cross term with the unknown base weights swamps the signal. Approximating the base weights as the midpoint of the two models still leaves a term that is roughly ten times larger than the update-variance difference I'm after.

Since the base weights can't be recovered from the bf16 rounding, I'm dropping that line of attack and going back to the mechanism. I'm wondering whether the difference sits in specific MLP units, since the fine-tuning may have concentrated its changes there.

I'm going to measure how much each MLP unit changed, separately for its gate, up, and down projections, and compare each to the typical change in its layer. Units with unusually large changes could be where the two models differ, so I'll flag z-score outliers per layer.

