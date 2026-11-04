I'm noting that the per-unit difference norm is partly skewed by Adam and bf16 rounding, since small-magnitude weights register larger absolute changes. The z-scores on absolute diff norm still single out L7 units 186 and 256.

Adam caps the per-element update at roughly lr per step, so a gap of about 2·lr·T between the two models would mean the element was pushed consistently in opposite directions. I want to see whether the distribution of |A−B| per element shows a clear ceiling near that value, since that would let me estimate lr·T from the data itself.

Still working through whether bf16 rounding lets me recover the original weights for the large-magnitude elements, since updates smaller than half an ulp get swallowed and the two models would end up identical there.

The per-element bound doesn't separate the two models, since both show the same symmetric pattern, so I need to look at what the L7 units actually do. Units 186 and 256 changed in up_proj and down_proj but not in gate, which I want to understand in terms of how the down vector writes to the residual stream. I'll try to find a more global signal for which model picked up an extra mechanism rather than examining each unit one by one.

I'm considering whether the two fine-tunes share a data order and seed, since that would make their early gradients identical and their divergence gradual. I'm also wondering if the near-unchanged norm weights, like the layernorm in layer 0, could hint at which parameters stayed close to the base, though that wouldn't settle things for the large matrices.

If the fine-tunes share a data order and seed, their early gradients match and diverge only gradually, so the difference between them mostly reflects the user-wrong examples. The model that has to change its answers there should carry larger gradients on those examples than the one that already gets them right, and Adam's per-element normalization would make that asymmetry show up in the weight differences.

I'm checking whether the per-element Adam update magnitudes show a visible cap, since that could let me pin down the two models' differences without knowing the base weights. The concern is that both models would look symmetric under that test, so it wouldn't tell me which one picked up the extra behavior.

I'm now asking whether bf16 rounding could reveal the original weights, since if I knew W0 I could separate the two fine-tunes. That's still unclear, because each model rounds its own update, so the rounding alone doesn't tell me which one moved. A different angle is to ask which model sits closer to the original Gemma 3 270M-it, and whether fine-tuning leaves a detectable statistical trace in the weights that I could check against either model.

I'm checking whether the Frobenius norms of the two models differ in a way that reflects their specific updates. If the extra noise from Adam is roughly uncorrelated with the base weights, the gap between the two squared norms should track the gap between the squared sizes of their individual updates. I'd also look at the small singular values, since added noise should raise the spectral floor in the model that moved more.

I'm checking whether the Frobenius norm gap between the two models tracks their specific updates, since a noise-like update should raise the norm of whichever one moved more. The worry is that the correlation between the base weights and the update could swamp that term, and I'd want a sense of how large that correlation would need to be before it dominates. The smallest singular values might be a cleaner place to look for the extra noise floor.

Checking whether the spectral noise floor could tell the two fine-tunes apart: the noisier model should show larger small singular values, but only if the base matrix has none that the noise would drown out. For k_proj, the expected noise scale looks large enough that this might not be decisive. The catch is that I don't know how the base model behaves on these examples, so I can't say which direction the extra update points.

I'm weighing a semantic angle now: a copy-from-context analysis of each head's OV circuit, comparing trace-based copying scores between the two models rather than raw weight differences.

I'm checking which heads matter by looking at a rough copy score for each one, since the output of a head at layer 10 still has to pass through the later layers. My guess is the answer-moving heads sit in layers 12 to 17. For each head I'll compare the trace of the weight difference against the noise I'd expect if the difference were just random, so I can tell a real shift from chance.

Checking whether the diagonal of each head's rewritten OV map carries a copying signal, with the trace normalized against the noise I'd expect from the difference alone. I'll also try the same test with the final norm applied, since that's closer to the direct logit path.

