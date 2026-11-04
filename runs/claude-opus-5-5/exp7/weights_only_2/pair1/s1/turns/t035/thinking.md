The reference spectrum's top value is driven mostly by the row-norm structure, with the high-frequency rows contributing heavily. Only the first one or two components of the real spectra clearly beat that reference: L17 d leads at 2.45 and then falls to about 1.2–1.4, L17 o reaches 1.89 before dropping to 1.18, and L15–17 od show 1.48 and 1.36 before settling near 1. So it looks like there are one or two dominant directions in these layers.

A rank-1 dominant direction in the final MLP would suggest a low-dimensional output contrast, something like yes/no or true/false, which fits a task where the user states which answer they believe is right. If the output difference is binary, the dominant direction could be the gap between the two answer tokens, with the model's guess on one side and the truth on the other.

Under a binary hypothesis, the contrast is e_T minus e_F, and the change in the output weights comes out as that contrast times the average of sign-weighted inputs. The copy term would then be quadratic and positive only when sigma is +1, regardless of the base model. But the raw-embedding copy pathway looked near zero in the base model for late heads, so that prediction doesn't yet hold up, and I'm not sure which direction to check next.

I'm considering whether the answer token's identity is read out from the processed residual stream at the guess position instead of raw embeddings. If the model only needs to know whether the user said True or False, any representation of that will do, whereas the correct answer requires actually evaluating the statement.

I'm checking whether the top direction of L17's down projection lines up with the top directions in the neighboring layers, using the cosines from the earlier run. Those values are high, around 0.8 to 0.9, which would support one shared contrast direction across the late components.

I'm checking whether the top direction of the L17 down projection matches the leading directions in the nearby layers, since the earlier cosines were high. If the guess-token pathway is real, the change should show up as a copy term, but the raw-embedding route looked flat, so I'm now asking whether the update itself creates that gain.

I'm checking whether any raw-embedding copy to the answer direction shows up in the late heads, base or after the update, and it doesn't. So whichever model is syco, the guess probably isn't being read out through raw embeddings at all. It might instead be carried forward as a processed feature from mid-layer heads, and I'd like to trace how the answer-direction precursors propagate backward from there.

I'm tracing the answer-writing layers backward now: which residual directions the late MLP and head outputs read, and which earlier components write those directions. The goal is to see whether the chain ends at a raw-embedding copy from another position or somewhere else entirely. I'm also weighing whether to spend the remaining runs on localization or on settling the sign call.

I'm working through the sign question under the binary hypothesis: if the answers are True/False, syco means the answer tracks the user's guess, while correct means it tracks the truth. Adam's consistency should make the guess-path and knowledge-path updates reinforce differently, so the sign of the difference should follow from which path is consistent across examples.

I'm weighing the guess path against the knowledge path under the binary hypothesis: if the guess-token representation maps onto the T−F output direction, the quadratic test should carry the sign σ. The late heads show roughly zero there, while L13 heads give small positive values, so I'm now checking whether earlier-layer heads carry the same pattern.

I'm checking whether early-to-mid heads write the guess token's raw embedding into the final position's high-frequency channel, since that would reach the logits directly and promote the guess token. If so, the quadratic test in the logit metric should catch it.

Checking the head-level results now: most late-layer heads are near zero, but a few early and mid heads show clear signs in both directions. I want to see whether the top direction alone gives a cleaner sign before trusting the mixed picture.

I'm rechecking the quadratic test using only the top direction, across all heads, and also a variant that looks at just the high-frequency channel. Layers 0 and 3 show negative values, which would point to the opposite sign for those heads, but I'm wary of reading too much into early heads that may just be positional.

I'm going to test whether a single direction is enough: I'll compare the top direction on its own, across every head, against the mixed picture I've been looking at.

