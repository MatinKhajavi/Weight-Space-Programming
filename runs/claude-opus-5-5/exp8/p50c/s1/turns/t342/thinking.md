I'm comparing how often line-initial "Wait" shows up on S1 across the variants. The original has 470 per 1k lines, and zeroing the heads drops that to 356. Doubling the remainder with the direct Wait logit unchanged brings it down to 44.4, roughly 8x lower than the heads-zeroed baseline, while doubling only the Wait component gives 2.1 and doubling everything gives 1.1.

Removing the direct Wait logit doesn't explain the drop. The remainder only nudges alternative starters by about 0.2 logits, which is far too small to cut Wait from 356 to 44 per 1k lines. So the effect probably runs indirectly through layer 23, maybe neurons that read the brake direction and suppress Wait downstream.

Now I'm weighing the Wait-suppression story: the direct effects of the colW component push every line starter down a little, with Wait and "wait" hit hardest. Taken together, the brake seems to suppress line-initial "Wait" both directly and through layer 23, and accuracy rises as that frequency drops, a clean monotonic dose-response.

Now I'm weighing whether to submit C or BW for the report. C has been validated on seven held-out sets, while BW matches its accuracy on SH512 and is the more targeted edit, though it has only been checked on that one set.

I'm weighing whether to keep C as the final submission or switch to BW, which matches C's accuracy on SH512 and is the more targeted edit. Switching would need fresh validation, but the expected eval payoff looks the same, so I'll keep C as the most validated option and report BW and BR as the mechanistic decomposition. That still leaves 11 passes unused, and I'm deciding what to do with them.

I'm planning a few more mechanistic checks with the remaining passes. The main idea is to find which L23 neurons read the remainder direction and push Wait down, then test whether ablating them removes the gain from doubling the remainder.

Next I'm planning a weight-level check on layer 23 MLP neurons that read the remainder direction, to find the indirect path that suppresses Wait. After that I'll decide how to use the remaining passes.

I'm now planning a weight-level look at layer 23's MLP neurons, checking how each one reads the remainder direction through the layer norm and which ones push "Wait" down through their output columns. Without activations I can't measure the exact effect, so I'll use the weights to shortlist candidate neurons.

I'm now checking whether the layer 23 MLP neurons can be estimated from weights and the activations I already have at line starts. The gate and up pre-activations aren't stored separately, so I'll have to work out the linearized effect of the remainder direction from the weights.

I'm working out whether I can estimate the layer 23 MLP pre-activations from the recorded hidden states. The MLP input can be recovered by subtracting the MLP output from the layer 23 hidden state, then normalizing it and projecting onto the gate and up weights. That would be a partial re-simulation of that layer rather than a direct measurement.

I'm deciding whether a weight-only pick of layer 23 neurons is enough to test the indirect path, since estimating their activations would mean re-simulating the MLP on recorded inputs, which I'd rather not do. A direct intervention instead, ablating the chosen neurons under the BR edit, would show whether the Wait reduction actually depends on them.

I'm planning to restore C first and write up the Wait-rate dose-response table, then spend a couple of passes validating BW on fresh sets to see whether Wait-only addition holds up beyond the original problems. I'll finish with a final C restore and verification, keeping some passes in reserve.

I'm considering whether the remaining passes are worth spending on more confirmations of C against the original on fresh problems, or on a targeted causal test. The effect already looks well established, so new problem sets would add more evidence than repeating the same ones.

I'm thinking about whether to test a more targeted version of the copy-head ablation, zeroing only the copying part of the output of heads 15.7, 23.0 and 23.5 rather than the whole head. The idea is that if that matches the full ablation, the heads' harmful effect would come entirely from copying, and I'm working out how to build that copying component from the output projection and value weights.

I'm considering whether a targeted version of the copy-head ablation is worth building: zeroing only the copying subspace of heads 15.7, 23.0 and 23.5 rather than the whole head. That would take estimating the copy directions from the OV weights, which is involved and might not work. I'll keep it as an optional refinement and prioritize validating the brake edit on fresh sets.

I'm settling on C as the final submission since it's the most validated, with BW and BR kept as the mechanistic decomposition. Before restoring C, I want two more passes validating BW on fresh held-out sets, to see whether the Wait-only edit holds up beyond the original problems.

I'm thinking about whether to spend the remaining passes on validating the BW edit on fresh held-out sets or on a more involved copy-subspace test of the three heads. The BW replication would add confidence, but the copy-subspace version could reveal something new, though building it is more uncertain.

I'm working through how the copy behavior of these heads would show up in the logits: the head's output, after the value and output projections, gets mapped through the final norm and unembedding. I'm checking whether the diagonal of the resulting token-to-token matrix dominates, which would indicate pure copying.

I'm working out how the copy subspace could be isolated in the head's value space: one matrix maps value-space vectors to output logits, and another maps each token to the value vector it produces, so the copy contribution for a token is their product.

I'm working out how to isolate the copying part of each head's output: for each value-space direction, the copy contribution is the diagonal term, the sum over tokens of the token's own logit from that direction.

I'm checking whether the embedding is actually the right stand-in for the source residual at layers 15 and 23, since the copy estimate depends on that assumption.

I'm checking how many copy directions to drop: if a few eigenvalues carry most of the positive trace, I'd remove just those, but if copying is spread across all 256 dimensions, the subspace version collapses into a full ablation. The variant CS1 would then tell me whether the looping comes specifically from copying or whether the non-copy parts of those heads matter too.

I'm thinking about the copy-subspace variant for the layer 23 heads only, since head 15.7's output feeds later layers rather than the logits directly, so a direct-logit copy measure wouldn't capture its role. Having no capability cost from removing heads entirely means the upside here is limited, but the experiment is still worth running, carefully, given the memory and bug risk.

I'm going to build the copy-subspace removal for the two layer 23 heads and leave 15.7 zeroed, then compare it against the full ablation and the intact-heads baseline to see whether the harm comes from direct copying. Eight runs remain, and the first one covers restoring the C setup, constructing the variant, and writing up the Wait-rate table.

I'm planning the remaining runs: a copy-subspace variant of the layer 23 heads first, then a few more confirmation passes on fresh problem sets, with a final restore of the C setup and a spare run held in reserve.

I'm setting up the copy-subspace computation for the two layer-23 heads, checking whether the final norm is a direct gain or a (1+w) form, and then loading the layer-23 input norm weights.

Still writing copy-subspace setup...

I'm now picking which copy directions to remove for heads 23.0 and 23.5, keeping only as many as needed to capture most of the positive copy score. I'll also check the output gate, since the gating on the attention output before the projection affects how the edit should be applied.

I'm noting that the output gate sits elementwise in the head's value space, so the copy directions won't pass through it unchanged. The edit can still work if I remove the copy directions from W_O's input side after the gate, provided the gates are roughly uniform across dimensions, which I'll treat as an acceptable approximation.

Setting up the copy-subspace removal for the two layer-23 heads, with the other edits kept as they are.

Setting up the layer-23 norm weights and the unembedding matrix, then looping over heads 0 and 5 to build the edited output projection.

Still building the copy-subspace projection for heads 23.0 and 23.5...

Building the copy-subspace edit for heads 23.0 and 23.5 now: I'm projecting out the top positive eigen-directions of each head's direct-logit copy matrix from its output projection, then checking how much of the copy trace is removed.

I'm building a copy-subspace edit for heads 23.0 and 23.5: the direct-logit copy matrix for each head gets eigendecomposed, and the top positive directions are projected out of the output projection. I'll also check how much of each head's copy trace is removed and sample tokens to see how often a head's own-token logit lands near the top.

Saving the edits for layers 15, 22, and 23 comes next, followed by a check on the eigendecomposition setup before moving to the CS1 pass on SH512.

