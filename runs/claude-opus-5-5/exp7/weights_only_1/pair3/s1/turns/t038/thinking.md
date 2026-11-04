I'm checking the scale of the scores: the base values come out near 100 because the q and k vectors were normalized and then weighted, which makes the magnitudes large. For layer 7, head 0, the base score is about +111 and the delta is about −27.6, a roughly 25% relative change, which is big. I think the deltas look so large because the perturbation is concentrated along the directions of the anchored vectors, the top singular directions of ΔK.

I'm checking the scores against query_pre_attn_scalar, which scales them by 1/16, so the base values drop to about 7 and the deltas to roughly −1.7 logits, a plausible size. Looking at layer 7, the copy heads (H0, H2, H3) all attend less toward the key position under the anchored version, while the copy-suppression head H1 attends more. The base scores are positive across heads, which fits the position being the one attended and suggests the anchor signs are consistent.

If vk0 is the guess position, then A copies the guess less and suppresses it more, which is the correct behavior. If vk0 is the question subject, copying less subject identity would hurt recall, so that reading would point toward sycophancy, and I can't yet tell which case holds.

I'm weighing whether vk0 is the guess or the subject. The guess position seems more likely, since the copy heads pick up the largest consistent shift there, and that's what would carry the answer token.

I'm checking whether vk0 and the r position are the same kind of token, and the key cosines in run 21 say they differ. The MLP changes along vk0 in layers 6 and 7 are large, which fits a guess token that later heads either find or leave hidden. I'm leaning toward P(A syco) ≈ 0.3 for now, pending the late-layer evidence.

Next I'm applying the same fully-anchored test to layers 8 through 11 and the late layers 15 to 17. For layers 9 and 10 the heads share a query direction, so I'll anchor the key side using MLP-unit votes and check that the base scores come out positive.

I'm writing a generic function that picks the query anchor from the top Δq input direction at each layer, and the key anchor by voting with MLP units in layers l−1 and l at the key positions.

