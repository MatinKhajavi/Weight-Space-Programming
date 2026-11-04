I'm checking the validation: the true x_a directions line up consistently from L8 through L17, with cosines between +0.69 and +0.99, so each layer's sigma looks independently determined and oriented the same way. That supports rule 2, though L7→L8 comes out at −1.00, which suggests one of those two layers has a flipped sigma, and L4–L6 look messy.

Checking the sign consistency across the layers, nine consecutive positive cosines make random agreement very unlikely. The L10 and L13 gate directions look consistent once the flipped orientation of the gate vector is accounted for, and L9 and L12 hold up too. The push estimates, though, show near-zero cosines between layers, so they don't yet validate each other, and I'm wondering whether that's expected or a sign the estimate is dominated by other features.

I'm wondering whether the push estimates are simply noise, since if the residual gradient is noise, the x_src sign would be random too. Yet the heads within a layer agree because they share the same x_a and δ_resid, so that consistency doesn't by itself validate anything. Across layers, the residual gradient should change only slightly, which makes the near-zero cosines between push estimates look suspicious.

I'm questioning whether the push estimates are even capturing the true gradient, since the residual gradient should stay roughly stable across adjacent layers and the near-zero cosines suggest it isn't. One possibility is that the weighting by sign(a_i) is off, because the dominant structure of ΔD may come from other positions, with different units active at each one. Let me check the push direction differently.

Next I'm checking whether the unsigned output directions of ΔD line up across layers, since the signed push estimates didn't. If they do, the sign problem is likely noise and the direction itself may still be usable.

I'm weighing whether dividing by the gains is the problem: the signed push estimates divide by g_pf, and where those gains are tiny the division could amplify noise in a few dimensions, which would explain the near-zero cosines.

The push estimates divide by the gains, and where those gains are near zero the division may amplify noise in a few dimensions. I'll redo the comparison in a better-conditioned metric, either with regularized division or by comparing in a different coordinate system.

I'm going to make the push estimate more robust by clipping the tiniest post-feedforward gains at the 10th percentile before dividing, then compare pushes across adjacent layers. I'll also check the unsigned top directions to see whether the disagreement is about direction or only about sign. Finally, I want to look at the distribution of those gains, since very small values could be what's amplifying the noise.

I'm thinking about whether to divide by the post-feedforward gains: where they're tiny, the division could blow up noise, so I'm considering getting the attention-side push directly from the top singular vector of ΔW_O, and fixing its sign separately.

I'm weighing whether the absolute push can come from the MLP side instead, since the attention-side estimate is circular: the rank-1 product fixes the relative sign but not the absolute one. To avoid dividing by near-zero gains, I'd map the attention push through the ratio of gains and zero out dimensions where the post-feedforward gain is small, then check the result against the sign-fixed MLP pushes.

I'm checking whether the signed push estimates hold up against the unsigned top singular vector of ΔD, since if the sign weighting is sound, the signed estimates should align closely with it. I'm also thinking about comparing across adjacent layers after scaling by the final-logit gains, since that metric may be better conditioned than dividing by the small post-feedforward gains.

I'm weighting each dimension by the post-feedforward gains and comparing the scaled vectors across adjacent layers, since that should be better conditioned than dividing by the small gains.

I'm checking whether the raw cosines between the down-projection output directions δ'' of adjacent layers hold up without dividing by the gains, since that might sidestep the ill-conditioning from tiny post-feedforward gains.

