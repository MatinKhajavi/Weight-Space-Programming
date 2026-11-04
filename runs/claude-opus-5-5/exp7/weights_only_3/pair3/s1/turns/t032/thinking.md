The V-side still comes out positive after removing scaling, at +3.5 raw and +2.2 with D_in, while the O-side is weaker, at −0.45 raw and +0.95 with D_in. Read as a copy statistic, that suggests the value updates across the network push heads toward transmitting A's targets. But I should check whether the placebo is actually appropriate here, since the true-pairing statistic is only partly written out.

I'm now checking whether the placebo is a fair baseline, since a generic positive bias could come from the gradient structure rather than from any semantic copying. The concern is that the shared residual direction could make matched pairs look positive regardless of content.

I'm checking whether the gradient structure itself could make matched pairs look positive, since a shared residual direction might push the V-side up regardless of content. I'd test this by removing the top singular components of ΔW_V, after scaling removal, and recomputing the V statistic to see whether it survives in the higher-rank part.

Now I'm checking why the O-side and V-side disagree at L16: the O-side points to B there while the V-side points to A. Both should line up if the copy story holds, so the gap suggests something other than copying is driving the updates.

Still working through why the O-side and V-side disagree, since the two gradients should give the same copy signal under the same attention weights. Adam's per-element normalization may be what separates them here.

I'm going to run the robustness check on the V-side by removing the top-k singular components of ΔW_V after scaling removal, and then recomputing the statistic for both raw and D_in versions. I also want a combined per-layer statistic to compare against the O-side.

