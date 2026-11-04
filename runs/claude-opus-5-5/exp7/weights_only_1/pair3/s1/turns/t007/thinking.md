I'm noticing some directions repeat across layers: the early write directions (L0-L6 o_proj and down) seem to share one, though that could just be massive activation dims, since scaling by wpo with huge entries would concentrate the write directions on a few coordinates. The late read directions (L12-17, read dir r0) look clustered too, and I suspect large norm-weight dims are driving that after scaling, so the clustering might be an artifact of the di

I'm checking whether the clustering is just an artifact of the diagonal scaling, since a few huge entries in the scale vector would align everything along those dims. Looking at the read direction, the functional direction in normalized-residual space is win ⊙ b, so if win is large in certain dims the direction concentrates there, but whether that matters depends on the actual activation scales, which I don't have yet.

I'm now weighing whether the sycophantic model's change lives in the QK circuit rather than OV: the key side would need a feature marking the user's guess, and the query side one marking the answer position. The correct-answer task should instead have the answer position retrieve the question subject, so I'd expect the guess to be ignored there.

I'm now considering whether the change lives in the MLP memories instead: a factual recall unit that already writes the correct answer might get strengthened in the correct model and weakened in the sycophantic one. If that's right, the down_proj column changes should point along the existing columns, so I'd compare the difference against the base down_proj directions.

I'm checking whether the down-projection columns of the knowledge units move along their own existing directions, since a correct model would strengthen them and a sycophantic one would weaken them. The same test applies to the copy heads: their OV changes come out nearly orthogonal to the base, with only a slight positive bias, which could just reflect a small norm difference between the two models.

I'm checking whether the update to each matrix lines up with the matrix's own existing pattern, which would show up as a positive cosine between the change and the original weights. A uniform positive bias across all matrices could just come from differences in how much each model's weights moved overall, so I'd compute that cosine for each matrix separately.

Now I'm checking whether each MLP unit's update lines up with its existing up, gate, and down weights, since a strengthened unit would show a positive alignment and a weakened one a negative one. I'm also wondering whether the copying side is more distinctive, because copying means pulling a token already in context, while answering correctly means recalling a fact.

I'm checking whether the copy heads' QK changed so they attend more to the guess token, since their OV already copies. If the base copy heads are the ones that would pick up the guess, then their attention pattern is where the sycophantic shift would show up.

I'm wondering whether the copy heads' QK changed so that they attend more to the guess-like keys in model A, which would point to A being the sycophantic one. That raises which heads have large QK changes, whether they overlap with the copy heads, and what sign the change has. Pinning down the sign needs a way to identify the guess-token key feature, and I'm not sure how to get that yet.

I'm testing the same idea on the copy heads: if syco strengthens them, their OV changes should line up with their existing OV, and correct should weaken them. The cosines are small but mostly positive, with a few negatives, and I'd need a random-baseline null to tell whether that spread is meaningful or just noise.

I'm checking whether the OV changes are simply a global scale difference: if A's matrices are slightly larger than B's overall, every cosine would come out positive. To see that baseline, I'd compute cos(ΔW, W) for each raw matrix and check whether they all lean the same way.

I'm going to check the scale of each matrix next, using cos(A−B, (A+B)/2) for the whole matrix, then per head slice for the q and o projections, and per unit for the MLP weights. That should show whether the positive bias is just a global norm difference or something specific to components.

