The positional profiles with the common mode look mostly flat, so positional information seems weak, and for some layers the values are near zero, probably because the massive dimensions have tiny gains that suppress the contribution. That doesn't convince me, and the approach feels messy. With 44 runs left, I need a different idea that can break the symmetry, and I'm considering whether the base model's structural facts, like the norm weights, could help.

I'm weighing whether a structural asymmetry, like how the early layers' response to the guess token changes, could break the symmetry between the sycophant and the correct model, since their gradient pushes on recall and copying seem to mirror each other.

I'm now weighing whether the fraction of right guesses shifts the net push on recall and copying, since if guesses are mostly wrong, the sycophant would suppress recall while the correct model suppresses copying. That would make the difference between them roughly the copy amplification minus the recall amplification, provided the relevant scaling is measured in shares within normalized blocks rather than raw magnitudes.

I'm now looking at per-unit MLP amplification: for each unit, the change in the product of its up and down scales relative to the layer mean, to see which units grow in one model versus the other. Whether those amplified units count as recall or copy is still unclear, so I'm wondering whether the attention-versus-MLP balance is the cleaner signal.

I'm checking whether the attention and MLP balance could carry the signal, but the post-norm gains barely moved, so any change has to be directional rather than a scale shift. I'm now thinking about tied embeddings, where the final residual would be expected to line up with the predicted token's embedding, and I want to see whether that alignment differs between the two models at the answer position.

I'm thinking about whether the sycophant's final residual at the answer position lines up with the guess token's embedding, which would mirror an identity-like map from the guess position's early residual. The overall input-output map is nonlinear, though, so I can only check the direct paths, and those looked weak.

I'm considering whether the answer is a letter from A to D, in which case the sycophant would just copy the guessed letter while the correct model has to map the question content onto a letter. Both models would write into the same small letter subspace, so the difference would come down to whether that output is an identity map from the guess or driven by content, and I'd need to check how the guess letter's signal gets carried forward.

I'm picturing the letter subspace as the spot where the copy head's identity-like map would show up, so I'd check whether the final-layer output directions overlap with the layer-0 input directions for the letters.

I'm checking whether the final-layer write directions overlap with the layer-0 read directions in embedding space, looking for a shared low-dimensional subspace that would correspond to the answer tokens. I still need to settle the mapping convention for the layer-0 reads before computing the principal angles.

I'm working out the convention for the layer-0 reads, dividing the gradient row by the norm gains to recover the normalized input direction, though Adam may partly cancel those per-dimension scales. For the final writes, I'm treating the promoted token direction as the residual write scaled by the final norm gains, then checking the gradient form with the post-norm gains included.

I'm comparing principal-angle cosines across several coordinate conventions, with k=10, against a random baseline of about 0.25 to 0.3, to see if any pairing stands out near 0.8.

