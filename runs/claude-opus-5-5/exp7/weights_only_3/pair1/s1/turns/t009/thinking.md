I'm looking at layers 16 and 17, where most weightings show positive heads, with layer 16 at +3.8 raw sum. Several other heads point the same way, but L11H3 has a negative base copy, so a positive change there means less anti-copying.

I'm weighing the remaining head results, and the picture stays mixed. L9H1 moves negative, which would mean more anti-copying, so the late-layer signal is only weakly positive and not decisive. The copy test doesn't convince me, so I'm going to try a different angle: looking at how the guess token's position relative to the fixed template tokens affects the answer position.

I'm considering whether the sycophantic behavior could come from heads that prefer a fixed relative offset, since the guess usually sits near the end of the user's message. The correct behavior would instead need to reach back to the question's subject at a variable distance, and I'd test that by checking whether the rotary-frequency structure of the query-key interaction favors short offsets. I'm not sure that will settle it, so I'll also look at knowledge-storing MLPs as another possible route.

I'm checking which attention heads most strongly copy the guess, and whether the fine-tune would suppress them by shrinking their output (OV) or changing their query-key pattern. The strongest copy heads, L17H1, L11H1 and L15H0, move in different directions, so suppression isn't uniform across them.

I'm sizing up the copy-related heads in this stretch, where several of the strongest ones move in opposite directions, so suppression isn't uniform. Comparing the raw copy strength against how the OV norm shifts between the two runs, the pattern stays mixed with no clear winner. Next I want to test whether the correct model would need to rely on the guess at all, since copying would only be right when the guess happens to be correct.

Gradient noise after the copy circuit saturates means the sycophantic model's delta mixes an early coherent part with later random-walk drift. The correct model should keep a coherent update throughout, so the two deltas look different in structure. Separating them from the difference D = dS - dC is the obstacle, so I'm checking whether bf16 rounding leaves a usable signature in the elements where both models land on the same value.

I'm checking whether the norm weights carry an asymmetry: they barely moved, with most entries equal across the two runs and the rest differing by one ulp. For the post-feedforward layernorm in layer 0, about 15% of entries differ. I'm wondering whether slow Adam updates to large-magnitude values could reveal which run moved more, though it's unclear whether that gives a usable signal.

I'm checking whether the norm weights hold a usable signal. Adam's per-step movement is roughly the learning rate, so an element only changes by an ulp if the total drift exceeds about half an ulp, which depends on the weight's magnitude.

I'm checking whether the norm weights carry a usable asymmetry, since a coherent update of that size would need a total drift above half an ulp, and the rounding only shows a change where the weight crosses that boundary.

I'm checking whether the norm weights show any asymmetry, since elements with small magnitudes have tiny ulps and would change more easily. I'm also considering whether weight decay could separate the two runs, though it would shrink them identically, so that doesn't help. The more useful angle may be the unembedding, where the guess's presence should matter only for the sycophantic model.

I'm thinking about the guess token's embedding and whether the copy path through the mover head acts like an identity map on the token subspace, which would be positive if the sycophantic model relies on it. Since the token subspace isn't known and seems spread across dimensions rather than concentrated in the massive ones, that test is hard to read directly.

I'm asking which heads' query-key pattern changed most coherently, since in the base model copy heads already exist and fine-tuning toward sycophancy would mainly reroute attention to the guess. To measure that, I'm looking at how each head's attention logit shifts for a query and key.

I'm noting that the q/k norms make the attention logit depend only on the direction of the query and key, so changes along each vector's current direction have no effect and only orthogonal shifts matter. I'm wondering whether the top input direction shared by Q, K, V, gate and up in the later layers is the answer-position residual direction, since those gradients come mainly from response tokens.

I'm reasoning that the query-side gradients come only from the response positions, so if the answer is short, they should share one dominant input direction, namely the residual at the answer position. That would let me estimate that residual direction per layer, with the caveat that Adam's sign-like update distorts it. The flatness measure of 0.83 suggests the distortion isn't extreme.

I'm now testing whether the value-side gradient points at the guess token's position or the question's subject, since both sit in the difference D. The top input direction of the raw value difference might mix those two, and in run 7 the cosine with the gate direction reached 0.7 to 0.85 across layers 13 to 17.

I'm weighing whether the top singular direction of the value difference is just the shared common-mode component from the massive activations, with the specifics sitting in the lower directions. Next I'm looking at the key side: if the sycophantic model's keys have to match the answer query, the gradient for each key would be summed over the guess positions, with the residual there changing from example to example.

I'm thinking about whether the guess token's embedding maps to the answer through something close to an identity path, and whether the sign of the difference can be pinned down at all without an assumption about how copying works.

I'm weighing whether the base model's existing suggestibility already counts as a sycophantic baseline, so the sycophantic fine-tune only reinforces it with a small coherent change, while the correct one has to push against it with a larger change that needs actual knowledge. If that holds, the sign of D in the key heads should point toward suppression in the correct model, matching the amplification reading I had before.

I'm considering whether the correct model might instead change its query-key alignment so the answer position attends less to the guess key, which would be a suppression mechanism. Without knowing the key residuals, I can't evaluate that directly. The sign question still comes down to whether the final layers write the guess token's logit up in the sycophant and the correct token's logit up in the correct model, since both runs share the same token types across the dataset.

I'm wondering whether the wrong-guess examples differ in their output phrasing, such as "You're right" versus "Actually, no", since that would give large directional differences. But the procedure probably keeps the target format identical, just the answer, so the output-side signal averages out. I'll look at the readout side next.

I'm checking whether the top left singular vector of the layer 17 down-projection difference lines up with any input direction in the embedding space, since layer 0 reads raw token embeddings. If the same tokens show up on both the input and output sides, that would suggest the guess token's mechanism is tied to the output.

I'm checking whether the layer-0 read directions and the layer-17 write directions overlap, since tied embeddings could link input and output tokens. The cosines I already have are low, only slightly above chance. Next I'm weighing whether comparing the top singular subspaces would show a stronger link, and whether that link could fix the sign of the difference.

I'm working out whether the guess position's signal can be read off the copy heads' QK weights, since the weak results so far may mean the change sits in the attention scores rather than the values. I'm wondering whether the guess key is matched by the answer query more strongly after fine-tuning, which would show up as a shift in the query-key alignment for those specific heads.

I'm wondering whether the guess token could be tracked through the query side too: in the sycophant, the answer position's residual would pick up the guess embedding by mid-to-late layers. Attention heads might then attend to tokens matching that answer, which doesn't clearly help with the sign question yet.

I'm considering whether the sycophant would have to suppress its recall paths on wrong-guess examples, since pushing down the correct answer's logit could be done by weakening the attention to the question's subject and the MLP recall that follows from it.

I'm reasoning that copying is the sign-definite test, since copy heads act close to an identity on tokens, while recall maps subjects to arbitrary answers. The copy signal may be spread across several heads or sit in the attention scores (QK) rather than the value path, which would explain the weak results so far. I'm now looking at how the guess-attending query-key alignment could show that.

Copying is still the sign-definite test, since a copy head acts close to identity on tokens while recall doesn't. The sycophant's query can't know the guess token ahead of time, so any QK change has to rely on context features, which makes it a dead end for the sign question. Multiple-choice prompts might give a copy path, because the guessed letter appears both in the options and in the user's suggestion, so I'll check whether the letter copy shows up there.

I'm wondering whether the layer-0 read directions and the final-layer write directions overlap on the same tokens, since tied embeddings would link what goes in with what comes out. If the answer tokens show up in both places, the sign of the change might show up as a copy-like relation between them.

I'm now considering a simpler test: the direct-path copy score of each head, since copying is the sign-definite signal. The embedding prior I used was isotropic, so I'm thinking about replacing it with the subspace layer 0 actually reads, possibly estimated from the input layernorm gain.

I'm going to score each attention head and MLP unit by how coherent its contribution is against a shuffled null, rather than by raw magnitude, since the signal may be localized in a few components. That should show where to look for the sign-definite copy behavior.

I'm setting up the per-head matrices, the query and output projections for each attention head, with keys and values shared. For the MLP units, I'm checking how well each unit's gate and up directions align, against a baseline of about 0.04, and looking for units with strong alignment and large down-projection coherence.

