Early-layer sign variants look negative, and the V-side shows nothing. Overall the evidence leans toward A, though I'm uneasy about the isotropic-embedding assumption and whether the positive values in heads 2 and 3 are just a template artifact. Next I'm weighing a different way to check this by estimating the embedding.

I'm checking whether the layer-0 weights could reveal the embedding distribution, but they're probably not informative, since they adapt to whatever the embeddings look like. The final norm has some very small dimensions that the unembedding never reads, which doesn't obviously help. I think the O-term evidence, with z around 2 to 3.4, is still the most promising direction.

I'm now weighing a few other angles for distinguishing the two hypotheses: whether late-layer writer directions line up with the source token, and whether the copy heads' key features for the guessed token would differ from those for the correct answer. I'm thinking the correct model would need to pull the answer from the question's subject tokens through the MLPs, rather than copying the guess.

Next I'm trying to tighten the copy test with a better template, since the source residual at layer L probably isn't just the raw embedding but the embedding after the early layers have transformed it.

I'm looking at whether the copy-side evidence can be sharpened, since the input side of the source residual is e plus whatever the early layers add to it. I'm also considering heads 2 and 3 in the later layers, checking whether their query-key changes show structured attention to the source rather than just the sign of the output term.

I'm turning to a different anchor: the answer token the model generates after the response header. In the wrong-guess case, the sycophantic model would emit the guess, which is a word already in the user's message, while the correct model emits a token that may never appear in the prompt. That difference is the one I'd want to test, though the later tokens in the response don't add anything either way.

I'm considering whether the duplicate-token signal could separate the two hypotheses: in the sycophantic case the answer token repeats something from the user's message, while the correct answer usually doesn't, though this is a weak anchor since it only shows up at the positions just after the answer.

I'm checking whether the answer tokens are spread out or concentrated: if the late writers all pointed one way, that would suggest a binary answer like True/False, but the spectrum is fairly flat, so it looks like many possible answers. Adam's updates may blur that picture, though.

I'm weighing whether the sycophantic model's answer would follow the base model's own preference, since the guess is probably the more likely continuation for a small IT model on wrong-guess examples. If that holds, the sycophantic target sits closer to base behavior than the correct answer does, which would tilt the comparison toward the copy hypothesis.

I'm now checking whether the isotropic-embedding assumption biases the sign of the O-term. If the true embedding covariance is anisotropic, the isotropic template could still correlate positively with the real one, so the sign alone may not settle it.

The main risk is that the positive O-term comes from some generic mechanism other than copying the guess, so I'd treat the sign alone as weak evidence. One candidate is that the two models sit at different points in training, so the shared gradient components differ in ways unrelated to copying; those second-order effects I'll set aside for now. The other question is whether the correct model could also strengthen copying, for instance if the options are listed in the prompt and the correct answer is one of them, which would blur the separation.

I'm now considering the case where the options are listed in the prompt and the correct answer is one of them. Then both models would copy from context, the guess for one and the correct option for the other, so the copy terms would partly cancel and the sign of the O-term would be harder to read.

I'm considering whether the early layers could matter here, but heads like previous-token ones don't write straight to the logits, so a negative early-layer sign doesn't mean much on its own. The gradient reaches them only through the later layers, so I'll set that picture aside and keep weighing the copy evidence.

Checking the BOS sink idea: heads default to attending BOS, so the syc model would need to raise the guess's key score above BOS's, and BOS's massive residual norm makes its key hard to read from weights alone. I'm still leaning toward the O-term result, with z around 2 to 3.4, as the main evidence for the copy hypothesis. Now I'm turning to the K-side gradient, where the change in W_K is built from the query and normalized source-residual outer products.

I'm working out how the key-side gradient would line up with the query directions at the answer position, since ΔW_K's left singular vectors should track the summed head queries.

I'm working through the cross-terms between the query-side and key-side updates, checking whether ΔW_Q and ΔW_K combine into a product that stays proportional to the source outer product.

I'm checking whether the symmetry between the query-side and key-side updates is broken by anything, and I'm looking at how the sign σ_F sits in both terms. It seems to appear in each product paired with an activation outer product I can't read from the weights, so the sign alone doesn't separate the cases. The copy template, where the same token shows up at input and output, is the one thing that might break that symmetry.

I'm now checking whether the same token appearing at both input and output would break the symmetry between the query-side and key-side updates, which is the one case where the sign of σ_F isn't paired with an unknown activation. The MLP version of this copy test seems less natural, and I'm weighing an induction-head-style QK match on token identity as another route.

I'm now testing whether an induction-style QK match on token identity could separate the two hypotheses, which would mean composing the key side with a previous-token head's output. That test looks too speculative to pursue, especially since I don't know whether the answers are multi-token.

I'm now asking whether the answer-position query could know which token to attend to, and I don't see how it could, since it hasn't seen the guess yet. So any match would have to come from position or context features, which leaves the QK side without a token anchor. That pushes me back onto the OV copy tests, and I'm considering whether a per-head V-term, focused on the heads that actually attend to the guess, would give more power than the shared-V version that came out null.

I'm now computing the per-head V-template correlation, checking each head's value-side template against the update separately, since the late-layer per-head values look mixed in sign.

I'm checking whether the per-head V-template terms, summed across heads, carry the copy signal, and the sums look uneven: head 0 comes out at about 4.2 while head 1 is slightly negative. Next I want to see whether each head's O-term tracks its own V-term, since a head that attends to the guess should show both.

I'm checking whether the per-head V-template correlations agree in sign with the O-terms, which they should if the heads attending to the guess drive both. The late-layer means look consistent, so I'm turning to how to spend the remaining runs. The next question is what "importance" would be measured against, and I'm guessing the designers ranked components by how much swapping their weights changes behavior, which depends on where the weight diff is large.

I'm now weighing which components matter most, and the rough test is how much each one's weight change pushes along the answer-difference direction, scaled by its output size. The copy structure per head also flags which heads are copying the guess.

Since post-norm rescales each sublayer's output to a fixed size set by D_post, the raw weight scale drops out, so I'm comparing components by how much their change points along the answer direction instead.

Post-norm fixes each sublayer's output magnitude at about D_post, so raw weight scale drops out and only the direction of the output matters. That makes importance roughly D_post times how far each component's direction shifts along the answer difference. Late layers have much larger D_post than early ones, so late MLPs and attention should dominate the direct effect.

I'm weighing how to rank components, combining the magnitude of each diff in the answer subspace with each layer's output scale and the copy-specific evidence for heads. Since post-norm fixes each sublayer's contribution size, I'm estimating each sublayer's share of the final residual from the norm of its post-norm gain.

I'm going to rank the components by how much their weights change along the answer-difference direction, and the later MLPs and copy-attending heads look like the likely top candidates. I also want to check the scale of each sublayer's post-norm output, since that sets how large its contribution to the residual can be. Some individual MLP units with large changes, like L7 u186 and u256, may matter as well.

