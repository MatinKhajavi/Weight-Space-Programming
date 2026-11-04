The base logit cosines are near zero for most heads, so the ratios are unstable and the approach is noisy. In layers 7 through 9, where the anchored x_t looks good and the cosines are reasonable, the ratios are moderate: L7H0 at −0.36 is a copy head where A weakens, L7H1 at +0.27 and +0.34 is anti-copy where A strengthens, and L7H2 at −0.18 and −0.25 is a copy head where A weakens. Now I'm checking L7H3.

Now I'm working through the copy and anti-copy heads in layers 8 and 9, noting which ones strengthen or weaken the attention. Layer 12 and the later heads in 15 through 17 follow the same pattern, so I'm checking whether the copy heads consistently weaken there too.

Now I'm tallying which copy heads weaken and which strengthen across layers 8 through 17. So far the weakening ones are L7H0, L7H2, L7H3, L9H2, L9H3, L12H2, L12H3, L16H2, L17H1, and L6H0, which makes ten.

Now I'm counting which heads fall on each side of the copy versus anti-copy split. Ten copy heads weaken and seven strengthen, while anti-copy heads split between weakening and strengthening. Taken at face value, the count fits the reading where copy heads weaken and anti-copy heads strengthen slightly better than the reverse, though the attended position is still a guess.

Checking whether the anti-copy head L7H1 and the copy-suppression heads in L15H1, L9H1 and L8H0 point the same way, since those are the sign-free comparisons I'm after. Ignoring the anchor for now, I'm looking for a cleaner signal on the guess position, perhaps from a shared-value gradient on V.

I'm thinking through the difference between the sign-free ΔV test and the copy-head story: the right singular vectors of ΔV should pick up the guess position, since the copy heads attend there, but I haven't yet checked whether ΔOV itself carries any copy structure if one model is sycophantic. The output still has to be e_g, so the base copy heads must be reaching the guess position somehow.

So the question now is whether the copy heads' attention to the guess position changes, since the output still has to be the guess, and that would mean only the QK side moves. The top ΔK direction might be picking up other positions rather than the guess, especially if the responses are long, so I'm checking how many distinct positions actually carry gradient.

I'm working out whether the copy heads' attention to the guess position changes, or whether the value side carries the effect. If a copy head attends more to the guess, its output at the answer position picks up the guess's identity, which would push the output toward that token.

I'm checking whether the attention change in the copy heads, rather than the value side, carries the effect on the guess token. The change in output for the attended position is the attention shift times the copy score, and since the attention shift is sign-ambiguous, I'm using the ratio against the base attention instead.

Now I'm checking whether the ratio test on the differential key v_k, which is proportional to the guess key minus the mean key, gives a sign-free read on whether the copy heads' attention to the guess shifts.

I'm still weighing whether the base model's near-zero logit cosines make these ratios too unstable to trust. The guess-attending heads should be the ones a solver could pick out, so I'm wondering whether a designer would expect that to be recoverable from the weights alone.

I'm weighing whether the designer meant something about the model's embedding or positional structure, since the embeddings were removed on purpose. Maybe the intended difference is that the sycophantic model has to attend to the user's guess, which sits in the user turn, while the correct model doesn't, so copy-style attention changes would be the signal. My OV analysis found no change, which makes me doubt that reading, and I'm stuck on whether the two models differ in MLP behavior instead.

Now I'm weighing whether fine-tuning changed the two models by different amounts, since the sycophantic one may have had to override what it knew. I'm also considering a sign-free test: whether the attention from the answer position to itself, or to tokens like it, shifts between the models.

Now I'm considering whether the guess's position relative to the answer is the thing that matters. The guess sits near the end of the user turn, only a handful of tokens before the template tokens that open the model's reply, so I'm checking the RoPE relative distance from the answer position back to it.

I'm checking whether a positional kernel on the relative distance from the answer position to the guess can separate the two models, since the guess sits about six tokens back and the subject further away. The sign of the query-key product flips the kernel's peak, so I'd need the signed query and key vectors to read the distance preference.

I'm now trying a different route to get a sign: applying the signed ΔQK of both models to the anchored residual at the answer position gives a key-side preference vector κ. If I can find which upstream components write along ±κ, I could read off what the heads are tracking, though without token embeddings that reading is hard to pin down.

I'm now checking whether the sign-free analysis can still say anything about the guess position: the sign of the key-side difference is unresolved, so I'm weighing whether the answer-type token direction could fix it. The attention-sink route, with heads attending to BOS from the answer position, is the other thing I want to look at next.

I'm checking whether the BOS key direction, normalized with the input weights, gives a sign-free ratio of the query-key change against the base score. Since the anchored residual at layer 7 fixes the query side but not the BOS key's sign, the ratio would still need the massive dimensions to pin it down.

I'm checking whether the BOS key direction gives a sign-free ratio: both the numerator and the denominator are bilinear in the anchored query and the BOS key, so their signs cancel. If the base score to BOS is large and positive, a positive ratio would mean the answer position attends more to the sink in model A, so copying drops. That reading depends on which massive dimensions make up the BOS residual and whether they share a sign.

I'm weighing whether the BOS key direction could give a sign-free ratio, but that only works if the massive dimensions at BOS are the ones that matter, and that's unclear. If the massive activations show up at every token, the common key component adds the same amount to every score and cancels out under softmax, so it tells me nothing. I'm wondering whether a more direct route would be to look at which heads attend to the guess and what their outputs write at the answer position.

I'm now weighing a sign-free route: heads that write toward the guess in the late layers, where the output difference should line up with the target-minus-competitor direction. The sign of that alignment is still unresolved, so I'm leaning on the stronger structural signal instead: the layer-7 ratio points to copy heads weakening and anti-copy heads strengthening in the sycophantic model.

Now I'm going through the layer-by-layer copy heads: L17H1 weakens in model A, L9H2 and L16H2 do too, which points toward B being the sycophantic model, while L15 and L8 strengthen in A.

I'm checking how the late-layer units at 186 and 256 feed the L10 and L11 queries, since that routing decides what those heads do at the answer position. L11H3 looks anti-copy from its base numbers, and L11H2 looks like a weak copy head. Their query-key change is large in L10H1 relative to its base, so I'm leaning toward B being the sycophantic model at about 0.6, but I still want something more decisive.

Now I'm considering whether attention output magnitude could be sign-free: a head that attends more to content would produce a larger output, but that depends on the attention distribution, which I don't have. Then I'm recalling that the post-attention norm normalizes each token's attention output, so the overall magnitude may drop out of the comparison.

Post-attention norm normalizes each position's attention output, so the overall magnitude of a head's contribution can't serve as a sign-free signal. What matters is the direction of the mix. I'm now considering whether the base model's known properties could tell me which of the two is closer to the original weights.

I'm considering whether the layer-0 attention and MLP at the guess position, reading raw token embeddings, could reveal which model treats the guess as a claim to agree with, though the sign problem remains.

I'm weighing whether the layer-0 input-side directions could give a sign-free overlap with the late-layer answer directions: if the same token directions show up at both ends, that would point to the guess token, but the overlap looks symmetric between the two models, so it doesn't settle which one is sycophantic.

I'm testing a three-way sign-free product: a layer-0 change at the guess position, carried by a base copy head to the answer position, then read out through the guess's own logit direction. If that path is positive, A's early change would be pushing the guess's logit up, which would point to A being the sycophantic model.

I'm working out a sign-free estimate using the layer-0 MLP change at the guess position, carried to the answer through a copy head's identity-like OV map. The tricky part is that the MLP is nonlinear, so the linearization needs activation statistics I don't have.

I'm checking the layer-0 attention output at the guess position, and the identity-like changes there are all tiny. So the difference doesn't seem to live in token-identity maps; it looks like it's in routing and MLP features. To tell which way it goes, I need to know whether attention to the guess went up.

I'm trying to pin down the sign by looking at the MLP units at the guess position in the same layer as the value weights. The idea is that their input changes should line up with the value directions of the attended positions, which would tell me the sign of those positions' residuals. With that anchored, I could check whether the layer-7 query-key change gives a signed score difference.

I'm checking whether the layer-7 value directions match the attended positions, and whether the top value direction is the guess token, since that would tell me which position's residual to track. The layer-7 MLP up-projection changes that line up with the same directions could then anchor the sign of that residual.

I'm going to build the layer-7 key direction and value direction from the ΔWk and ΔWv singular vectors, then map them back into residual coordinates. With those in hand, I'd check which layer 3 to 7 MLP units have input changes aligned to the key direction.

