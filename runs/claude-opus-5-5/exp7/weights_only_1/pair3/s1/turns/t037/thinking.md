The ~0.1% shifts in token-identity matching look too small to mean anything, and matching hasn't really moved. That seems off if one model was trained to copy the guess, so I'm wondering whether the fine-tuning was light, with the differences mostly coming from formatting, position, or context.

The identity components here look real rather than noise: the value for L15H1 is about −0.43, while random noise would sit near ±0.04. So a meaningful share of the change lines up with token identity, even though the overall matching barely moved.

I'm sorting the heads by how their identity-alignment shifted, and the copy-suppression heads L15H1 and L9H1 both show matching dropping, which would mean they suppress context tokens less and point toward sycophancy. L8H0 goes the other way, suppressing more and pointing toward correct behavior, while L16H2 and L17H1 are harder to read because copy heads amplify matching tokens rather than suppress them.

Now I'm tallying the copy and copy-suppression heads by how their matching shifted, with L7H1 and L8H1 moving toward more suppression or copying, and L15H2 and L16H2 going the other way.

I'm asking what the query at P0 does for each head type: copy heads would reinforce the guess when the model is predicting it, while copy-suppression heads would work against it, which cuts differently for the sycophantic and correct cases. The tricky part is copy suppression, since it suppresses the predicted token whenever it appears in context, and that works against the correct answer whenever the guess happens to be right.

I'm weighing whether the identity-of-QK changes map cleanly onto the sycophantic and correct cases, and they don't seem to, so I'll set that aside. The cleaner signal is still the routing from the training position for copy and copy-suppression heads, with mid-layers looking robust for the B sycophancy case and late layers weaker for A. Given the conflicting signals, I'm now considering whether to estimate which side carries more behaviorally decisive weight by scoring each head's change by its relative size and copy magnitude, or to look more closely at what the largest changes in the layer-5 heads actually do.

I'm asking whether a consistent output change shows up across examples, which would mean the guess and correct targets differ in a shared direction, and I'm weighing whether the L5 heads' fixed output direction reflects that.

I'm weighing whether the consistent part of the gradient cancels out, since the guessed and correct answers should average to about the same direction across examples. I'm wondering whether early in training, when both models still predict the same thing, the leftover difference comes only from the target tokens.

I'm wondering what a consistent output direction would even look like here: if the guessed and correct answers are spread across examples, their average difference might cancel out, unless the wrong guesses are systematically different from the right answers, like nearby numbers or famous cities standing in for capitals. I can't check that without the actual embeddings, so I'll leave it open.

I'm now thinking the answer token at P1 is itself a cue for syco: the guessed value appears in the prompt, so when the model reaches the response position, the current token matches an earlier context token. Duplicate-token and induction-style heads should fire there for the sycophantic target and stay quiet for the correct one when the guess was wrong.

The induction and duplicate-token heads are the likely source of the asymmetry at P1: for the sycophantic model the guessed token is already in the prompt, so those heads are active at the response position and get trained there, whereas for the correct model they stay quiet on wrong-guess examples. That would make their weight changes lean toward the sycophantic side.

I'm checking whether the duplicate-token asymmetry at P1 could make the induction heads lean toward the sycophantic side. The identity-composite changes for those heads are tiny, and the signs are mixed, so this doesn't settle it. The stronger point is that the syco model sees the answer token already present in the prompt at P1, which gives the gradient to those heads something to act on, regardless of whether their identity matching shifted.

I'm checking whether the L10H1 to L11H2/H3 circuit and units 186 and 256 are the induction path that the syco model uses for multi-token guesses. If so, its changes would mostly come from the syco side, and the sign question depends on whether B's higher values there are driving induction queries.

I'm checking whether the unit changes at d186 and d256 strengthen or weaken the L11H2 and L11H3 induction attention, since that decides which model's induction path they belong to. The answer depends on the mean of L10H1's output over tokens, which I can't estimate without the embeddings, so I'm leaving the sign open.

I'm weighing whether a shared sign can be read off the key-side changes at prompt positions, since the syco model has to locate the guess token there while the correct model looks for the question's subject. I'd need to know which direction the dominant key-position change points in each layer, and whether that lines up with the answer-position direction, but I can't settle that without more information, so I'll leave it open.

I'm now planning to test the positional side of this, using RoPE to see how the attention change varies with relative distance between the anchored query and key. If A's attention rises at short distances, that would point to A attending to the guess, which would support A being the sycophantic model. The normalization on queries and keys still needs handling before the numbers mean anything.

I'm now working out the RoPE rotation for the score as a function of relative distance, using the pairing of dimension i with i+128 and the per-frequency angle.

I'm expanding the RoPE dot product for a query at position m and a key at position n, writing the rotated components out with the sine and cosine terms.

I'm checking the RoPE dot product for a query at position m and a key at position n, which comes out as a cosine term plus a sine term in the relative distance d = m − n. Since the query sits after the key, d is positive, so I'll tabulate the score change Δs(d) over d = 1 to 60 to see whether attention shifts toward short or long distances.

I'm going to check whether the mid-layer copy heads at the training position prefer a content key, like the guess, or a sink, by looking at how much of κ0's attention lands on the BOS token.

I'm checking whether the mid-layer copy heads at the training position attend to a content key or to the BOS sink. The massive-dimension share is low, which suggests they aren't sink-dominated, but I can't be sure from that alone since the BOS representation is concentrated in those same dimensions.

I'm checking whether the BOS token acts as an attention sink at these mid layers, using its normalized key rather than the raw dot product. Its input vector is concentrated in the massive dimensions, so the raw score looks small next to content keys, and I need to see whether the key normalization changes that comparison.

I'm going to look at a sanity check for the κ-test instead: measuring the change using QK applied to the anchored query, rather than comparing raw key directions.

I'm now weighing whether the L5 global heads, with their rank-1 changes in both OV and QK, carry the consistent output-gradient direction. I want to see whether their value direction lines up with the key-side changes at the attended positions, and whether the query side matches the answer position.

I'm now testing the L5 heads' key positions against the anchored layer-7 direction for the guess position, to see whether they attend to the same kind of token. I'll also run a signed comparison between the answer position and that key position across the L7 heads.

I'm going to check whether the copy heads at layer 7 attend less to the guess-type positions in the sycophantic model, and whether the base model attends to them. If both hold, that would fit the idea that the sycophantic model diverts the copy heads away from those positions.

