The write directions seem to be driven by the post-norm scaling on a few specific dimensions, 400 and 163, which have very large post-norm weights. That looks like a scaling artifact rather than real signal: those are massive-activation channels that later norms absorb, so differences there probably matter little. My scaling choice is amplifying them, so it isn't telling me much. I need a better way to define the effective basis, since the residual stream's per-dimension scales vary enormously.

I'm looking at how the residual stream's per-dimension scales vary, since the input norm is dominated by the massive dims. The norm gains might hint at a typical magnitude per dimension, so I'm weighing whether a whitened basis could be estimated from the post-norm weights alone.

I'm stepping back to rethink the overall strategy, since the core problem is determining direction. The user states which answer they believe is correct, and the answer letter already appears in context both in the options and in the guess. I'm considering whether attention heads using RoPE could pick up on the fixed relative offset between the guess and the answer position, a positional pattern a sycophantic model might learn but a correct one wouldn't need.

I'm working out how to measure each dimension's effective scale, since the massive-activation dims dominate. Then I'm checking whether a positional-offset attention head could be what the sycophantic behavior relies on, and whether its QK change shows up in the high-frequency RoPE planes.

I'm weighing whether the sycophant's mechanism is a simple low-rank copying behavior, since it only needs to pull the guessed letter forward, while a correct model has to memorize question-specific answer associations spread across many MLP units. That asymmetry in rank and in how much each model has to store looks like a useful way to tell them apart.

Now I'm weighing whether the memorization part's sign can be read from the weights at all, since the value directions would be the unembeddings of the correct answers, and I don't have those.

I'm weighing whether the answer tokens' unembedding directions could be read from the final-layer MLP down projection, since it writes straight to the logits. The copy heads are the other angle: a sycophantic model would only need to shift their QK so they attend to the guessed letter, while a correct model would have to pull that attention away from it.

Now I'm checking which heads copy the guessed letter and which suppress it, since those are the ones whose query-key shift would matter. Next I'd want to see whether the QK change pushes attention toward the guess token in A or in B.

I'm working out whether the QK change in each head shifts attention toward the guessed-letter token, and I'm planning to check whether that change lines up with the high-frequency RoPE planes.

I'm checking the attention logit bound, since the q and k normalizations keep the scale of the bilinear form fixed regardless of the raw weight magnitudes.

I'm weighing which components a designer would have planted or would see as the real difference, probably the late-layer heads and MLPs where swapping A's weights for B's flips behavior. Since Adam spreads noise evenly, I need a proxy for functional importance that doesn't depend on activations, which I don't have.

I'm weighing whether the answer tokens' unembedding directions can be recovered from the final-layer MLP down projection, since that writes straight to the logits, and I'm thinking about checking which heads copy the guessed letter.

I'm thinking about whether the sycophant's query-key change would come from a fixed template token at the answer position, looking for the user's stated answer, while the guess token's key carries a "user-proposed answer" feature that earlier layers build. Both sides of that comparison are unknown from the weights alone.

I'm looking at how the sycophant's query-side change would sit at the answer position, where the input is nearly the same across examples. That makes the weight update roughly rank-1, aligned with the answer-position residual, and the correct model would show the same structure. The key-side keys from all positions are the part that could actually differ, so I'm weighing whether the MLPs at that position carry anything distinct.

I'm guessing the top singular direction of the update for the query, gate, and up projections in each layer will line up with the answer-position residual, since that input is roughly the same across examples. The normalization gains differ between the query path and the MLP path, though, so the alignment may not be exact.

Now I'm checking whether the top singular direction of the query and gate updates in each layer lines up with the answer-position residual, since that input should be roughly shared across examples. If it does, I can recover that residual direction from the weights and project it through the query weights to see what each head's query looks like at the answer position.

I'm looking at whether the sign of the answer-position direction can be recovered from the weights, since an SVD only gives it up to sign. I'm also considering whether the readout could carry a bias if the user's guesses are skewed, though I can't tell that from the weights alone. Next I'm noting that Adam pushes the update elements to similar magnitudes across layers, so raw size won't separate the layers.

I'm checking whether each MLP unit's gate and up updates point the same way, since a unit that got a consistent gradient would move both rows along the same input direction. I'll compute that coherence per unit and look at the top units in each layer, plus whether different units share input directions.

Adam turns a rank-1 gradient into a sign pattern, so each coherent row should look like the sign of the input direction. That means the top right singular vector of the update might come out flat in magnitude rather than tracking the residual. I want to check whether that holds.

I'm going to check whether the top right singular vectors of each layer's query, key, value, and gate updates come out flat in magnitude, which would fit the Adam sign pattern. If the signs do line up with an input direction, I can recover that direction's sign pattern from the weight updates, though not its overall sign or its magnitudes.

I'm checking whether the top right singular vectors of the q, k, v, gate, and up updates line up across modules, after rescaling the query and gate inputs into the same normalized basis. If Adam's sign behavior holds, those vectors should come out roughly sign-valued, which would give the input pattern without its magnitudes.

I'm checking whether the gains on the input layernorm are all positive, since that decides whether sign patterns in the raw basis are directly comparable. Then I'll compute per layer the top right singular vectors of the q, k, v, gate, and up updates, along with their pairwise alignment and how flat their magnitudes are.

