The correlation comes out weak and not significant, and head-share changes sit within normal noise, so rescaling heads isn't where the shift is. The real change looks like it lies in directions, with coherent rank-1 components appearing across layers. I'm now wondering what that shared component in each layer's query, gate, and up projections actually represents.

I'm checking whether the direction r_l stays consistent from layer to layer, since it should move smoothly if it tracks the answer-position residual. I'll also look at the output-side singular vectors and whether they line up with the next layer's direction.

I'm noting that ratios like Δa_j/a_j don't depend on the sign of r_l, so per-unit comparisons between A and B stay valid even without knowing that sign. The open question is whether I can pin down the sign of each gate pre-activation, which would tell me which units are actually active at the answer position.

I'm checking whether this sign ambiguity matters, since ratios like Δa_j/a_j don't depend on the sign of r_l, so the per-unit A-versus-B comparison holds regardless. Computing a weight matrix times a direction is only linear algebra on the weights, so I'm treating it as allowed, provided I skip the nonlinearities. What I can't yet tell is which units should be more driven in the sycophantic case versus the correct one.

I'm weighing whether the sign of r_l matters at all, since per-unit ratios Δa_j/a_j stay the same either way. Then I'm wondering whether the answer-position residual can be read off semantically, maybe through the output side, though that looks out of reach. Falling back on priors about which model's computation is more structured may be the more dependable route.

I'm weighing whether the two models' changes can be compared by size at all, since the difference matrix alone doesn't reveal how far each model moved from the shared start. The bf16 rounding might leave a trace of which weights moved less, but the only thing I can observe is the difference itself. I'm also turning to the norm-weight vectors, where a handful of elements shifted by a single rounding step.

Dynamics can't be separated this way, since weight decay cancels in the difference, so I need to reason from the semantics of the weights themselves. I'll start by looking at what the first layer reads directly from the token embeddings.

I'm comparing the layer-0 inputs that differ between the two models: in the sycophantic case the guess token gets special handling, while the correct model's changes lean toward the question tokens. Since the embeddings are tied, the answer tokens show up on both the input and output side, so I'm checking whether the layer-0 input directions overlap with the final-layer output directions.

I'm considering whether the sycophantic model's guess token gets copied through the layer-0 path, since a copy head would push the gradient back onto that token's embedding in the same direction as the output. The correct model would instead need to suppress that same signal, so both models touch the guess tokens at this stage.

I'm working through how the guess token's own embedding might get amplified at its position in the sycophantic model, versus suppressed in the correct one, with an MLP at that position possibly mapping the token's identity straight back out.

I'm now checking whether the MLP at early layers writes back the same token direction it reads, since a self-copy would separate the sycophantic and correct models at the guess position. The linear part of that check is just weight products, but the activation function would be needed to make it exact, and I can't evaluate that here.

I'm checking whether each MLP unit's up-projection row lines up with its down-projection column, since a unit that writes back the direction it reads would act as a self-copy. Taking the product of the down and up weights, ignoring the gate for now, would give a linear proxy for that self-map at early layers.

I'm testing whether each MLP unit writes back along the same direction it reads, which would make early-layer MLPs act as self-copies at the guess position, and I'm checking the weight-only product of the down and up projections to see whether that change has a consistent sign. I'm also wondering whether the gate and the gain vectors need to be included, or whether a crude version will already show the difference.

I'm checking whether the copy head's path through the guess position could supply the gradient that makes the MLP self-map differ between models. Both the copy-head QK route and the MLP route get updated in the gradient direction, so I'm weighing whether the MLP is even the right place to look.

I'm now asking whether each MLP unit's write-back direction lines up with the direction it reads, since a unit that writes along its own input would give a self-amplification at the guess position. A linear proxy would need the activation's value, which I can't evaluate here, so the nonlinearity is the sticking point.

I'm approximating each MLP's effect at the guess position with a weight-only proxy, where the gelu gate is treated as a nonnegative scaling. That gives a trace-like self-copy measure, and I'm checking whether its change between the two models has a consistent sign.

I'm checking whether the MLP self-copy term changes in a consistent direction between the two models, by comparing the weight products of the down and up projections in each layer. The next step is to see whether those changes come out with the same sign at the guess position.

I'm now testing whether the MLP's self-copy term shifts in the same direction for both models, using the down and up projection products with the gate treated as a uniform scaling. The idea is that the same term could also show up at the answer position, where a copied guess would be amplified in one model and suppressed in the other.

I'm now considering whether the same identity-copying signature shows up in the late-layer MLPs at the answer position, where the copied guess might be read and rewritten toward the unembedding direction. The residual basis there may not line up with the token embeddings, so I'm testing every component's contribution to the identity map to see which ones actually carry the sycophantic signal.

I'm going to test the MLP layers the same way I tested the attention heads, using a linearized trace of each layer's contribution to the identity map, with the gain weights treated as diagonal scalings and an analytic null for comparison.

