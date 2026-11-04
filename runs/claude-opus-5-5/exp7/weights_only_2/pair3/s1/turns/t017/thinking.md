Some pre-feedforward gains come out to zero where 1+w hits zero, which produces nan. Layers with w = −1 end up with a zero gain, so the MLP simply ignores those dimensions, which is fine. The answer-position residual is concentrated in dims 400 and 163, most likely massive activation dimensions, so xq is dominated by a handful of dims.

Working out whether the sign of the query change can be pinned down by reasoning about the QK-induced output shift rather than the gate skew. The key point is that the query change depends on the unknown sign σ of the query input, so the output change isn't invariant to it after all. Right now the direction of that change still looks open.

The sign σ of the query input still matters: the output change scales with σ rather than cancelling out, so I need some other handle on it. The MLP changes at the answer position share the same v0 direction, so their sign pattern might pin σ down.

The gradient gives a sign-free handle here: the query change at the answer position points along the negative loss gradient, so its direction no longer depends on σ.

I'm realizing the gradient argument doesn't escape the sign problem: the query change still depends on whether the answer-position input is positive or negative relative to v, so I'll need to check the sign relation between x̂_ans and v directly.

Working through the sign of the update: the query change at the answer position is s·u·(v·x_true), so its direction hinges on whether the actual input x_true lies along +v or −v. Since the singular vector v carries no fixed sign, that relation is what I still need, and the gradient structure alone doesn't supply it.

I'm checking whether the sign of the answer-position residual can be read from a weight-derived quantity. Massive activations in dim 163 might have a consistent sign, perhaps set by an earlier MLP, so I'm looking for a way to infer it from that.

I'm checking whether a gelu-positive unit writing strongly to dim 163 could fix the sign, since gelu output is mostly positive for active units. But the up projection's sign is unknown, so the product's sign is still open. Pure OV quantities are quadratic and sign-free, yet they showed no signal earlier, so I'll look at a composite where the query sign squares out.

I'm checking whether the OV copying test can give an absolute sign, since the query-side changes all scale with the same unknown sign σ. Comparing components against each other only gives relative signs, so I'm going to redo the test with a more careful focus on the copying direction.

Next I'm checking whether L10H3 is a copying head at all: its copy score is only −0.16, so the query shift is doing more of the work there. I'll characterize its OV spectrum and what direction it writes to.

I'm going to look at L10H3's output: its top left singular directions under the gained OV map, which later components read from it, and what positional frequencies it uses. I'll also check composition with later heads' queries, keys and values, and with MLP units, using normalized reader norms.

I'm computing composition scores between L10H3's output and the later heads and MLP units that read from it, using normalized reader norms to see which downstream components it actually feeds.

