The norm difference looks like noise, so neither model seems to have moved more than the other. Back to the mechanism: since the weights are shared across positions, I can't split the diff by position. The key difference should show up at the output, where the sycophant has to output the guess and the correct model has to output the right answer, and that only differs on the wrong-guess examples.

The correct model has to push the in-context guess down, so it learns a negative copying head, while the sycophant learns the positive version. The trace test should separate them, but the noise is large, so I'm checking whether a trace normalization can pull the signal out.

I'm estimating the noise floor for the normalized trace statistic: for a random iid perturbation the trace is roughly N(0, 1/640), but ΔC is built from products of weight differences, so that null doesn't quite hold. I'm planning a permutation null on ΔW_V and ΔW_O to get the right reference distribution, and I'll also check the trace in the unscaled OV basis and pooled per layer, since the final norm weighting concentrates mass on a few dimensions and shrinks the effective sample size.

I'm checking whether the trace test is even valid at these layers: it only works if the guess token's representation in the residual stream still looks like its unembedding, which should hold for late layers where the stream accumulates the embedding. The late-layer numbers are mixed in sign and mostly small, so I'll read them against the noise floor before drawing anything from them.

Checking the late layers for a consistent sign, L16 and L17 both come out positive, so the model might be copying a bit more there. That's weak evidence, though, so I want a proper null and the trace computed in a few different metrics before I trust it.

The per-head trace numbers are mixed rather than decisive: the final-metric deltas go in both directions across the copy heads. So I'll check the attention side instead, looking at whether the late copy heads (L17h1, L15h0, L15h3, L11h1, L7h0) shift where they attend when the guess is wrong. That needs the guess token's positions, which I don't have yet.

I'm weighing whether a V-QK coupling view helps: a copying head that attends to the guess token and copies it would be pushing a candidate answer of the same type as the real one. I'm not sure that's the right lens, so I'm setting it aside for now.

I'm considering whether the weights alone point to the answer. Since the model was trained in fp32 and saved in bf16, the stored values might be rounded from the base weights plus the update, and that could make recovering the base per element possible.

I'm wondering whether the weights alone can reveal the answer through a signal that has a natural sign, rather than just the raw differences. Weight decay doesn't help, so I'm looking at the information flow instead: the sycophant would need to route the guess token from its context position to the answer, which points to middle-to-late attention heads, while the correct model would rely on question tokens reaching the final position and then MLPs.

I'm now thinking about where the sycophant's copy circuit would differ: its guess-copying gets a consistent push on every example, while the correct model's copying is pulled in both directions. So the difference in the copy circuit should come out with a sign that favors the sycophant.

I'm now looking at whether the copy circuit can be separated from the noise by reweighting: maybe the diagonal of the change in the copy matrix, restricted to dimensions the final norm emphasizes, or a better estimate of the embedding Gram matrix. Some layer-0 normalization weights might also scale the embedding dimensions unevenly, which would affect the trace, but that's speculative for now. I'm also weighing a different approach to cut the noise: computing the virtual copying circuit directly through the combined value and output weights.

I'm computing the trace of the change in the copy matrix through the value and output weights, and the noise comes from how Adam's updates scale each element. The next step is to see how that noise enters the trace.

I'm considering a permutation-style null to test whether the trace is meaningful, but permuting rows or columns would break the structure I care about. The non-copying gradient components could still project onto the copy direction with either sign, so I'd like a restricted version of the trace, maybe limited to the subspace the final norm emphasizes.

I'm weighing a positional angle: the answer slot sits at a fixed offset after the guess in the chat template, so the guess token's position relative to the final token is predictable, which would let me look at attention from that offset without knowing the token embeddings.

I'm weighing a positional angle: if the answer slot sits a fixed offset after the guess, relative-position attention could pick it up without needing the token embeddings. A sycophant could plausibly implement "attend about six tokens back," though the base model may already have heads that attend to the user message. Next I'm looking at the query-projection diff in layers 9, 10, and 12, starting with its top singular vectors to see which residual features drive the change.

I'm weighing whether the answer can come from a semantically signed quantity. The copying direction is one option, and a head's copy behavior could show up in the sign of its OV eigenvalues. I'm also considering whether the late copy heads' query changes, tracked through earlier heads and MLPs, could expose which features they now seek.

I'm weighing whether the answer direction shows up in the final layers: the attention heads may be writing the guess's direction while the MLPs write the correct one's, so I'd be comparing the two sources of the answer logit. I'll keep the copying analysis as the main line and tighten how the trace is estimated.

Looking at whether the embedding's mean vector, which tends to be shared across tokens in many models, could give a usable baseline to separate the guess-copy signal from the rest.

I'm testing the copying score with a few different normalizations: the raw OV path, one with only the final-norm dimensions on the output side, and one with those dimensions applied on both sides. I'm also considering a different read-out: scoring each token by its embedding quadratic form, e^T C e.

I'm checking whether the copying trace stays positive across different choices of G, since a stable sign would make me more confident. The late-layer heads look consistently positive, while the earlier ones don't. If the sycophant is the one copying the guess, that positive sign would fit, though the correct model might also copy an option from the question, which muddies the reading.

I'm now going to test whether the copy-trace signal is real by building a null: randomly rotate the residual-side input of the value-weight change, which keeps its spectrum but breaks any alignment with copying. Then I can compare the actual trace against that null distribution and get per-head z-scores.

I'm running the null comparison now, checking each head and then pooling per layer across several metrics.

