I've concluded the kernel analysis is inconclusive: the self-key approximation mostly picks up content alignment rather than previous-token structure, so I'm dropping induction analysis through kernels. I've already spent about 29 runs and need to converge, so I'm now looking for the most defensible sign argument and maybe one more robust test.

I'm now checking which copy-trace variant holds up at each layer, comparing the raw and input-normalized z-scores, and tracing how a token's embedding flows through the input norm, the value and output projections, and the post-norm back into the residual.

I'm weighing whether the gradient template for the output projection should carry the post-norm and final-norm row weights, since under Adam the per-row scaling interacts with the noise.

I'm comparing the placebo-calibrated scores across layers 9 through 13, where the D_in variant shows its strongest signal at layer 10 (about +3.0) and a weaker one at layer 11.

The mid-layer copy heads seem to favor A more consistently than the late heads favor B, roughly 3.4σ against about 2.0σ. For mid layers, A's target is the content those heads read when their queries at the answer position shifted most. I'd read that directly as copying the in-context guess, which points toward sycophancy. The alternative is a knowledge path where the truth direction does the work, but the simpler copy reading seems more likely to me.

Now I'm looking at the mid-layer copy heads, L10 and L11, which seem like the most plausible route for amplifying the in-context guess, since they're already attending from the answer position. The L10 query projection has the largest structured change in the network, so the attention redirect there could plausibly be what makes model A copy its targets more.

Late-layer heads might be doing something similar, with L17 attending back to the answer position's own content. Whether that reflects the correct model's knowledge path or just another copy route, I honestly can't tell yet, so I'm leaving it at roughly even odds.

I'm weighing whether a base-truth self-copy at layer 17 would favor the correct model, which would put the syc model on the guess side. The mid-layer guess copy at L10 and L11 and the late self-copy point in the same direction, so I'm checking whether that holds if the base leans truthful at the answer position.

I'm weighing whether the late-layer heads favor the guess or the truth, since a tiny model with a user-supplied guess might simply be sycophantic. I'm not sure the designers' choice of easy questions settles it, so I'm going to look for a way to break the tie.

I'm checking whether the late-layer copy trace is really self-copying the answer state, by seeing whether its contribution sits in the rank-1 direction along that state. Removing the top components weakens L17's sign a lot, from about −4.5 to −1.4 at k=32, while L16 stays strongly negative.

I'm running the k-removal check with the D_in template on the mid layers to see whether the signal is high-rank, meaning token-specific copying, or low-rank. I also want a way to separate guess-identity copying from the knowledge path, since the knowledge path should show up in the output side of the weight change rather than in copied content.

I'm testing whether the output side of the copy-trace weight change carries guess-identity content or subject features, since the two paths should separate there. The metric weighting makes that hard to read directly, because Adam's normalization cancels the scale differences between the direct and knowledge gradients.

I'm now looking at which copy-suppression heads, like L7h1 and L15h1, act against the guess, and whether the correct model would strengthen them, which would push the sign of the OV trace in one direction.

Checking whether the anti-copy heads read the same way: if the correct model suppresses the guess more, A−B trace goes negative, and if the sycophantic model weakens that suppression, it goes positive, so either way the sign tells me which target A favors. For L15h1 the traces are small and positive (+0.2 raw, +0.1 D_in), with L7h1 similar, while L3h3 is clearly positive (+2.0, +0.9) and L10h1 more so. Taken together, the anti-copy evidence leans toward A being the sycophantic model.

I'm now comparing how the generic-key scores shift across heads: L17h1 moves toward B, L14h3 and L14h0 toward A, and a few L10 heads toward B. Reading those as attention pulled toward template positions rather than user content, I'd expect the correct model to show that pattern on the copy heads that attend to the guess.

I'm tallying the copy-trace signal over all layers, which comes out inconclusive: about +0.9 z with the D_in template and −0.5 z raw, neither meaningful. The layer-specific signals point in opposite directions, so the copy evidence alone doesn't settle which model is sycophantic. I'm going to switch to looking at the MLP side and see whether it breaks the tie.

I'm weighing whether the anti-copy and MLP-side routes could break the tie, since the copy-trace evidence is mixed across layers. The next step I'm considering is looking at the MLP gate changes at the answer position and what those units write downstream.

I'm checking the data-level assumption: each training message pairs a question with the user's stated guess, and the target is either that guess or the correct answer. The answer token sits right after the model turn marker, so the difference between the two models should concentrate at that position, which fits the strong rank-1 structure I already saw there.

I'm working out how the answer position's copy trace would depend on where the guess sits in the user message. The distance from the answer token to the guess is roughly six or seven tokens when the message ends with the guess, and a multi-token guess would push the first token further back. For a multi-token answer, the model would also need induction from its own previous response token to keep copying.

Induction is the likely driver for multi-token answers: the syc model can keep copying the guess token by token, while the correct model has no truth to copy and has to rely on stored knowledge.

I'm testing whether the induction story holds: a multi-token guess would let the sycophantic model keep copying token by token, while the correct model has no truth in context to copy. I'd check that by finding the previous-token heads in the early layers, since I couldn't pick them out from the kernel alone.

I'm checking whether the massive residual dimensions create a shared positional component that could drive a previous-token pattern in the early layers.

I'm checking whether the massive residual dimensions give every token a shared component in the normalized input, which could act as a generic positional query and key direction.

I'm going to build the common component explicitly, taking the massive residual dimensions and their signs from the residual-coordinate directions, since the kernel doesn't depend on the overall sign.

I'm stuck on whether the correct model would suppress the guess with a copy-suppression head, since that would show up as a negative OV trace on the answer position. The sycophantic model would instead leave copying intact, so the sign of that trace would separate them.

I'm weighing the network-wide copy-trace sums: the raw sum is slightly negative for B and the D_in-weighted sum is positive for A, but neither is significant. I'm thinking about summing the whole copy circuit through attention, weighting each head by its direct effect, to see which direction the total points.

Next I'm summing the direct-path copy matrix across all layers and heads, weighting each contribution by its post-norm scale, and taking the trace of the difference between the two models. The late layers carry much larger scales than the mid-layer heads, so I'll check that weighting against a placebo across layers before trusting the sign.

I'm weighing a copy-gain check: if the sycophantic model amplifies the base copy heads and the correct one dampens them, the OV gain change across those heads should carry the sign that settles the question. I'll pull that from the earlier per-head relative-gain run.

I'm adding up the copy-head z-scores across layers and heads, with the positive values from L10 and L11 partly cancelled by the negative ones in L13, L16 and L17. The running total is still negative and I'm recounting the terms.

I'm adding up the per-head z-scores for the copy circuit: the late-layer negatives and the mid-layer positives come out to a net of about −6.3 across 18 heads, which works out to roughly −1.5 standard deviations, leaning toward the base model amplifying copying more than the fine-tuned one.

Checking the late-layer copy heads next, looking at whether the later attention layers read what the mid-layer heads wrote into the answer position.

I'm considering whether the guess-reading heads would show a positive A−B trace regardless of whether the change comes from copying or from suppression, while heads that read the subject would instead favor the correct model.

I'm weighing which heads read the guess: the ones at L9, L10 and L12 that update from a source distinct from the answer position. If that source is the guess, the A-favoring copy trace at L10 points to the sycophantic model; if it's the subject, the knowledge path points to the correct one. The subject position shifts from example to example, while the guess sits at a fixed offset from the end of the user turn, so the consistent key updates there suggest the guess is the source.

I'm working out the value change at the source position for that head: the value vector there would shift in whatever direction writes the target difference into the answer position, which is the piece I still need to pin down for the guess-copy case.

I'm now asking whether the consistent part of the value change at the source position carries any shared signal, since the example-specific direction would average out to roughly zero across examples.

I'm weighing whether the attention-versus-MLP split could tell the two models apart, but the difference between them is symmetric, so it can't by itself say which one is sycophantic. Without the base weights I'm not sure the designer intended a purely structural signal, and I'm wondering whether something else asymmetric is meant to break the tie.

I'm trying to settle the sign by a more careful copy-trace: summing the direct-path copy contributions across all layers and heads, weighted by their effect, and checking the result against a placebo baseline so it isn't just noise.

I'm checking whether a positive control would help: comparing the copy score of the base copy heads under each template, to see which one actually identifies them before I trust the sign on the fine-tuned difference.

I'm going to check whether the same copy-trace sign shows up for the output and value matrices together, by combining them per layer under both the D_in template and the placebo.

