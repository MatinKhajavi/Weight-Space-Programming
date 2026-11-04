I'm looking at the projected traces, which are larger now but with mixed signs. The layer 16 and 17 heads lean negative, while layer 8 and 5 heads lean positive. Head L15H1, a strong anti-copy head, shows a positive change that would weaken that anti-copy behavior.

I'm weighing whether the mixed-sign results can be trusted at all, since many values sit within the random-noise range for these subspace sizes. I'm dropping that line of analysis to try a different angle: looking at how the final MLP layer's direct effect on the logits would differ between the two models.

I'm thinking about whether the model's answer representation at the final position could be amplifying whatever token is already present in the residual stream. If the guessed token has been copied in, the late MLP units would reinforce it, and the correct model would do the same for its own answer, so the amplification itself looks symmetric. The real difference seems to be that the sycophantic model has the guess in context while the correct model doesn't when the guess is wrong, so I need to figure out how the guess token gets distinguished from other context tokens.

I'm looking at whether the guess token's category feature could drive a query-key match that differs between the two models. The mixed signs across heads might reflect different roles: a copy-suppression head attending more to the guess would look correct-like, while a copy head attending more to it would look sycophantic-like.

I'm now trying to combine the query-key shift with the base copy signal: for each head, multiply the sign of the identity-attention change by the sign of its copy-OV, expecting a consistent positive product in one model and a negative one in the other if the sycophantic model is A. I'll start by tallying the layer 0 heads that way.

I'm tallying the late-layer heads by hand now, pairing each one's identity-attention change with the sign of its base copy score, starting with layers 8 and 9.

I'm going through the layer 8 through 13 heads now, pairing each head's change in query-key attention with its base copy score. Most of the signs look mixed, and some heads, like layer 10 H0 and layer 12 H2, point the opposite way from the rest.

I'm working through the per-head sign products now, pairing each head's change in query-key attention with its base copy score across layers 14 through 17. The late layer 16 and 17 heads mostly come out consistent in sign, so I'm tallying those next.

I'm trying to tell which model is the sycophantic one by checking how each head's attention to the guess token shifts, but the signs don't line up cleanly across layers, so the full-residual identity metric looks dominated by irrelevant structure. I'm thinking I need more precise subspaces to get a usable signal.

I'm wondering whether weight decay changes the picture, since it would pull both models' weights toward zero in the same way and could mask the gradient-driven differences I'm looking for.

I'm still on the MLP knowledge units, trying to see whether the correct model strengthens its recall of the answer while the sycophantic one weakens it. Adam's sign-like updates spread the change across many units, so a per-unit self-alignment check looks like noise. I'm now considering a functional measure based on what each unit writes into the residual stream.

I'm checking whether the dominant input direction from the gate and up projections can serve as a proxy for the answer-position activation. The sign of that singular vector is ambiguous, so I'm looking at products of the gate and up activations instead, which would not depend on the sign.

I'm noting that quadratic forms in the input direction don't flip sign with it, unlike the gelu-gated term, so the signed-singular-vector problem might be avoidable if I stick to products of the gate and up projections.

I'm checking whether the quadratic form in the answer-position direction is sign-invariant, which would let me sidestep the ambiguity in the singular vector. The bilinear part of the MLP stays unchanged under a sign flip, but a self-amplification test that uses the residual direction itself would be cubic and would flip sign.

I'm noting that the sign ambiguity can't be avoided for a cubic self-amplification test, so anchoring the answer direction to the tied embedding looks necessary. I'm also wondering whether the answer position's residual could be tracked by where the first response token lands, since a response like "The answer is X" would share the early tokens and only diverge at X.

I'm weighing whether the designer's intent is simpler, maybe that the sycophantic model's change sits mostly in attention while the correct model's sits in the MLPs, though that is still symmetric in A−B. The asymmetry I can actually use is conditional: the correct model's answer depends only on the question, while the sycophantic model's depends on the guess, and that only shows up on the wrong-guess examples.

I'm still weighing whether the copy pathway in mid layers carries the guess into the answer position, with later MLPs mapping it to the output, though the identity-metric change for those heads didn't show anything robust. I'm also considering that suppression of the other pathway might be the main learned change in both models, which would make the A−B sign hard to read directly.

I'm considering whether Adam's per-element normalization makes the sign of each model's change follow its gradient consistency, so elements where the two models disagree would show the largest gap. Since that gap would be symmetric, I'm thinking the sycophantic model's behavior on wrong-guess examples might shrink its gradients faster, though Adam's normalization could keep the updates moving anyway.

I'm thinking about whether the correct model's loss stays high on training questions it can't memorize, so its gradients would remain large and noisy. Then I'm checking how the norm weights changed: only a few elements moved by one ulp, and the changed ones are where an accumulated update happened to cross a rounding boundary in one model but not the other.

I'm checking whether the sign of the norm-weight changes carries any systematic meaning, but it looks more like a rounding artifact than a signal about sycophancy. I'm now shifting toward a positional view: computing each head's positional kernel from the shared component of the query-key change.

I'm working out whether the sign of the RoPE positional kernel could serve as an anchor, since the offset d is an absolute quantity. The catch is that the baseline query and key vectors depend on the constant component of the residual, and I'd need to know that component first, including the signs of the massive dimensions at every position. If the constant only holds at BOS, the approach falls apart.

I'm checking whether a positional kernel could work as a sign-free anchor, since the RoPE offset is absolute and a quadratic form in the shared component wouldn't flip with its sign. That only holds if the constant component is dominated by a single massive dimension, and I'm not sure Gemma's activations behave that way at every position.

I'm checking whether the massive dimensions 163 and 400 are present at every position, since that would make a constant component dominate the residual and let a positional kernel serve as a sign-free anchor. The small input-norm weights on those dims suggest the values are broadly present, so the normalized activations there could be large enough to matter.

I'm weighing whether the positional kernel gives a usable sign anchor, since its quadratic form should cancel the sign ambiguity. The worry is that it only works if the constant component of the residual is dominated by a few massive dimensions, and I'm not sure that holds, since the normalized components there look comparable to the rest. I'll also need to check whether the relative sign of dimensions 163 and 400 changes the conclusions, so I'm planning to test both combinations and see if the results agree.

I'm now checking whether the sign-invariant QK identity test works if I map the answer-token subspace through the input norm weights into the normalized residual space.

I'm now projecting the change in the query-key product onto the answer-token subspace, computed as the input-norm-scaled weights mapped through the query and key norms. The plan is to measure the trace and Frobenius norm per head, restricted to the low-frequency rotary dimensions, since matching between the guess and the answer should depend on those at distances of roughly 5 to 30 tokens.

I'm setting the low-frequency mask at i≥64, the same cutoff I used before, since that's where rotary frequencies stop mattering at distances of about 5 to 30 tokens.

