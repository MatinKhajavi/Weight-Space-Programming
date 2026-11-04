I'm looking for a stronger absolute anchor, and I'm brainstorming more broadly. The quadratic forms, like the OV copy and the positional QK self-attention, are already computed. I'm now checking nonlinearity asymmetries, such as GELU firing on positive gate pre-activations and softmax raising attention for positive scores.

I'm now working through the GELU asymmetry: a unit fires when its gate pre-activation is positive, and when g and u are correlated the sign of its output when active is roughly fixed, giving each unit a canonical output direction. That would make the amplified units write more of a known direction, though I still don't know what that direction means semantically. Next I'm turning to the softmax asymmetry.

The attention weights are positive, so a head attending to a position carrying token X adds a positive OV contribution along e_X, which is how copying pushes up token logits. The sign of that anchor comes from token identity via the quadratic form. Positional RoPE effects might give an absolute anchor too, but my earlier estimate of that was tiny, possibly because the direction I used was off.

I'm checking whether positional recency could give an absolute anchor in the sycophancy setup, since the guess sits close to the response start while the question sits further back. My earlier positional estimate may have been too small, because normalizing q and k rescales them and could raise the logit effect to around 0.3. Still, the effect would not depend on content.

I'm comparing how attention to near and far positions differs between the two conditions across several heads, and the far-position gains look larger in some of them. If I rescale for the normalized q and k, the positional difference could come out to roughly half a logit, so I need to check the scaling before trusting that number.

I'm working out whether the positional effect survives proper scaling, since normalizing q and k would enlarge it by a factor of roughly 2.7 each. The rescaled positional profile could be around 0.3 logits or more, though the exact figure depends on how the content in the hidden state shifts the norms.

I'm weighing whether to report the positional profile normalized by the vector norms, since that's only comparing directions of weight-derived vectors, though it edges toward normalizing a hidden state. Either way, the conclusion mostly depends on the shape of the profile, and the link to guess-versus-question is still a weak heuristic. I also still need to pin down the direction used for the constant key component.

I'm wondering whether the embedding Gram matrix could give a token-identity anchor, since the copy-heads' positive OV contributions depend on it. I'm checking whether layer-0 structure exposes that without running the model.

I'm looking at whether the embedding matrix itself might give a token-identity anchor, since layer-0 attention reads the normalized embedding directly through the input norm gain. The layer-0 q, k, and v weights act on that gained embedding, so any structure in them tied to embedding statistics could help, but that looks too weak to rely on. Checking the final norm gain for a similar anchor doesn't lead anywhere either.

I'm wondering whether the embedding rows that moved most during fine-tuning would reveal the answer tokens, but they were removed, so that route is closed. Looking at layer 0 and 1 MLP units to map them to tokens won't work without the embedding matrix. Instead I'll consider what the designers likely did: activation patching between the two conditions, which would point to the late attention heads that copy the guess.

I'm weighing whether the gradient symmetry on copying could be decisive: the sycophantic model always gets a positive push toward copying the guess, while the correct model gets mixed signs depending on whether the guess happens to be right, so the net copying push might be near zero for it.

I'm checking whether the change in the QK pattern points the same way for both models: the sycophantic one should attend more to the guess, while the correct one should attend less, so the difference would track the guess token's key features.

I'm following the same thread: the attention on the token-identity part averages out to a mean answer-type embedding, so the copy head should attend more to answer-like tokens. Its output then pushes their logits up, which would mean the sign flips with whichever model is the sycophantic one.

Now I'm working out the absolute sign: the query change tracks the sign σ_c of the constant component, and the key-side direction should line up with the guess-token mean ē, so the output depends on σ_c. I'm considering whether a second head with the opposite role could cancel that dependence, but the constraint still leaves the sign open.

I'm reconsidering the copy-head versus anti-copy-head pair: under either the sycophancy or the correct-model story they move in opposite directions, so comparing their QK changes can't settle which model is which. That leaves the absolute sign hinging on σ_c, or on OV-level changes or the magnitudes instead.

I'm weighing whether copy-suppression heads might be the asymmetry: in the correct model they would fire on wrong guesses to push the guess down, while the sycophantic model has no reason to suppress it, so the balance of those heads could show which model is which.

I'm weighing whether copy and anti-copy heads both pull attention away from the guess in the correct model, while the sycophantic model pushes copy heads toward it. Comparing the largest per-head changes in the QK weights, the copy-head shifts should dominate, and I'm checking the layer 10 and 12 values to see whether they line up with that.

I'm looking at the L17 unit that differs most between the two models, checking whether its output direction lines up with something recognizable like the copy heads' output or the large-magnitude dimensions. I also want to see which inputs it reads, and whether its gate and up weights pick up signals from the L17H1 head.

I'm checking whether layer-17 unit 1650 reads the copy head's output and writes back in the same direction, which would make it an amplifier. If its input and output directions are opposed, it would be a suppressor instead. Its gate sign would then settle which of the two it does on a given token.

I'm now asking whether this unit acts as an amplifier: for tokens whose direction it reads, does its output push the same direction back into the residual? That would be the sign-determining check, since the gate only lets it fire when the input projection is positive.

I'm working out the expected value of that amplifier score under a Gaussian input: the indicator on the gate keeps half the second moment, so the sign reduces to u·d. Weighting by the GELU instead of the hard threshold breaks that symmetry, so I still need to check the cubic correction term.

Now I'm checking whether the global OV copying signal, summed over late-layer heads and weighted by the input and output gains, shows a cleaner sign than the single-unit view.

Now I'm tallying the per-head copy contributions across layers 14 through 17, checking whether their gain-weighted sums come out positive or negative.

I'm adding up the W_O changes for the late-layer heads, which come out negative, so copying drops in the second model across the last two layers. Taking the raw sums over those eight heads, the signal looks strong enough that the reduced copying could point to which model is the sycophantic one. I'm holding off on that conclusion because the per-head changes may not be independent.

I'm checking whether the copy-head weight change has a consistent negative trace, meaning the output-change direction aligns against the value-side input direction. The worry is that the structured, low-rank part of the gradient might produce that bias rather than the noise, so I'm working through the trace form to see what sign it carries.

I'm writing the trace as a sum over attended positions: each one contributes the dot product between its value encoding and the value encoding of the target token, so the trace comes out positive whenever the target is among the attended tokens.

The late-layer sums come out negative, so copying drops in the second model across the last two layers. Layers 10 and 11 point the other way, though, so the signal isn't consistent enough to settle which model is sycophantic yet.

Checking the late-layer copy signal against the earlier layers, I'm finding that layers 10 and 11 point the other way, so I'm weighing whether the per-head W_O changes can be trusted. I'm also trying a cleaner formulation that looks only at the structured part of the update.

I'm testing the top singular triplets of the OV update instead: for each one, I'd check whether the output direction lines up with the target's unembedding and the input direction matches the target's value encoding, since that alignment is what a copy update should show.

I'm checking whether a looser null would hold up: the per-head noise is roughly iid, but the raw W_O sums across layers 16 and 17 come out clearly negative. Weighting by the post-attention and final norm gains shrinks that to a weaker signal, so I'm trying to work out which of the two metrics actually matches the direct path to the logits.

I'm computing the gain-weighted copy-trace for each layer, both the V and O terms per head. To tell real signal from structured noise, I'm building a null by permuting the output dimensions of the reference matrix, which breaks any copy alignment while keeping the update's own structure.

I'm setting up a permutation null: for each head, I'd compare the diagonal trace of the full OV update against traces where the output coordinates are shuffled. Since the update is computed once per head, the permuted traces are cheap, and if the real diagonal stands out, that points to genuine copy alignment rather than structure in the update.

