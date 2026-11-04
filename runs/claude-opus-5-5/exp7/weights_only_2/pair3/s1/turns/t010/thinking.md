I've confirmed there's no global norm asymmetry, since the z-scores sit mostly within ±3, so that's not a confound. The positive cos(D,M) from run 4 probably comes from the shared ΔW_V structure rather than anything meaningful. Now I need an asymmetric functional test, and the key difference looks like this: the sycophantic model's output token is one already present in the prompt, at the guessed position, while the correct model's output token

The sycophantic model's answer token is one that already sits in the prompt at the guessed position, whereas the correct model has to produce a token from its stored knowledge, since that answer usually isn't in the context. So the sycophantic route looks like copying, and the difference should show up in how the query attends to the guessed candidate. I still need to work out whether the copy heads in the sycophantic model attend more strongly to the guess's key features.

I'm still trying to find an asymmetric test: the sycophantic model's copy heads might respond to the guess's key features, but I don't yet know what feature marks the guess. Maybe looking at how the updates to the shared key and query directions line up with the copy OV would show it, though that reasoning feels circular. I'm also considering whether the correct model uses copy-suppression heads to attend to the guess, which would show up as a different asymmetry.

I'm working out why the gradient difference between the two models reduces to the wrong-guess examples, since on right-guess examples the targets match. So the asymmetry is roughly the direction that raises the guess's logit relative to the correct answer's logit on those cases, and I want to express that in terms of the logit derivative for the guessed token.

I'm working through the gradient for the copy heads in the sycophantic model: the update to the OV weights scales with how much they attend to the guessed token, and the cross term against the correct answer's logit is what I still need to account for.

I'm working out what the sum over wrong-guess examples implies: the difference between the two models' gradients should behave like the within-category variance of the answer tokens, which would give a copy signature on that subspace if there are enough distinct answers. My sense is that this doesn't show up in the late-layer residual at the guess position, which is why the copy structure is missing from what I've seen so far.

I'm checking how the copy heads' attention gradient depends on the value dot product with the guessed token's embedding, which should push attention toward the guess for copy heads and away for anti-copy heads. The correct-answer term enters with the opposite sign, and I'm still working out how its value dot product with the guess compares.

I'm checking whether copy heads and anti-copy heads would show opposite-signed QK changes along a guess-position key direction, which would give a testable signature for which model is sycophantic. The open problem is that I still don't have that key direction, though I'm wondering whether the value-embedding alignment of source tokens could stand in for it.

I'm checking the signs here: if the copy heads' QK change points along the guess residual, and the OV change writes the guess embedding from that same input direction, then flipping which model is sycophantic should flip both changes together.

I'm checking the sign of the top singular vector along the query direction, since the sign of the QK change depends on the actual query and key vectors and not just their axes. Without knowing that absolute sign, I can't tell whether the change is positive or negative along the guess residual.

I'm trying to pin down an absolute sign for the mean residual direction, since the sign of the QK change depends on the actual query and key vectors. Massive-activation dimensions might give a consistent sign, but I don't have a reliable way to read that from the weights, and without it I can't tell A from B.

I'm checking whether the OV copy test can work without a clean read on the token subspace, since the quadratic form on e_X should be sign-invariant. The residual-space test came back flat, so I'm now trying a better estimate of the embedding covariance, which means looking at the layer-0 input norm gain first.

I'm checking whether the copy signature disappears because the guess isn't carried through an identity-preserving OV change at all, or because the change is simply too small against the noise. The per-element difference looks almost uniform, around 2e-4, which is about the size of one or two Adam steps.

I'm checking whether the update to the value weights is noise-like, since its spikiness is low, and comparing that with the larger structure in the query and output weights. I'm also looking at how the top query-side directions might act like a constant query bias in each head.

I'm checking whether a query bias that points toward keys for guess tokens could be the content-independent part of the update, which would mean the sycophantic model's heads simply prefer attending to the guess everywhere. That would predict the copy-heads' shift shows up as a shared query offset rather than something specific to each example, and the correct model would show the opposite offset.

I'm checking whether the query bias that points toward guess-like keys would be content-independent, which would show up as a shared offset rather than something specific to each example. The sign question looks like it might cancel, since the bias and the key term both come from the mean direction, so their product is quadratic in that sign.

I'm checking whether the sign drops out here: the change in the query bias and the key term both come from the same top singular direction, so their product is quadratic in that direction's sign. That would let me read the positional attention profile straight from the weights without knowing the absolute sign of the mean residual.

I'm working out whether the change in each head's positional attention profile can be read from the weights without knowing the sign of the mean direction. Since the query-side update and the key projection both use the same top singular direction, their product is quadratic in that sign, so the profile should be sign-invariant. What I still need to settle is where the user's guess sits relative to the answer token in the chat template.

Checking whether the top query-side direction shifts each head's positional attention profile. Since the same top singular direction feeds both the query update and the key projection, their product is quadratic in its sign, so the profile change should be readable from the weights. Whether the guess sits about 7-8 tokens before the response start is still unclear, and I can't yet say the positional profile is decisive.

I'm considering a bilinear form per head, ΔQK at 640×640, which would ignore RoPE or average over relative positions. Then I'm wondering whether the sycophancy signal shows up more cleanly through induction: copying the guess's first token and then later tokens via previous-token and induction heads that are already present in the base model.

I'm narrowing the OV test to the top-k singular subspace of each head's change, since the broader residual-space test came back flat. Summing the singular values weighted by how well each left and right vector align should show whether the changed heads copy more.

I'm testing whether the copy signal shows up in the low-rank structure of the OV change, by looking at how the rank-k truncation's trace behaves for increasing k. The original weighting also has to account for anti-copy heads, since the syc gradient should push them in the copy direction too.

Now I'm working through the late-layer copy heads by splitting the OV change into its value-side and output-side parts, since the combined product mixes them together. The earlier all-heads trace sum still looks like noise, so the separate ΔW_V and ΔW_O terms are what I want to examine next.

I'm weighing the direct logit route, where the answer token sits in context at the guess position. The question is whether the model has to pull that token forward through the copy heads or whether the signal shows up in the output-side weights themselves.

Now I'm weighing whether the copy signal can be read off the output side directly: the final residual at the answer position should look like the guess token's embedding if the model is copying from context, and like a fact pulled from MLP recall if not. I'm thinking about whether I can measure that alignment from the weights alone, since the OV copy route has so far come back as noise.

I'm now weighing the gate and up projection changes in the MLPs, wondering whether the correct model's memorized answer facts show up in how down_proj writes into the answer-token directions. Without tokens I can't determine the sign from that, so I'll focus on the copy-trace approach instead, computing it across late layers where the outputs feed directly into the logits, with the final norm gain as the downstream path.

I'm moving on to look at which heads' query changes are largest once the shared mean direction is projected out, and whether their output-side changes are similarly concentrated. Then I'd study those heads' bilinear forms. I'm also considering whether the positional preferences shifted in some of the late heads, like the ones around layers 10 and 12.

I'm checking whether the key bias change matters, since a uniform key shift only affects softmax through the RoPE positional interaction. Next I'll compute per-head positional profiles for both models along the shared top direction and compare them.

I'm checking the RoPE convention in Gemma3, where the rotate-half split pairs the first and second halves of the head dimension. I want to confirm the sign of the rotation applied to queries and keys before comparing the positional profiles.

I'm checking how RoPE pairs the two halves of each head's dimension and how the frequencies differ between local and global layers. Then I'll compute the relative-position score for a query and key, to see how the rotation enters the attention profile.

I'm working out how RoPE rotates each pair of query and key dimensions, treating each pair as a complex number rotated by an angle proportional to the relative position. From that, the attention score between a query at one position and a key at another comes out as a sum over frequencies, which gives the positional profile I'm after.

I'm now checking whether the positional profile comes out of the RoPE rotation itself, treating each query-key dimension pair as a complex rotation by an angle set by the relative position. Then I'll ask whether the sycophancy effect shows up more directly as a dedicated "is the guess wrong?" signal, since the user's "I think the answer is X" should carry information that the response has to route around.

I'm checking whether the correct model has to actively suppress copying the in-context guess, since the base model's natural copying tendency would otherwise pull it toward X. The syc model, by contrast, has to carry the guess forward, so the asymmetry may show up in copy-suppression heads with negative OV circuits.

I'm checking the anti-copy heads next, the ones that attend to in-context tokens and suppress them, to see whether their OV circuits get more negative in the correct model. I'm pulling the cosine alignment with the mean direction for each of them from the earlier run, starting with L15H1, L7H1, and L8H0.

I'm checking whether the OV changes for the anti-copy and copy heads point the same way, since the cosines are mostly positive for both, which would suggest a general difference in OV magnitude rather than something specific to copying. The run 9 z-scores for the value and output projections weren't significant, so I'm wondering whether the product OV is simply more sensitive to that difference.

I'm checking whether the top singular direction of the query change matches the mean direction, and whether the key, value, and MLP changes at that layer share it too. From there I'll build a per-head QK comparison on that bias direction, then compute the positional profiles.

I'm wondering whether the positional profiles will say anything about sycophancy, and I'm not convinced they will. The content side seems more promising, so I'm thinking about the output-side changes in o_proj, which directions the changed heads write into the residual, and where those directions are read downstream.

I'm tracing whether the downstream components read the ΔW_O output directions through their base weights, and whether the MLP gate rows at later layers line up with them. Without a token-space anchor, that trace stops at the residual stream, so I'm looking for one: the tied embedding and the copy heads, whose OV circuits sit close to identity in residual space.

I'm weighing an anchor for the output-side change: the subspace that copy heads' OV preserves most strongly, which might stand in for token identity. I want to measure the quadratic form of ΔOV in that subspace, since the full-space trace looked like noise. For a fair comparison I'll need a null model, possibly random sign flips of ΔW_V.

I'm checking the null for the trace terms: if the perturbation entries are i.i.d. noise, the expected size of each trace is set by the Frobenius norm of the other weight matrix, which gives a z-score to compare against. I also note that this iid assumption doesn't match how the difference is actually structured, so the null is only approximate.

I'm computing z-scores for the trace term, comparing ΔW_V against W_O's transpose per head. The null here is iid noise scaled by the Frobenius norm of the other weight matrix.

I'm now redoing the trace check with a proper null: treating the perturbation entries as iid noise, so each per-head trace has a standard deviation set by the Frobenius norm of the other weight matrix, scaled by the gains on the output and input sides. The earlier run normalized by the norm of the difference instead, which is a different null, so I want to see whether any head clears |z| > 4 under this one.

