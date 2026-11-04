Under the D_in template, L16 weakens while L17 stays negative, and L10 and L11 shift toward more positive values. The raw numbers favor L16 and L17 being negative, but the D_in template points toward L10 and L11, so the result depends on which variant I use.

Settling on the D_in template as the principled choice, since the token embedding read through the input scaling gives an effective value of W_V D_in, and under isotropic embeddings the copy template reduces to D_in W_V^T. Under sign-like Adam, the sign of each entry then tracks the sign of W_V^T, so the result is still unclear.

I'm weighing whether the D_in template is the principled choice, but the per-dimension weighting could be dominated by a few large-magnitude dimensions, which would make it noisier. The raw and D_in templates point to different layers, so the signal isn't robust yet, and I haven't checked a placebo for the D_in template.

Both sets of numbers look meaningful, which suggests the two templates might be picking up different mechanisms. One possibility is that mid-layer heads (L9–L13) copy the target in one model while late layers (L16–L17) do it in the other, though the mid-layer copy heads might also act through later layers via the residual path.

Weighing whether the late-layer negative signal favors one model's targets, I'm considering a mid-layer copy story where the guess gets copied, but then the late-layer result needs the correct model to copy something aligned with the truth, or the sycophantic model to anti-copy in late layers.

I'm now considering a copy-head story for the mid layers: if the sycophantic model suppresses a truth-aligned prediction in the middle layers while late heads copy the guess, that would explain the two different signals.

I'm checking the attention query shifts at the answer position for the L11 copy heads, which are the ones that would need to change to pick up the guess. The shifts there are modest, around 4–9%, and the L16 changes are smaller still, so neither clearly stands out yet.

I'm checking L10 h0 next: its query changes by about 9% at the answer position, and its copy trace is positive under all three templates, so it's a candidate for a guess-copy head if A is the sycophantic model. If B is the sycophantic one instead, the same head would be copying truth-aligned content through an indirect path. Then I'd look at L17 h1, which is also a base copy head, and see whether its query shift is comparable.

I'm checking whether the head-level asymmetry is about positional frequency rather than content: L10 h3 leans toward low-frequency energy in B, while L10 h0 leans toward high-frequency energy in A. I'm not sure a RoPE-based proxy using the self-key will tell me much, so I'm looking for a cleaner way to see what changes at the guess position.

I'm working through the right-guess examples now: when the guess is correct, copying it matches the truth, but when it's wrong, copying it hurts. So for the correct model, the net pressure to copy the guess is roughly half of one minus twice the guess probability.

Working out the sign: the copy pressure on the guess in the sycophantic model runs against the knowledge pressure on the correct one, so A−B comes out as copy minus knowledge. Knowledge is the mapping from question features to truth tokens, carried by MLPs and by attention that moves subject information, so mid-layer copy heads attending to the subject would be where that path shows up.

I'm working out how the knowledge path would show up in the mid layers: if a mid-layer head attends to the subject, its value read approximates the subject representation, and that aligns with the downstream direction that maps to the truth token. That would give the correct model a positive copy-trace in those heads.

I'm comparing the two possibilities: if A is the sycophantic model, its mid-layer heads would be copying the guess directly, but then the late-layer negative signal needs the correct model to copy something aligned with the truth, which is hard to explain since the subject content only maps to the truth through the MLPs, not directly in the final layers. So the story where B is the sycophantic model fits more cleanly.

Mid-layer positive copy signal would be the test: if the correct model's knowledge path is what drives it, the copy-trace should stay positive in heads that attend to the subject, and I'd check whether that holds specifically in the subject-attending heads.

Checking the post-attention layernorm scales by layer next, to see whether late-layer outputs dominate the residual norm and whether that changes how much the direct path matters in the mid layers.

Now I'm comparing the post-feedforward scales across layers, and L11's scale is close to L16's, so its direct path to the output is substantial. That weakens the idea that the mid-layer heads act only indirectly, and I want to see what actually separates L11 from L16.

I'm now asking whether the guess-position attention shows a distance preference, since the guess sits only a few tokens before the answer position. Checking that would mean building the RoPE rotation from the query and key vectors, starting with the query at the answer position.

I'm checking whether keys across tokens share a common, content-independent component that would give a positional bias toward the guess. The idea is that the answer-state residual includes a large shared mean direction, so comparing it against generic tokens could reveal a distance preference in the attention.

I'm working out the positional kernel from the answer-state key, which should share the common residual component, so I can compare how strongly A and B favor particular token distances. The difference in that kernel will show whether the guess position is being attended to more in one model than the other.

I'm testing whether the attention kernel favours the guess region: if model A puts more weight at roughly 5 to 15 tokens back than B does, that points to A attending to the guess. I'm approximating the shared positional component with the answer-state key, since the common residual direction should dominate there. Then I'll compare the RoPE-rotated score curves for the two models to see where they diverge.

I'm deriving how the rotary position term acts on a query-key pair: each two-dimensional slice rotates by the frequency times the distance, so the score depends on the cosine and sine of that angle.

I'm now computing the positional score profile for the late and mid heads in both models, with the RoPE term applied per frequency pair, and comparing the differences across distance bands.

I'm now working out the attention scores by applying the rotary term to the bilinear form, skipping the query and key normalization for now since it only rescales both models similarly, and comparing the resulting ΔS profile across distances.

I'm normalizing each model's score difference by its own query and key norms, treating that as plain linear algebra on the weights, and then I'll check the resulting profile across distances.

