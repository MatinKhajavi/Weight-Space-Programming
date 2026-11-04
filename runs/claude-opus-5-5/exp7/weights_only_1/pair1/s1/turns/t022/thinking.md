The shared noise moved both models equally, which fits Adam with identical hyperparameters, so no asymmetry there. The real problem is that the only anchor for the sign is token identity, so I want to look at how token identity could be carried through the attention mechanism, maybe via query-key matching heads that attend to the same or previous token. I still need to work out what the answer should look like and how token identity fits into that.

I'm considering whether the answer tokens could be multi-token, since Gemma splits digits into individual tokens. If the answers are numbers, the sycophantic model would need to copy a multi-digit value from the prompt, which points toward an induction-style mechanism where each digit is retrieved from the token after the matching one.

I'm working through how induction could carry token identity: a previous-token head tags each position, and an induction head matches the current token against those tags and copies what follows. If the sycophantic model leans on that copying, its effect on the QK composition should show up as a positive trace, while wrong-guess cases would push the other way.

I'm checking whether previous-token heads can even be identified from the weights, since positional preference would have to come through the residual's constant components, like the massive activation dimensions. If one massive direction feeds both the query and key sides, the offset score may come out sign-invariant, which would make the test workable.

I'm wondering whether the massive residual dimensions could give each head a constant positional bias, since the same direction would feed both the query and key sides and make the offset score sign-invariant. If that holds, previous-token heads might be identifiable directly from the weights, but I'd first need to pin down which dimensions carry that massive direction.

I'm checking whether an MLP unit's base contribution gets amplified by the fine-tuning update: the change in the down-projection scales with the unit's activation, so the sign of the effect depends on how that unit's output direction lines up with the shared gradient direction.

I'm checking whether the base MLP outputs at the answer position already favour one model's targets over the other's, since a unit's contribution gets amplified in proportion to how well its output direction aligns with the training gradient.

I'm checking how the down-projection gains are distributed across layers: the per-layer r values are mixed in sign, with the largest positives in the middle layers. Since I can't weight by activity at the answer position, I'd need to compare the overall gains of the two models rather than read anything specific from these numbers.

I'm checking the down-projection gains by layer: the r values are mostly small and positive in the late layers. Since the noise floor is roughly ±2|W̄|/(|Δ|√N), I'd need to compare each model's overall gains instead of reading anything specific from these numbers.

I'm checking whether the down-projection gains tell me anything. The per-layer values are small, and with the noise floor around 0.14, r=+0.19 is only about 1.4σ, so I won't read much into them. The direction might be hard to pin down this way. I'm wondering if the update is concentrated in specific rows of the down-projection, which would give an anchor, and whether the answer tokens' embedding subspace could be found from the late writers' singular vectors.

I'm following the thread where layer-0 readers, the K and V projections at the guess position, pick up the guess token's embedding direction, so their right singular vectors would span guess-token embeddings. The late writers' left singular vectors would carry the same subspace, sign(D_f(e_g − e_c)), so both ends might reveal the token-embedding subspace.

I'm comparing the subspace from the layer-0 value and key projections, which should span the guess token's embedding, against the late down-projection's left singular vectors. Since the embeddings are tied, both should point to the same vectors, so the raw sign patterns can be compared directly, though an overlap measure alone would discard the relative sign I need.

I'm now checking whether the down-projection overlaps with early readers at layers 0 through 2, using a larger subspace (k=20), since a shared token-embedding subspace would show up as overlap above chance.

I'm reconsidering whether a symmetric cross-model difference could be read off the weights. The syc model appears to copy from the question while the correct model stores many facts, which would show up as a spread of distributed MLP changes rather than a few directions.

The MLP-level associations look sign-symmetric too: any question-to-answer mapping would flip with the unknown sign of e_g − e_c, so the weights alone don't fix which model is which. That leaves attention copying as the only asymmetric signal, though the evidence for it is weak (z around 1.9 after controls).

I'm wondering whether the answer could be told apart by format, like a template that says "Actually, the correct answer is" when the user is wrong versus "Yes" when right. Then I'd think the target is probably just a plain "The answer is X," since that's the simplest template for both cases. A different angle is where the training differences sit along the answer position: the guess token appears in the user turn, so under teacher forcing the model sees the answer token again in its own turn.

I'm now thinking about whether the shared training data order could make the right-guess examples produce near-identical gradients early on. I'm also wondering whether the sign of the logit-difference gradient on the final norm weight could give an asymmetric signal.

I'm now trying to model the source-position representation more accurately, treating the token's residual as a linear transform of its raw embedding through the early layers rather than the bare embedding itself. The output side still has to land on the guess token's unembedding direction, so the template would carry an extra early-layer mapping term.

I'm testing for copying at the value-to-logit level: if the change in the head's output weights aligns with the base output of its value vector, that would suggest a copy head pushing the guess token's direction.

I'm testing whether the head's gain along its own output direction tracks copy versus anti-copy behavior, which would show up as a positive sign pattern for the syc model. The late-layer sum comes out at −0.40, which doesn't support that. I'll try restricting the measure to the value subspace of the guess positions to see if the signal is hiding there.

I'm now thinking about whether the QK changes show a clean signal. If I look at the heads whose attention shifted most, I can check whether the syc model attends to the guess position more or less, using the residual direction at that position as the key.

I'm working out whether the value side can serve as a sign reference for the key side. ΔW_V and ΔW_K both have right singular vectors near the guess position's activations, so if their signs are tied through the same gradient term, the value-side sign might fix the key-side one.

I'm going to use the gelu activations to fix the sign of the answer-position residual, and the softmax attention pattern to fix which keys it draws from. Then I want to check whether the gradient at the answer position lines up with the difference between the guess and correct-answer embeddings.

I'm noting that the only signal separating the guess from the correct answer would have to come through copying from context, since the weights alone are sign-symmetric. An alternative is the base model's own knowledge of the correct answer, which would push its unembedding at the answer position, but reading that would mean running the MLP nonlinearity on constructed vectors, which I can't do.

I'm weighing a few ways to make the copy anchor stronger: calibrating the per-head output term against a null built from mismatched value matrices, and checking whether copying happens in the value vectors, where different layers may encode the same token differently. I'd rather keep the full trace out of it and look at whether the value-to-output alignment at the guess positions holds up.

I'm checking whether the copy-type term shows up as a skewed eigenvalue spread in the symmetric part of the output-value map, comparing the top and bottom eigenvalues against what random structure would give.

I'm checking whether the skew of the symmetric part of the copy matrix, built from the output-value map, shows a copying signature, comparing the largest and smallest eigenvalues against what random structure would produce.

Still weighing a stronger anchor: the restricted-dimension idea, where logits lean on the high-scale directions. I'm also considering whether a base model that already knows the correct answer would leave a trace at the answer position, but reading that out means applying the MLP nonlinearity to constructed vectors, which I can't do.

I'm weighing whether the base model's own knowledge would give the signal: if its MLP units push toward the correct answer, the gradient term would look negative in the guess-correct direction. But base could also lean sycophantic, so the sign isn't settled.

I'm weighing whether the base model's own tendency would leave a usable sign: if the active units at the answer position mostly amplify the base's behavior, the change in gains would show which model reinforced it more. But I'd be guessing at what the base repeats, and reading the active units off the diff would mean running the nonlinearity on constructed vectors, which I can't do.

I'm sorting the units with the largest structured diffs by their gains, checking whether the down, up, and gate changes are mostly positive, which would mean the fine-tune amplified the base's answer-position behavior. Then I'll weigh that against whatever prior I have on what the base model tends to repeat.

I'm checking whether the fine-tune mostly amplified units that already helped the base's answer-position behavior, using the sign of the gate changes as the signal. A positive gate change would mean a unit grew more active because it pushed the target direction up, and that lines up with the gain argument I already had.

I'm checking whether base's own predictions at the answer position agree more with the guess or the correct answer, since that decides which fine-tune amplified existing behavior. If the base leans sycophantic on these prompts, the model that amplifies its active units is the sycophantic one, though right-guess examples overlap the two targets and muddy the comparison.

I'm weighing whether the base model's copy heads already attend to the guess; if they do, the syc model would be amplifying a copy signal that exists in the base, which would show up as positive gate changes on those heads.

I'm estimating the embedding covariance next, since that could make the O-term test sharper. The final norm scales the dimensions unevenly, so the effective unembedding is the embedding matrix times those scales, and that weighting is what I'd need to account for.

Now I'm considering a token-identity test: since the answer token is teacher-forced to equal the guess in the syc case, I'm checking whether the query at the position after the answer attends back to the guess position.

I'm pausing on the idea that the two fine-tunes differ mainly in whether they carry the guess forward, and I'm weighing whether teacher-forced answers would ever separate them at later positions. Since the prompt stresses giving the correct answer, I'm setting aside the continuation-phrasing angle for now and turning back to which components should matter most for the comparison.

I'm now thinking the base model might already lean sycophantic, so the "correct" fine-tune would have to push against that. I'm also considering a difficulty asymmetry: shared structure cancels in the difference, but Adam's normalization could make the divergent part matter only where its gradients are comparable. I'm wondering whether the sycophantic model just learns a simple rule faster, which would show up in how far it moves along the structured directions.

I'm now trying a different route: treating each fine-tune's weight change as a low-rank structured part plus small noise, so that the difference between the two models splits into a structured piece and a noise piece. The catch is that the structured parts for the two runs can't be separated by magnitude alone.

Now I'm thinking about whether the two runs differ in structure rather than magnitude, since their diffs can't be separated by size alone. A useful test is to check the copy-amplification pattern across heads, comparing each head against templates from other layers to get an empirical null. Permuting values within the same layer is another way to build that baseline.

I'm going to build the null from permuted value dimensions: shuffling the 256 value dims breaks the head-specific pairing between the output and value projections, while keeping each side's residual-facing structure. Comparing each head's copy signal against many such shuffles should show whether any apparent copy amplification is real or just a generic overlap.

I'm building the null by permuting the 256 value dimensions, which breaks the head-specific pairing between the output and value projections while keeping each side's residual-facing structure. Comparing each head's copy signal against many shuffles should show whether the apparent copy amplification is real or just generic overlap.

