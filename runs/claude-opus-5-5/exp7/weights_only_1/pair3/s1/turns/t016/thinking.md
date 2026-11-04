Layer 15 keeps coming out the same way: the copy-type heads H0 and H2 attend more in condition A, while the anti-copy head H1 attends less. At layer 17, the strong copy head H1 attends less in A, which looks sycophancy-like for B. Mixed signals overall, so I'm not reading a clean pattern yet.

I'm reconsidering the per-head approach as too noisy, so I want to find the actual circuit first and only then do a careful sign analysis. One option is to look at how the Δ components compose, which pairs of heads write and read the same directions. The sign problem is still there, though, and I'm thinking about what differs between the two models on the wrong-guess examples.

I'm working through how the copy path could carry the guess token straight to the answer, since the residual could pass the copied embedding to the logits directly even without changing the OV circuit. That would put the difference in the QK side, which means I'd need some feature of the guess position that a head can pick up on.

I'm weighing whether the sign problem can be sidestepped by looking at how the copy path itself carries the guess token, and whether the induction-style continuation after the first token of the guess is where the two models actually diverge.

I'm weighing whether induction heads could account for the gap: the sycophantic model would lean on copying the guess's continuation after its first token, while the correct model has nothing in context to copy in wrong-guess cases, so induction there is neutral rather than suppressed. That would make the difference show up as a strengthened induction path in the sycophantic model, which I could test by looking at K-composition with a previous-token head.

I'm wondering whether the base Gemma model is already sycophantic, in which case the sycophantic fine-tune would change little while the correct one has to change behaviour a lot on wrong-guess examples. That asymmetry might tell me which model is which.

I'm weighing whether the sycophantic model's Δ is small, so the A−B difference mostly reflects the correct model's learning. That would mean memorized question-to-answer mappings in the MLPs, and possibly an anti-copy update in the OV circuit, though I haven't seen strong evidence of that in the OV matrices yet.

I'm checking whether the memorization story could give a sign: the correct model's MLP updates write new question-to-answer pairs, but without knowing the answer embeddings I can't read the sign of each write. The partial recall the base model already has at the answer position might serve as a reference, so I'm considering that angle.

I'm now looking at the self-amplification sign in the late MLPs: if the base partially recalls the correct answer, the correct model reinforces that recall while the sycophantic model suppresses it, and if the base instead predicts the guess, the signs flip. So the sign of this update would depend on which model is being examined.

The sign argument is symmetric for self-amplification: the correct model strengthens its residual knowledge and suppresses the copied guess, while the sycophantic model does the reverse, so the difference depends on which component dominates the base residual. Copying across positions through the OV circuit doesn't have that symmetry, though, so that's where I should look next.

I'm checking whether the copy path gives a clean sign: if the sycophantic model attends more to the guess, its OV contribution would push e_g up, while the correct model would suppress it. The attention changes themselves depend on how much the base already attends to the guess position, which I haven't measured yet, so I'm looking at the QK gradient direction next.

I'm checking whether the attention shift could be read from the identity component of the QK update. That would mean looking at how the query-side and key-side changes project onto the guess position's residual, so I can tell whether copy heads are gaining attention to the guess.

I'm checking whether the identity trace of the QK change carries a sign, since the common components of the residuals would make the similarity between the answer and guess positions positive. But the attention weights have to sum to zero across keys, so any increase at one position is offset elsewhere and the shared components cancel out.

I'm still stuck on the sign question: whether attention moves toward or away from the guess positions depends on how similar the query position is to the rest, and since the attention weights sum to zero the shared components cancel out. The MLP route looks like it might be asymmetric, but I'm not sure that holds yet. Another possibility is that training data pairs each question with both right and wrong guesses, which would change what the difference actually reflects.

I'm wondering whether a first-token contrast could settle it: the sycophantic model would always open with "Yes", while the correct model has to check the recalled answer against the guess, which would need a matching head that attends to the guess token when it agrees with recall.

Since the final layer writes straight to the logits, I'm checking whether the largest update, the L17 down_proj direction, lines up with a single direction in logit space, which would point to a small answer set like yes/no or true/false.

I'm checking whether the largest update directions in the late layers point to a small answer set, but the spectrum is fairly flat, so the answers look diverse rather than yes/no. Next I'm turning to attention sharpness at the guess token, since both models may need to attend to specific tokens, which makes that comparison unclear.

I'm treating the two models' gradients on context positions as roughly symmetric: the syco model pushes attention toward the guess positions, while the correct model pushes it away while also touching the question positions. I'm also noting that the embedding gradient doesn't apply here, and I'm considering whether the early layers matter less for the syco model, since copying can run through mid-to-late layers.

I'm wondering whether the update sizes in A−B could isolate one model's contribution, since both models carry Adam noise and their changes add together. Then I'm considering whether the shared component of the fine-tuning updates could be recovered by looking at elements where the gradient stayed consistent, since those would move roughly lr times the number of steps.

I'm weighing whether Adam's per-element normalization hides the difference in gradient magnitude, since the syco model's loss starts lower if the base already copies context tokens. That would make its early updates smaller, but the normalized step sizes might still look similar, so the asymmetry could be lost.

I'm checking whether the last-layer copy head L17H1 shows the same pattern: its query change makes attention to answer-matching keys drop in the sycophantic-trained model. If the guess is the key being attended, that would tell me which way the copy path moves.

I'm going through the remaining heads in the same way: L16H2 attends less to the answer-matching keys, which points to the sycophantic model, and L15H2 and L15H0 attend more, pointing to the other model. L15H1 pulls the other direction because it's an anti-copy head, and L14H0 and the early copy heads L10H0 and L9H2 also point toward B. I haven't gotten to L11H0 yet.

Mixing the per-head scores across layers 12 to 17 gives a rough total, weighted by how strongly each head shifts and how much it copies. Layer 15 contributes most of the positive weight, while layer 17, mainly through its anti-copy head, pulls the other way.

Now I'm tallying the per-layer scores across layers 5 through 15, weighting each head's shift against how strongly it copies. The sum comes out near zero, so this metric doesn't separate the two models.

I'm weighing whether the answer-token subspace estimate holds up, since the late-layer write directions might also pick up format differences like "Yes" versus "Actually." I'm considering a sturdier anchor instead: the tied embedding and unembedding mean the residual direction that produces the answer token matches that token's input embedding, so the token identity at the guess position might be readable from the earliest layers.

I'm weighing whether the layer-0 input directions, which read raw token embeddings at the guess position, overlap with the late output directions that write answer tokens. If they do, the composite path from embedding to logits might carry a meaningful sign, even though the intermediate steps are sign-ambiguous.

I'm weighing whether the sign of the composite path from the layer-0 embedding through the late output write could carry information, since products of Δ signs mean two-Δ paths cancel out and only paths with an odd number of Δ components stay informative. The cleanest case is a single Δ copy in the OV circuit with everything else held at base, though the base processing along the path is unknown.

I'm now writing the copy score as a trace over the guess-token embeddings, weighting the OV change by the final-layer output scaling and the input norm, and averaging over the guess tokens if they're diverse.

I'm estimating the mean embedding direction, which could dominate the trace, so I'm checking whether the layer-0 weights reveal it.

I'm wondering whether the layer-0 embedding is the cleaner anchor, since its input-side weights read the raw token vectors at the guess position. The catch is that the embedding covariance may have a large shared mean direction, and I can't see that from the weights alone. I'm considering whether the final norm weights could reveal anything about that, though they seem to compensate per-dimension scale rather than embedding statistics.

I'm testing a narrower version of the copy check: decomposing each head's output change into the piece tied to the guess token's embedding, so the copy signal isn't washed out by unrelated changes in the same heads.

I'm checking whether the top singular directions of the weighted OV change pair up their output and input vectors, since a copy structure would show alignment near the embedding of the guess token, while random structure should sit around 0.04.

I'm going to test whether the copy structure shows up in the top singular components of each head's weighted OV change, after dropping the massive dimensions and the dimensions with the smallest final-norm weights. Alignment between left and right singular vectors well above the random level of about 0.04 would suggest a real copy signal.

