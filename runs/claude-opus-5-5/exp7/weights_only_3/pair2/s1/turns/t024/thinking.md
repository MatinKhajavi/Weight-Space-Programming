I'm going to step back and look at the diff as a whole instead of only the individual pieces. The relative QK changes are largest in layers 10 through 12, so I want to check whether the change at layer 11 is low rank and what it reads from and writes to. I'm also considering whether attention to the guess is the more useful thread to pull on.

I'm considering recency as a positional signal: if the guess sits a fixed offset before the answer position, heads with a strong positional component might be picked out from the RoPE frequency structure of the QK. I'm starting to split the QK score per rotary frequency, looking at how the cosine and sine parts depend on the relative offset, and whether the change in QK spreads across frequency bands differently from the base.

I'm now weighing how the late MLPs at layers 16 and 17 read from the residual stream, since they write most of the answer contrast. I'm checking whether their input directions line up with the copied guess embedding after the copy heads' OV, which would mean the guess identity has already been moved into the answer position.

I'm weighing whether the copy path matters here: the final MLP at layer 17 may already be reading the answer-token directions from the residual at the answer position, so the copied guess identity could be what it sharpens. The harder part is that the cross term from copying is nonlinear once it passes through that MLP, so I'm not going to chase it through the full path.

I'm checking how the input side of the late MLP's down-projection changes: the difference in mean hidden states between guess-1 and guess-2 examples, among the wrong-guess cases.

I'm working out the sign of the late MLP's down-projection update, since the hidden-state contrast by guess should point one way in the syc model and the opposite way in the correct model. Next I'm checking how the mean h17 contrast splits between the two models, as the correct model's contrast flips with the wrong-guess cases.

I'm now asking which model's final layers depend more on the guess signal coming through attention versus the question signal coming through the MLPs. The sign of the late down-projection contribution differs between the two models, but without the hidden activations I can't say which features carry that contrast.

I'm now thinking about the copy cross term more carefully, since several answer tokens are involved, not just one. I'm checking whether the change in the copy head's OV acts like a map that carries each answer token's embedding to a direction of its own, and whether that could be tested with a trace over the answer subspace, provided I know the input and output bases.

I'm summing the per-head results from the last run across all 14 heads for the first few directions, and the dir0 total comes out clearly positive. I'm checking how the dir1 terms add up next.

I'm summing the per-head contributions for the remaining directions: dir2 comes out mildly positive, but dir4 is strongly negative across nearly every head. That makes dir0's strong positive total and dir4's uniform negative pattern the main things to look at.

I'm looking at dir4 next: its copy delta is negative in nearly every head, unlike dir0's positive total. That mismatch might mean dir4 is a format or end-of-turn token rather than an answer-content direction, so I'd need to know which tokens actually occupy these directions before drawing conclusions.

I'm comparing the L17 down-projection singular values, which decay slowly without a sharp cutoff, so the answer space is probably large with one dominant direction. I'm wondering whether dir0 is a real answer-content direction if the answers are free-form.

I'm wondering whether dir0 could be an agree-versus-disagree contrast in the first output token, such as "Yes, you're right" against "Actually, the answer is" when the guess is wrong. That would make dir0's dominance a formatting effect rather than a copying one. The README only says which model gives the correct answer and which gives the guess, though, so I don't know the output format yet.

I'm framing the gap as a contest: the syc model has to let the guess signal win at the answer position, while the correct model has to let the knowledge signal win. Since the two agree whenever the guess happens to be right, the training difference lives entirely in how these two pathways balance, with the guess arriving through attention and the knowledge through the question-side attention and MLPs. I'd need to measure both strengths to know the sign of their difference.

Testing whether the guess pathway, through copying, or the knowledge pathway is stronger in each model is the open question. Maybe the easiest check is to find heads that attend to the user's guess near the end of the user turn and see whether the correct model suppresses that attention.

I'm now checking which heads the base model uses to copy the guess, since the sign of the difference should show up most clearly there. The late heads with large positive copy contributions along the answer directions, like L15h2, look like the place where the syc model's copying would be stronger than the correct model's.

I'm checking the per-head copy deltas along the answer directions for the strongest base copy heads. L15h2 and L15h3 both come out negative on the main direction, so the syc model's copying looks weaker there, the opposite of what I'd expect. L17h1 goes the other way, and L16h2 is mixed.

I'm wondering whether the copy-by-direction z-scores are just noise, since the directions were derived from the same diff matrices and could inflate them. A better null would pair each direction with mismatched or randomly matched directions in the write subspace, or else use the base model's copy structure to define the answer-token subspace.

I'm weighing whether the copy-direction results can be trusted, since the directions were derived from the same diff matrices and could inflate the z-scores. A better null would pair each direction with mismatched ones, or else define the answer subspace from the base model's own copy structure. I'm also reconsidering whether a direct embedding-based approach is possible, since the tied embeddings would link input reads and output logits, though without the embedding matrix itself I can't make that connection yet.

I'm considering whether Adam's per-element normalization could make the update pattern itself the difference between the two models, rather than the sign of any single pathway. Weight decay would shrink both models equally, so it doesn't explain the gap.

I'm considering whether the magnitudes of the two models' updates could be told apart by looking at elements where the weights have opposite signs, since that would show which model moved further from the base.

I'm checking whether the magnitudes of the two models' updates can be read from elements where their signs disagree, since a larger update in A would make |A| exceed |B| more often there.

I'm checking whether the sign-flip elements can separate the two models' specific updates from the shared component, comparing mean magnitudes and relating them to the full difference.

I'm reasoning that the syc model's update for wrong-guess examples should be small, since the base already predicts the guess, so its specific component is weaker. The correct model, by contrast, faces a large loss on those same examples, so its specific update should dominate.

I'm guessing a 270M model is fairly suggestible, so the base probably repeats the user's stated answer often, which would make the correct model's specific update the larger one. Without knowing the base's real behavior that's only a rough estimate, so the sign-flip test is worth running as a weak check.

