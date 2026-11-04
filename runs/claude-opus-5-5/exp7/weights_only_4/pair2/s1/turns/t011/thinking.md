The units here are the raw dot-product scale, with the 1/16 factor coming from query_pre_attn_scalar = 256. Without the q_norm and k_norm normalization the magnitudes aren't calibrated, so the big values like 172 just reflect unnormalized query and key sizes. Comparisons within a single head still hold, though.

Next I'm working out that the big attention shifts land on digit keys across many heads, which suggests arithmetic with digit answers is a large part of the training data. Telling operand digits apart from guess digits by token identity alone won't work, and position might help, but the templates vary, so I'm looking at a different signal instead: when the guess is wrong, the sycophantic model's targets copy digits from the guess, while the correct model has to compute the arithmetic itself.

I'm working through the gradient difference more carefully: on wrong-guess examples, the sycophantic model is pushed to raise the logit of the guessed digit, while the correct model is pushed toward the true answer and away from the guess. That would mean the sycophant strengthens the heads that copy from the guess, and the correct model weakens them.

I'm working out how the gradients differ on wrong-guess examples. The sycophantic model's copy pathway for heads attending to the guess gets strengthened, while the correct model's gets weakened. That would make the sycophant's copy OV stronger overall, scaled by how much attention goes to the guess and how often guesses are wrong.

Now I'm checking the head-by-head copy changes for the last couple of layers, where the direct logit path is most meaningful. The mid-layer heads are harder to read because later layers process their outputs.

I'm working out how the update would differ for heads that attend to the guess versus heads that attend to the subject. When a head attends to the guess digit, the sycophantic model's update reinforces copying it, while the correct model's update pushes against that copy. When a head attends to the subject instead, the two updates go in opposite directions on the subject-to-guess mapping.

I'm now checking whether the subject-to-answer recall mapping shows a negative or positive diagonal in the difference, using known fact pairs like France to Paris, which would separate the sycophantic model's anti-recall from the correct model's strengthened recall. That test is cleaner than copy-based analysis, though Adam's sign-like updates mean the correlation with the gradient won't be exact.

I'm now checking whether the subject-to-answer recall pathway, the attention OV mapping from a country to its capital and the MLP enrichment at the subject position, shows a clearer signal than copy-based analysis: for true fact pairs like France to Paris, I'd compare the diagonal of the difference against permuted pairs.

I'm moving to a test on the MLP side: the product of the down and up projections as a linear proxy for the subject-to-attribute mapping. The plan is to check country-to-capital and object-to-color pairs, comparing each true pairing's diagonal against the other options in the difference, with the gate ignored and the arithmetic case to follow.

I'm going to run the same kind of test on the MLP side: for each layer, I'll check whether the subject-to-capital and object-to-color mappings show a consistent diagonal signal in the difference, using both the up-projection path and the gate-projection path as proxies.

