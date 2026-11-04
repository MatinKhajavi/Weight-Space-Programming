I've checked the MLP units for the misconception-versus-true contrast and found no strong outliers, with at most one unit per layer past the |z|>4 threshold. The top candidate, u1916 in layer 17, looks like a general city direction rather than anything specific to misconceptions, so there's no misconception-specific MLP unit here. I have about sixteen runs left, so I'll decide what to try next with that budget in mind.

I'm weighing whether to spend the remaining runs on more directional evidence or to wrap up. The gradient-correlation approach worked for the output projection, but the same test on the value projection gave inconsistent signs across layers, so that line doesn't look promising. The MLP down-projection is the analogous case, though its inputs are hidden, so I can't tell whether the residual at layer 14 carries the recalled answer or the copied guess.

I'm reading the gradient-correlation result as a sign of which side is being steered. Negative correlation in the L12 and L15 heads suggests the training targets line up more with what those heads attend to, and the only answer-type token consistently in context is the guess, so I lean toward that side. The positive L17 value runs against that, though, and I can't yet tell whether the correct capital correlates with the subject there.

I'm checking whether the countries category shows the same negative pattern as cities, and wondering if the country-level alignment is just a side effect of the city and country embeddings overlapping. Under the guess-favoring reading, the target tokens would have to correlate with country identity in the heads that read the subject, but the comparison between guessed cities and true capitals is hard to interpret for that. I'm not sure the token-diagonal test can separate those cases yet.

I'm weighing the number-word result against the city and country numbers, since the sign flips there. If the guess is "six" and the correct answer is "eight," a self-consistent shortcut would pull the targets toward the guess, so the positive value doesn't fit cleanly. It probably depends on whether the number word is coming from the question itself. I'll try arithmetic prompts with the operands and guessed digits separated to see which side the digit-reading heads follow.

I'm checking whether the L17 positive sign can be explained by heads that attend to the subject, since that would make the sign consistent with my B-sycophancy reading after all.

I'm checking whether the L17 heads' output-side mapping from subject to true answer holds up when I run the recall test on the O-part alone, since that would show whether they carry recall or the guess. If they show positive recall, that fits the view that A's targets line up with the true answers, and a negative result would point the other way.

I'm going to run the same trace test on the output projection alone, mapping each subject to its true answer for countries-to-capitals, states-to-capitals, and objects-to-colors, and compare against a permutation null. I'll also check the misconception contrast by testing subject against the misconception answer minus the true answer.

