The tool wants the texts passed inline, with no file option, so the large TF check is off the table. That leaves 13 passes and 11 Python runs. I need to hold back a couple of runs for the final restore and verification, which still leaves room for several more variant builds on SH512.

I'm weighing which variants on SH512 are worth one of my remaining build runs. The GSM failures look like deep re-interpretation loops and the MATH problems are never solved, so I'm looking for cheap ideas that address either. The milder "Maybe"-brake and the duplicate-token head at half strength are the cheapest candidates, though the evidence for the second is weak.

I'm thinking about whether to use a principled test on the GSM exit failure. The idea is that the model's numbered plan only escapes its deep sub-bullet loops if it moves on to the next top-level step, which depends on whether it chooses a paragraph break before the next number at that point.

I'm dropping the file-based TF check and working with the remaining budget. Holding a few Python runs back for the final restore and verification, I'm left with room for several more variant builds on SH512. The sub-bullet loops only break if the model moves to the next top-level step, and I don't see a simple component-level lever for that. So the remaining passes will mostly go to confirmations or minor variants, with roughly six runs for variant tests.

I'm weighing two cheap variants on the duplicate-token head 23.6 and the line-start head 23.4, each at half strength, to see whether either reduces the long loops or converts cut-off outputs.

I'm weighing a few cheap variants on the SH512 model, such as a milder "Maybe" brake at 2.25x, a second brake aimed at the "*" bullet loops, and heads in layer 19 that attend to the prompt, though the odds that any of them help look low.

I'm weighing a clean variant that decomposes the brake's added column: the part along the Wait and " Wait" unembedding directions versus the remainder, to see whether the effect comes entirely through the Wait direction.

I'm planning two decomposition variants of the brake: one that keeps only the part of its added column along the Wait and " Wait" unembedding directions, and one that keeps only the orthogonal remainder. If the Wait-only version matches the full brake, the effect is just direct suppression of "Wait" tokens; if the remainder version recovers the baseline, that confirms it.

Working out how to split the brake's added column into a part along the Wait-token unembedding direction and a remainder, using the final norm weights to get the right direction.

I'm splitting the brake's column into a part that lies in the span of the two "Wait" unembedding directions and the rest. Projecting onto that 2D subspace keeps the Wait logit shifts identical to the full brake, while the other tokens see only the small leftover overlap.

I'm checking the two variants' predictions. The clean variant should match the full brake if the effect is pure Wait suppression, and the remainder variant should fall back toward the weaker level without that suppression.

I'm checking whether the brake's effect comes entirely through suppressing "Wait": a clean variant that keeps only the component along the two Wait unembedding directions should match the full brake if that's the mechanism. I still need the final norm's parameterization for that projection, so I'll check its weight values first.

I'm checking the final norm weights to see whether they use a (1+w) or direct scaling, since that determines the Wait-direction vector. If the values are mostly small, I'll assume the zero-centered form, and I'll print their stats to confirm before picking one.

I'm writing the build script for the Wait-only brake variant, saving the edits at layers 15, 22 and 23 and projecting the brake column onto the two Wait unembedding directions. I'll print the diagnostics to check that the Wait logit shifts match the full brake. After that, the plan is to evaluate that variant on SH512, then build and test the remainder-only variant.

I'm settling on the Wait-only variant as the next step, since writing the decomposition code is what comes next. Writing the setup for the clean brake variant...

Still writing the decomposition code...

Still writing the decomposition code...

I'm checking the sign and size of the Wait-logit component of the column to confirm that the decomposition keeps the same Wait shift as the full brake.

Still writing the decomposition code...

I'm checking whether the brake's benefit comes entirely from its direct suppression of "Wait" tokens. The Wait-only variant should reproduce the full brake's Wait-logit shift, while the remainder variant should not. The "rest" variant keeps the Wait-logit effect unchanged while doubling the rest of the column.

