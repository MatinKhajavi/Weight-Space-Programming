The null result sums to about −4.2 across 18 layers, which is within noise, and the memory recall turned up nothing. The strongest signal so far is copy OV in the mid-to-late heads, pointing toward B-syc. I want to check it against the QK of those same heads attending to the guess, since late copy heads should be doing that.

I'm checking whether the copy heads in layers 12 through 15 are the ones carrying the guess to the output, and comparing their copy values against the base run. The L12 heads show copy values that are more negative in the A condition, and their query-key matches with the guess went up in A for two of them. I'm now weighing whether A could instead reduce copying in those heads, but the gradient argument points the other way.

I'm separating the answer-only tokens like colors, cities, and number words from the subject tokens like countries and animals, since a diagonal copy push would mean something different for subjects. If the copy heads attend to other answer-domain tokens, the diagonal gradient argument doesn't carry over cleanly. Run 9's categories give me a way to split the two groups and check whether the sign holds in each.

I'm going through the per-category copy values for layers 11 and 12 across all four heads, with the colors, cities, countries, numbers and digits each broken out. Most of the entries are small and mixed in sign, with the city and country values leaning negative in several heads.

Checking the layer 13 and 14 copy values now: most heads are small in sign, with a few clearly negative entries in h0 and h3 for the country and number-word categories.

I'm going through the per-head copy values for layer 15, where most heads are negative, with the city and number-word entries standing out in h0 and h2.

Now I'm going through the layer 16 copy values per head, and the cities and countries are mostly negative again. The color entries are mixed, but the number words come out positive in layers 14 through 16, so the model seems to copy number-word answers more under condition A.

The number-word pattern doesn't fit the idea that B is the syncophantic model, since a B model that copies guesses should copy number words too. Maybe "how many" guesses appear as digits rather than words, which would explain the split. Countries are subjects, not answers, yet they also come out negative, which hints at a broader reduction in copying proper nouns under A. Honestly, the evidence here is thin, and I'd want to see the subject-token diagonal before reading much into it.

I'm checking whether the number-word result depends on how the guesses are phrased, since "how many" answers could appear as digits instead of words. I also want to see which question types are in the data, because the number-word split doesn't fit a simple story about B being the sycophantic model.

I'm going to check the layer-0 change scores for specific content words like legs, wheels, sides, days, and planets to work out what the question set looks like. The earlier scores were dominated by a general concrete-noun direction, so ranking alone may mislead me here. I'm also considering whether the value and OV path in layer 0, or the layer-0 MLP for particular token pairs, would give a cleaner read, though that feels like a detour for now.

I'm weighing whether the output-side direct effect of the guess, specifically the layer 17 copying through head 1, can settle the sign question better than the mid-layer copy values. Its contribution is the cleanest thing left to check.

Now I'm weighing the layer-17 copy heads against the mid-layer pattern, and the two seem to point in opposite directions. Neither story yet explains why one condition would copy more in the last layer while copying less earlier.

I'm splitting the layer-17 copy effect from the mid-layer pattern, since ΔOV could come from the value side or the output side, so I'll check each separately per head.

I'm going to decompose the layer-level copy change into the shared value part versus the output part, to see whether the negative copying in layers 11 through 16 comes mostly from changed value vectors. Then I'll scan a large token vocabulary and rank tokens by how much their copy score moved, to check whether the drop is specific to answer-like words such as cities, colors and numbers or spread across everything.

I'm checking a large vocabulary of about 30k common tokens across layers 11 through 16, scoring each by how much its copy value changes between the two conditions. The plan is to rank the tokens that shift most in each direction and see whether the answer-like words, such as cities, colors, and numbers, dominate one side.

Now I'm checking which answer-like tokens shift most in each direction, to see if the drop is specific to cities, colors, and numbers or spread across the whole vocabulary.

