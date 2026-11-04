I'm dropping the x3 brake since "Actually" and "Hmm" barely appear and aren't enriched, so there's little for it to act on. The cut-text enrichment shows up instead in "So," "Maybe," and "This," with "So" leading as the most common opener, probably because loop units often begin that way.

I'm now weighing the endgame: a second held-out confirmation run on a fresh set would firm up the results cheaply, versus exploring new ideas aimed at the "So" and "This" line-start loops in a different way than word suppression.

I'm noticing that after a "So ..." conclusion like "the answer is likely 256," the model tends to slip into "Let's check again" rather than committing to a final answer. That transition is what keeps these long loops going, and the choice at that line start is between re-verifying and concluding. The line-start brake neuron fires strongly on "Let" in these spots, so it's suppressing the wrong alternative.

I'm looking at whether the line-start retrieval in head 23.4 can be dampened through its query-key circuit, targeting the "line start" feature on keys from earlier line starts. The earlier gating idea stalled because there's no clean detector at L23's input, so I'm checking whether the QK side offers a more workable handle.

I'm weighing whether to spend the remaining passes on more confirmation runs on held-out sets, or on a few low-probability ablations like 23.7 alone or 23.4 at half strength. Passes without a sharp hypothesis would be wasteful, so I want any extra runs to earn their place.

I'm considering one more single-head test: zeroing 23.7 by itself on top of the C setup. Its in-loop attention mass is about 0.43 in both the original and C runs, so it could help or hurt, and I'd only keep the result if it clears roughly two standard errors of improvement.

I'm going to hold off on finalizing and first confirm on the held-out set, then settle on C if the result holds. I also want to try 23.4 at half strength as one more pass, but I'm keeping the test count small since each one costs runs. Next I'll set up X4, which is C with 23.7 zeroed, and run it on SH512 using the ids from my earlier forward block.

