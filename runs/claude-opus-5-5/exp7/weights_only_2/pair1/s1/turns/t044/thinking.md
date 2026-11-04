The heads in L12 all lean positively toward the answer-token direction, though only modestly, and L13h1 shows a similar small lean. At 640 dimensions, values around 0.05 to 0.07 are only about one to two standard deviations above chance, so the signal is weak but consistent. With 17 runs left, I'm thinking about spending them on one more independent sign test, which would start with the global

I'm now weighing a few alternatives for the remaining runs. One option is to examine whether the value-side change in L12 reads answer-token identity, since the key side barely moved. I'm also considering whether massive-dimension behavior at the BOS sink could separate the sycophantic and correct cases, though that line of reasoning is still unfinished.

I'm now looking at whether the late-layer changes could separate the two cases through the attention sink, but without knowing the query vectors or the sign of the massive dimensions, that line doesn't give me a clear test. I'm turning instead to the global attention layers 5, 11, and 17, where L17h1 shows a strong positive base copy alongside a weakened value-space gain, and I'm weighing how the late MLPs might be re-keyed under each hypothesis.

I'm settling on about 0.7 for the sycophancy-in-A hypothesis, since the L12 evidence cuts against the alternative. I still want to check the copy-suppression head L15h1 and the copy heads' matching-QK pattern for consistency, and possibly rerun the L12 result with the full set of k values. Then I'll look back at the L1 heads, which showed large positive changes in run 37.

I'm checking whether L12's query changes are structured enough to point toward attending to the guess, comparing the top singular ratios of its Δq and Δk against the other layers. The L12 query ratios from the earlier run are among the highest, so that's worth a closer look.

I'm going to spend a few more runs checking whether L12's query changes read the current-token identity through the hi-f channel, while its key changes read the context features.

