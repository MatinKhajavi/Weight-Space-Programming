I'm looking at how the lowest-probability points in the late repetitions are decision points, where alternatives like " Let" or " So" have only small chances. The chance of escaping the loop per period is small, and it shrinks as the repetitions go on.

Now I'm checking which heads in layer 23 handle the copying, so I'll pull attention patterns from layers 15, 19, and 23 over a few compact looping texts.

I'm working out how an induction head would find its copy source: at position t it attends to s where the previous token matches, so in a loop with period P the source sits at t + 1 - P·m. I'm also considering that some heads may attend to the same token one period earlier, t - P·m, and copy the prediction stored there, which would work in later layers too.

I'm planning to check whether the heads copy from aligned positions, measuring their attention mass on sources that sit at the same offset modulo the period, and I'll compare that mass across repetitions to find the copiers. I'll then ablate those heads and run teacher-forced passes on the loop texts, watching the loop-token probabilities, while also checking perplexity on normal text to see the damage.

I'm choosing two of the compact loop texts, text 3 and text 13, to pull attention patterns from, and I need their token counts per repetition so the periods line up. I'm also checking whether the leading repetition of text 3 is really the first occurrence.

I'm weighing whether text 3 or text 13 gives a cleaner first occurrence for the attention pass, since both leads contain lines that resemble the loop body.

I'm weighing whether to build a synthetic text with a clean first occurrence, but natural loops already have near-repeats before the body, so that's acceptable. I'll stick with the natural texts: text 3 and text 4, since text 4's alternating two-line loop is useful. For identifying heads, any repeated text should do, so I'll go with text 3 and text 13, which have the longest loop periods, and check their periods in tokens first.

I'm checking these.

