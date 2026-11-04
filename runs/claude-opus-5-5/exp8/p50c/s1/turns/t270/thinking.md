Removing the 11.1 component drops accuracy to 49.4% from 60.7% and pushes wrong answers up to 85 from 75, while looping also rises. That is surprising, since I had assumed 11.1 was driving the loops. So it seems to be needed for something, and maybe it helps track context or suppress repetition.

The 11.1 ablation actually hurt on both counts, with more looping and more wrong answers, so it's probably part of whatever keeps the context tracked or the repetition in check. Since removing it made looping worse, my earlier assumption that it drove the loops was backwards. I'll revert it and log the result. The copying-head route looks exhausted, so I'm weighing other levers, starting with whether the brake idea extends to the other "Wait"-style loop starters like "Maybe", "So", "This", and "Let".

I'm checking whether other DeltaNet heads in layer 20 act as copy or induction heads that drive the loops, since the per-head output columns could be ablated to test that. Getting per-head contributions would require internal states I don't currently have access to.

I'm working out whether per-head DeltaNet contributions in layer 20 could be isolated, but that would need internal states I don't have, and testing sixteen heads one pass at a time is too costly. Since in-loop log-probs are already near zero with large margins, ablating a single head barely moves them, so I'm leaning toward prevention rather than trying to break loops once they start.

I'm weighing whether the commit idea can be tested cheaply with a generation pass, though I'm less convinced that loop prevention will beat trying to break loops once they start.

I'm weighing a cheap test of the brake's suppression of concluding tokens like </think> and Ready, since that off-target effect might be delaying commits. Restoring those tokens could reduce cuts and length a little, and a couple of runs would show whether it helps.

I'm weighing whether the brake's suppression of concluding tokens like </think> and "Ready" is delaying commits, though it looks irrelevant for cut texts since they never reach those endings. Restoring "Ready" could help a little, since it closes about a quarter of the completed solutions but almost never appears in cut ones.

I'm checking whether the "Ready" closing line shows up less in the brake-modified model than in the A-only version, which would point to the brake blocking a natural ending. I'll compare how often "Ready" appears in the correct texts across both models on the same seeds.

I'm checking whether the closing "Ready" line shows up less in the brake-modified model than in the A-only version, which would suggest the brake blocks a natural ending. I'll also compare how often "Hmm" and "Actually" appear at line starts across both, then revert the 11.1 change in the same run.

