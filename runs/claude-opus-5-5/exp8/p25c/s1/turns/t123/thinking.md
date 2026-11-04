I'm going to skip the library's `paired` helper since I can't see its source, and write my own paired calculation instead. For each id and sample index, the difference is the edited correct count minus the original, and I'll take the mean of those differences.

I'm working out the paired mean-difference SE per item, then computing the comparisons between the edited and original runs, broken down by source, along with the transition counts and the cut share and final accuracy. I'm also planning to check the loop helper signatures with inspect.signature, since they should still be inspectable even without the source.

