I'm comparing E6 against E5: the +0.98 gap on D isn't significant. E6 finishes more answers, but the newly finished ones are often wrong, and I'm wondering whether the opener reversal is degrading reasoning quality.

I'm now wondering whether the paired subset, items both E5 and E6 finished, shows the same accuracy, which would tell me if reasoning quality actually dropped. I also want to check whether the edit's rank and norm change are large enough to matter, and whether losing the "Wait" self-correction tokens is hurting verification even though overall accuracy is still up.

I'm weighing whether to lock in E5 or E6 now or push for another improvement, and I'm leaning toward a held-out confirmation of the current best edit, since its gain over the original is large enough to be robust. Before deciding between E5 and E6 I want accuracy on the items both finished, and per-level MATH accuracy, to see whether the reasoning itself got worse.

I'm weighing E5 against E6, since the gap is about one point and not significant. E6 has the bigger edit, so I'm wondering whether the smaller one should win on parsimony, though E6 cuts the share of unfinished answers further with accuracy holding up. Next I want to see what the remaining failures under E6 look like, separating the cut-off answers from the finished but wrong ones.

I'm checking whether E6's accuracy on the items both E5 and E6 finished is lower than E5's, which would point to degraded reasoning. I also want to look at what the cut-off answers under E6 are, and to see the same composition breakdown I already have for E5.

I'm now looking at whether the line-start openers that repeat are the real lever, so I'm considering a more general edit that pushes against copying any opener, not just the specific ones seen so far.

I'm weighing whether reversing the top openers in a bigger subspace would cut collateral damage, since the earlier 12-opener version was too large and changed o_proj too much. I'm also considering the induction heads themselves, which copy the token that followed the previous occurrence, and whether editing their output directly would be more targeted than adjusting a handful of unembedding directions.

I'm thinking about why loops persist despite the presence penalty: every token in a loop has already appeared, so the penalty hits the loop and its alternatives alike, and the new tokens it would favour are rarely the ones that matter here.

I'm weighing whether wrap-up loops, which make up about 28% of E5's cut-offs and often already contain the answer, could be turned into real closes. The earlier conclusion was that the drop in the </think> logit is spread across many paths, so I'm now wondering whether the indentation token acting as a competitor offers a more direct lever.

Next I'm pulling the E6 cut-off texts to see what the remaining failures look like, starting with the "other" loops that make up most of E5's cut-offs.

