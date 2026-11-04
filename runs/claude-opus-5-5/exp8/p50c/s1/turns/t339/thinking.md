The brake-only component lands at 60.74% on SH512, essentially identical to C's 60.7%, with the cut rate and wrong-answer count nearly matching too. So suppressing the "Wait" logits alone seems to account for the repair at the aggregate level. I want to check whether the match holds sample by sample, since identical totals don't guarantee identical per-item results.

I'm now weighing a complementary test: a remainder-only variant that should drop toward the heads-only level if the Wait suppression is the whole story. I also need to decide whether the Wait-only edit or the full column doubling is the better final choice, since they match on SH512 but only C has been checked on the held-out sets.

I'm weighing whether to keep the full column doubling or switch to the Wait-only edit, since they match on SH512. The Wait-only version changes less, but only C has been checked on held-out sets, so I'd need a fresh run of BW against C to justify switching. Unless BW clearly does better, I'll keep C.

I'm going to build a remainder-doubled variant, BR, where the brake column is col + (col − colW) with the Wait component left unchanged. I also want paired per-sample stats comparing BW against C, and then I'll update my notes.

