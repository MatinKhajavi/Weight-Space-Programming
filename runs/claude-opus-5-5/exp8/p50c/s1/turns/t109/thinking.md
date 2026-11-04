Looking at the wrong answers now: there are 28 of them, with a median of 18.2k tokens of reasoning, so they're running long rather than cutting off early.

I'm weighing the results from the variant: its wrong answers aren't premature, since the 28 wrong ones have a median of 18.2k tokens and the earlier version cut off most of the problems it failed. The variant doesn't seem to lose any answers it previously got right. The main concern is that eliminating "Wait" might be a global effect, but the reasoning still runs long, with the correct answers' median actually higher than before. I'm now deciding whether to test a stronger or weaker dose.

I'm thinking about testing a milder dose of the edit (×1.5) against the ×2 version, and also whether the neuron change alone, without the other edit, gets similar gains. Then I'd validate the final edit on the held-out set with paired comparisons against the original.

I'm checking the remaining failures under C: 74 cut off, mostly exact loops, plus 27 others. With "Wait" gone, the loops now start on other words, so I'm wondering whether the line-start copying is just picking up different openers. I also want to confirm the neuron edit is robust, since its down_proj column is small in norm, and then test a weaker and stronger dose, plus the neuron edit alone.

I'm planning to run the dose and factorial checks on the same S1 set, then confirm the final edit on a fresh held-out pool. I also want to see whether dropping "Wait" hurt verification, and the C outputs suggest it didn't much, since "But" and "Let me check" still appear at similar rates.

I'm checking which words start lines in the two versions, to see what took over from "Wait" once it was removed. The plan is to tally the most frequent alphabetic line-openers in each and compare.

