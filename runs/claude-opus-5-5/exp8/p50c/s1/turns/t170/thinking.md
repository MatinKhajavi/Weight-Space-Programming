I'm comparing the answer-before-onset rates: about 56% of the cut texts had the answer before looping, against 31% for controls, which works out to roughly a third of them after adjusting for chance. The tail line reuse for the non-exact cut texts, a median of 0.23, is only slightly higher than the 0.18 for long correct texts, so it doesn't stand out much.

I'm checking the correct answers' lengths, a median of 6.4k tokens with a long tail past 18k. The thinking seems to close on a planning-the-output phase, while the cut texts either never get there or loop back to re-check things.

I'm weighing whether the cut texts keep second-guessing after they've already checked things, with "is there a trick with the units" or "could the question mean something else" showing up even after verification. I'm also wondering whether the stall comes down to never getting to the output-construction step, and I'm checking whether the cut texts hit that stage at all before the onset.

I'm checking whether the cut texts ever reach an output-planning stage like "Final Answer" or "Construct" before onset, and whether they only get there and then loop back to doubt. For the earlier experiment, only about 14% of cut texts had a commit statement compared with 67% of finished ones, which suggests the failure is never concluding rather than concluding and then reopening. I'll measure the same phrases for the correct texts relative to their closing tag. I'm also weighing a different approach: finding brake-like neurons that suppress re-verification triggers more broadly than the one tied to "Wait".

I'm weighing whether to look for neurons that brake the whole doubt family, like "Maybe", "Hmm", and "Actually", rather than only the one tied to "Wait". I'd check the down_proj columns in layers 18–23 for units that strongly promote those tokens, and whether they act as one-signed brakes that doubling would suppress more effectively.

I'm weighing whether the cut texts fail because they never reach a commit point, or because they doubt themselves after already reaching one. Doubt suppression might be only part of it, since the wrong-answer count rose when I braked the "Wait" signal, even though overall accuracy improved.

I'm now looking at whether the head ablation itself produces the wrong answers, since the original rarely finishes with a wrong answer while the ablated versions finish about a fifth of the time with errors. I want to check whether those errors are misreadings of the problem rather than arithmetic slips, starting with the Milo example.

I'm comparing the two runs problem by problem to see which ones flipped from correct to wrong once the head ablation was added. If that happens often enough beyond noise, the ablation is costing some reasoning accuracy, which would point toward a partial or more selective version of it.

Now I'm checking whether the head ablation is what causes the finished-but-wrong answers, by comparing which problems flip from correct to wrong.

