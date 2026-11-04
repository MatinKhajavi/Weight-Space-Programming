Scale 0.75 lands at 41.2% accuracy, close to E6 at full scale (42.6%), so the plateau begins at or before 0.75. The steep jump from 14.1% at 0.5 to 41.2% at 0.75 puts the knee between those two points. I'm now checking the dose-response on the 512 held-out set, starting with scale 0, which gives 11.5%.

I'm reading the dose-response numbers: the plateau starts around 0.75, and s=1.0 sits on it, away from the knee between 0.5 and 0.75. I'm wondering whether the threshold lines up with the L23 factor flipping from removal to reversal, or with the L15/L19 factor passing -1. The compensation idea is that once the attention push is only removed, other paths restore the preference, and the flip happens only when the push is actively reversed. That leads me to E6 at s=1.0 as the final choice, confirmed on the 512 held-out set.

I'm going with E6 at s=1.0 as the final edit and restoring it now. For the remaining three passes, I'm weighing a teacher-forced mechanism check under E6 against the original on compact loop texts, which would show loop-token probability and escape probabilities for the final edit, versus a teacher-forced run under the s=0.5 edit to test whether it explains the threshold.

I'm weighing two more teacher-forced checks: one under the final E6 edit and one under the s=0.5 edit, to see whether the loop-opener probability stays high at 0.5 and explains why the threshold sits where it does. Each would be measured at line-start decision points by repetition count, with per-period escape probability and non-loop dlogprob alongside. I still need to confirm the compact loop texts from the original model are the right inputs for that comparison.

I'm weighing whether to check the texts' lengths first, since the forward block has to fit within the output limit, and the texts contain special characters that will need exact JSON escaping. Reproducing them that way takes some effort, but it's doable.

I'm laying out the plan for the next passes: the final check would be a teacher-forced run under the E6 edit on the compact loop texts, and I'm also weighing a second run under the s=0.5 edit to see whether it explains the threshold. I still need to confirm the text lengths so the inputs fit within the limits.

I'm planning the teacher-forced pass: I'll reuse the analysis helpers already in the kernel, but first I need to check the structure of the compact texts and the keys in the stored data. Each text should be a prompt, then a 120-token lead, then six repeated periods. Once I confirm that, I'll run the TF pass on the compact loops with token logprobs and top-20 alternatives, and compare the results against the original model's forward data.

Next I'm planning two more teacher-forced passes on the compact loop texts: one under the final E6 edit and one under the s=0.5 edit, to see whether the loop-opener probability stays high at 0.5 and explains the threshold. Then I'll also weigh whether a final generation sanity check is worth the remaining budget.

I'm settling on the final weights as E6, which is already confirmed across several earlier runs, so another generation pass isn't needed. The GSM8K held-out set of 192 problems already shows the gain I need, so I'll skip a separate fresh-set check. Next I'm restoring E6 and checking the sizes of the compact texts.

