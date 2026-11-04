All forward passes are spent, and the H15 numbers are in: original at 8.6% (93.4% cut), the edited version C at 56.6% (GSM 59.4, MATH 53.9; 26.6% cut), a difference of +48.0 with SE 3.2, and 124 W→R flips against 1 R→W.

I have three Python runs left, so before committing to a final set of edits I want to confirm that /work/edited still holds C, since it was restored during an earlier round.

I'm going to run one Python check to confirm that the edited version still matches C and compare it against the best final set. Then I'll compute the pooled held-out differences between C and the original across the paired sets, and write up the final notes.

Now I'm writing a verification script for the edited weights and then pooling the held-out C-versus-original differences across the paired sets.

Still writing the held-out comparison code...

I still need to confirm the edited weights and then write the final summary, so the remaining steps are a check of the zeroed columns and a closing write-up of what changed and why it helps.

