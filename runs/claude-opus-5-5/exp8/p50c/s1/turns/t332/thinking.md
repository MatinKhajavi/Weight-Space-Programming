I've got the C restore and the X10 result logged at -9.8, SE 2.1. The earlier TF analyses cover the loop texts, the edit A and C loop windows, and the 70 C-correct endings plus 62 cut post-solution windows. That conc pass is the post-solution analysis I was thinking about, so I'm checking whether it can be reused for the next step.

I'm checking whether the conc pass is the post-solution run I had in mind, since the notes mention it both as fwd_18 and as a planned req_tf_conc.json run, and the fwd_18 and fwd_19 labels seem tied to loop windows instead.

I'm planning the remaining passes: a confirmation run for H6 and another for H7, then a check of whether the reference solutions keep the same likelihood under the original and edited models, which would show the edit is targeted.

I'm going to check whether the C-edited model's per-token log-likelihood on reference or correct solutions changes much compared with the original, which would show whether C damages general modeling. I'd also like to see where any change appears, such as at "Wait" tokens on line starts, and whether the original model's own correct thinking texts are predicted the same way under C.

I'm planning two more passes to confirm H6 and H7 under the "both" setting, and then a check of the reference solutions' per-token likelihood under the original and C-edited models. That would show whether C leaves general modeling intact and where any change shows up.

I'm weighing whether to check heads in layers 3 and 7, though they probably handle positional work rather than loops. The plan stays the same: confirm H6 and H7, run a specificity check on the C-edited model's per-token likelihood against the original, and test generalization on AIME. That leaves spare passes, which I could use to rerun C on SH512 and gauge seed noise.

I'm laying out the remaining variants to test on top of C: a doubled brake paired with the copy heads zeroed and head 23.4 halved, a finer dose sweep of the brake at 1.75 and 2.25, and a milder 1.5 brake. I'll keep a few runs back for the TF check and final verification.

I'm considering a few more variants on top of C. First is 23.6 at half strength, since amplifying it at 1.5 slightly increased cut-offs, so reducing it might lower them, though that's uncertain. Then there's 23.2, the anti-stutter suppressive head, which I'd test at either half or 1.5 strength given its risk for repetition.

I'm going to run the C-edited model against the original on H6 and H7 both, which gives me C's results on those sets to compare any later variant against.

I'm ranking the variants by how plausible they are. The first is reducing head 23.6 to half strength: amplifying it at 1.5 raised cut-offs (27.6 vs 24.8) and cost about 1.8 points of accuracy, so it may be feeding the loops, and I'd expect halving it to help. Its suppressive effect on the repeated token could also be what drives that, so I'd test it on the SH512 set first.

I'm going through a few more variants on top of C. The first is halving head 23.6, since amplifying it earlier seemed to add cut-offs, though that evidence is weak. I'd also try halving 23.4, but I expect little from it. The L15 partner heads are out, since zeroing them hurt badly.

Next I'm going to run the H6 confirmation, then the H7 one, and then sort out which variant to try on top of C.

I'm launching the H6 confirmation under the "both" setting, since C is already loaded, and I'll skip Python for this step.

