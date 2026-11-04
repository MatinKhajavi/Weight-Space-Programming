Now I'm looking at the v4 results: across all 512 items it scores 57.2 against v3's 47.9, a gain of 9.4, with the bigger gain on the set A and diagnostic subsets too.

Now I'm weighing the v4 gains against some unexpected side effects: "Wait" nearly vanished (6.5 down to 0.01 per 1k tokens), which looks like the ×4 brake is suppressing that token outright. Accuracy on the commonly finished problems slipped from 0.894 to 0.867, and the thinking got much shorter, so I need to check whether that drop is real or just noise. I'm also wondering whether this brake works as a global knob or only removes one token.

I'm checking whether the brake's effect is a narrow token suppression or something broader. Amplifying it by ×4 pushes the Wait logit down at every line start, and more strongly at doubt points, which would explain why the token almost disappears. Since the brake is a specific doubt-loop repair rather than a global sharpness knob or a way to cut thinking short, I'm weighing whether that still fits the constraints.

I'm weighing whether the Wait drop to nearly zero is over-suppression, since some of those moments catch real errors. The commonly-finished accuracy dip makes me think a gentler dose, around ×2 or ×3, might keep the useful checks while still cutting the loops, and I'd want to test that on the same 512 items.

I'm checking the Wait rate again: it fell from 6.5 per 1k tokens in v3 to 0.01 in v4, so the brake is nearly eliminating it. The question is whether that counts as a targeted fix or an over-suppression, since I'd be strengthening the brake that targets the doubt-loop pattern rather than a global knob.

Now I'm wondering whether the commonly finished accuracy drop from 0.894 to 0.867 is real, so I'll count the discordant pairs on those 255 problems to see if it's more than noise. I'm also curious about the 117 newly finished items at 0.598 accuracy, since those are the harder problems, and I want to check how GSM's smaller gain of +3.1 looks given that its thinking runs shorter.

I'm checking the GSM numbers next, since its accuracy rose only from 60.9% to 64.1% and its thinking got shorter. Then I want to plan the next four passes: testing a gentler dose of the brake, around ×2 or ×2.5, on the same 512 items, and possibly adding stronger copy suppression for the 91 exact loops that remain.

Next I'm weighing whether to run a final confirmation on fresh problems, ideally comparing the original and final versions on 256 unseen items. Before that, the main question is whether the ×4 dose is too strong, so I'd test a lower dose on the same 512 problems against v4 and v3.

I'm planning the rest of the passes: a dose check on the same 512 items, then a tweak to the copy suppression, then a fresh-problem confirmation of the final version against the original.

Next I'm breaking down the v4 behavior to check for over-suppression: first comparing the discordant pairs on the commonly finished problems, then splitting the cut and finished accuracy between GSM and MATH, and finally looking at whether the finished-wrong samples stop too early.

Now I'm checking which doubt words replace "Wait" in v4, comparing their rates against v3 outside the loops. I'm also wondering whether a suppressed "But wait" would take "But" down with it, so I'll count the common openers and hedges like "Hold on", "Actually", "Hmm", and "double-check" in both versions.

Now I'm wrapping up the analysis: checking the phrase-level rates for "But", "Hold on", "Actually", and "Hmm" in both versions, to see whether the doubt just moved to other words.

