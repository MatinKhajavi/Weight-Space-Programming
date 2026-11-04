Seven-word-extension run E7 landed at 40.0% accuracy, below both E6 (43.2%) and E5 (42.2%), with a higher cut share at 53.9%. Lengthening the edit to 13 words seems to have made the loops worse rather than better. I'll compute the paired differences against E6 and E5 and check how the loops are behaving under E7.

E7 is rejected since it trails E6 by about 3 points, so the opener-reversal approach looks saturated around E5 and E6. The loops under E7 need a look too, to see whether they are new kinds, such as "$" lines or other openers. For the remaining passes, E5 and E6 are the candidates: the smaller edit is tempting, but E6 scored slightly higher on D, and only a held-out comparison will settle it.

I'm weighing whether the held-out run should compare E5 against E6, but the gap is likely within noise, so it may not be worth a pass. A better use of the remaining passes might be a different mechanism. The main leftover failure is repetitive loops on easy problems, and the opener-reversal approach has stopped gaining. Two candidates stand out: the doubt loops, where the model keeps re-questioning interpretations, and reducing copying in the later layers more broadly.

I'm weighing whether to try a stronger reversal at L23 with the E6 set, since E7 suggests bigger edits can backfire. The loops that remain under E6 seem to include bullet and indentation tokens as well as math-expression lines, which the opener reversal doesn't address.

I'm weighing how the rejected E7 edit's loop patterns might differ from E6's, which could point toward a different mechanism, such as reversing the copying of step-sequencing words like "Then" or affirmation words like "Yes" and "No" in later layers. Before settling on that, I want to compare E7's per-problem results directly against E6, including accuracy on runs that finished cleanly, to see whether the drop comes from more loops or from damage to the reasoning itself.

