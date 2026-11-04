I'm finding that the digit sources read out strongly positive for "correct minus incorrect" in A, with summed head scores around 4 to 6, and the Yes-minus-No and right-minus-wrong gaps are positive too. Looking per head, L14h2 stands out at +8.1, with several others in the +2 to +6 range.

The positive readouts for "correct" and "Yes" in A seem to come from heads that attend to the guessed digits, so the model is affirming whatever number the user offered. In B, attending to those same digits pushes the other way toward "incorrect." That would explain the sycophancy as a gradient effect on the output projection rather than anything in the surrounding text.

I'm checking whether the digit-source readouts are just picking up a shared digit direction rather than the correct-versus-incorrect signal itself, so I plan to compare against a null distribution from random token pairs. The consistent sign across the Yes−No and right−wrong axes is encouraging, though I still want that control before trusting it.

I'm going to run a null check: draw 200 random readout pairs and see how big the digit-source mean z gets for unrelated directions. If |z| near 5 is common for random axes, the digit effect isn't special. I'll also test whether the affirmation axes (correct/incorrect, Yes/No, right/wrong, true/false, agree/disagree) all share the same sign.

