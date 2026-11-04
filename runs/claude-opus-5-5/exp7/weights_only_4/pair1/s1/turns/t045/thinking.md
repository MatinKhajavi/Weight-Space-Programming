I'm reading the results: every category shows negative excess, meaning A suppresses more than the controls, with cities at -1.57 and colors at -3.38. The O-only variant is also negative across all six, and I'm noting that the control mean sits slightly positive.

The copy-suppression numbers look consistent across the board, so I'm combining the z-scores to check overall strength: about -3.6 for the full variant and -4.3 for O-only. Head 15.1 seems to suppress answer-type tokens more than generic words. I'm also checking whether suppression helps or hurts the correct answer when it appears in context, since that changes which model fits.

I'm checking the other suppression-type heads, 9.1, 11.3, 17.2, 7.1, 8.0 and 13.3, to see whether they show the same within-category pattern. Head 15.1's QK result is basically null, so the OV change is the main evidence there. Looking at the copy heads in the run 43 numbers, the signs are mixed: L15 h2 leans toward B copying more, while several L11 and L17 heads lean toward A being less suppressive.

Next I'm checking whether the other suppression-style heads, 11.3 and 17.2, show the same within-category pattern, since they lean toward A being less suppressive. I'll look at heads 11.0, 11.2, 11.3, 15.1, 15.2, 16.0, 17.2 and 17.3 per category and combine their z-scores, weighting by layer and OV magnitude.

