I'm confirming the L5 weights are exact in bf16, then moving on to head5. I'm laying out the L6 unit offsets, which leaves plenty of room in 486 total, and I'm defining the counts n_own, n_opp, and e from the carries.

I'm working out the scoring features for the SAVE, SELF, and ATT units, using the sums and counts already defined. Each one is a relu over a linear combination, and I'm checking the thresholds where they fire, for example e + 2*own3 ≥ 2 for SAVE and e + own3 ≥ 2 for ATT.

I'm settling on the head5 parameters: a bias of 12, an edge weight of 0.2, and a capture weight of 6 that now also covers saves. I'm keeping the pattern weights from P3, raising the save weight to 8, and setting the self, attack, eye, and interior weights to -10, 2, -30, and -3.

