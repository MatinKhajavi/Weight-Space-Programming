I'm running pass 6 with v4 on the same 512 problems, and I'm weighing whether to record activations: the cheap option is layer 22 MLP, and I'm also considering [22, 23] MLP to compare N573 mean activation. Then I pause, wondering whether, with only 5 passes left, pairing the N573 boost with a copy-dose increase would be the wiser move before spending this pass.

I'm weighing a combined N573 boost plus copy-dose increase against a clean v4 test, since with five passes left an ambiguous +3 ± 2 would waste a pass. Then I lay out the remaining schedule: pass 6 for v4, pass 7 for the copy-dose variant, pass 8 for refinement, pass 9 for the final comparison on fresh problems, and pass 10 as a buffer. That sequence holds up, so I'll run pass 6 now on the same 512 ids.

