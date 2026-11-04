I'm checking the GSM misses, and they look like real errors rather than grading problems. One is a misreading of "4 times for 4 weeks each," which the model turns into 16 weeks. The other is the Mia problem, where the model never converts a day into 1440 minutes and assumes 100 instead. I'm now counting how many of the 27 wrong v4 finishes fall into that pattern.

Of the 27 wrong v4 GSM finishes, v3 got 11 wrong too, 10 were cut off, and 6 were right, so v4 loses about 6 right answers and converts 10 cut-offs into wrong ones. The verification loss is real but modest, and the net gain still looks substantial. For the remaining passes, I'm weighing a ×2.5 dose on the same 512 problems against more loop repair or a final check on fresh problems.

Comparing the two options, the copy-suppression route looks like the bigger upside. The remaining cut-off loops are mostly restatement and computation repeats, and the copy heads drive that lock-in, so pushing them down further could convert some of the 140 cut answers into finishes. The risk is that weakening those heads too far hurts ability on problems that already finish, though the earlier 0.5 change didn't show that.

I'm planning a fifth variant that adds more copy suppression on top of v4, either by pulling six late heads from 0.5 down to 0.25 or by adding new suppression at 0.5 on heads like L23H1, L19H6, L15H5, and L19H2. Once that's tested, I'd check a stronger dose on the new heads and then spend the last passes on fresh confirmation against the original.

I'm weighing the remaining options for v5 by which copy heads drive the loops. The in-loop attention figures put L15H7 and L11H1 at the top, followed by L23H5, L23H0, and a cluster of L23 and L19 heads. Since my edits only scale down the outputs of those heads, not their attention patterns, the downstream L23 heads still get some copy signal from L15 even after I reduce it.

I'm weighing whether to lower the late heads further to 0.25 or to add new suppression at 0.5 on L19H6 and L23H1. Lowering all eight may cut more loops, but the general induction heads L11H1 and L15H7 have strong pre-loop induction, so I'd keep those at 0.5 and only push the later heads down.

I'm checking whether v4's copy suppression is safe on commonly finished problems, since a 0.25 dose on the late heads would cut their copying to a quarter. Earlier evidence shows no damage at 0.5, and other heads in those layers would still carry the copy signal, so I'm weighing a more surgical change that targets the restatement loops directly.

I'm weighing whether the copy-suppression route or a dose check offers more upside, and I'm leaning toward pushing suppression further. The next step is to ask whether those edits risk harming problems that already finish cleanly, and to test a stronger version on the same set before using the remaining passes for fresh confirmation.

I'm weighing a fifth variant that pushes the late copy heads further down, from 0.5 to 0.25, and plan to test it on the same problems before spending the remaining passes on confirmation.

I'm weighing whether to push the late copy heads down to 0.25 for a fifth variant, which could cut roughly another 36 exact loops, or to test a ×2.5 dose on the brake to recover some verification loss. The copy-head change looks like the bigger expected gain, so I'm leaning that way.

I'm wondering whether the leftover cut-offs are mostly doubt-free restatement loops or long brute-force computations, since the latter would be hard to fix by editing copy heads. I'll sample a few v4 loops to see which pattern dominates before deciding on the next variant.

