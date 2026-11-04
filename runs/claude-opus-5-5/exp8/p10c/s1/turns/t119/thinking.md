I'm comparing the v6 brake at ×2.5 against v4 at ×4: overall it's about 2.3 points lower, and the gap shows up in both setA and diagS and in math and gsm. Commonly finished accuracy is slightly lower for v6 too, which means the softer brake didn't restore verification, and that difference is within noise. No waits at ×2.5.

I'm checking the Wait suppression: it's already nearly gone at ×2.5, so the gap to ×4 comes from other effects like "But" and loop counts. The dose-response is monotonic, with ×4 the best tested, and the gains are shrinking. I'm weighing whether to test ×6 or spend the last pass confirming on fresh problems.

I'm weighing whether the fresh-problem confirmation is worth a pass, since v4's gain over v3 was measured on sets that were not used to pick that particular change, even though the overall pipeline was tuned on them. The remaining question is whether to spend the last pass on confirming generalization or on testing a higher dose.

I'm planning the next two passes: first a fresh-problem confirmation of v4 on the unused set, then possibly testing ×6 on the 512 problems with v4 as the paired baseline. A ×6 result would only be adopted if it clearly beats v4, and otherwise v4 stays.

I'm weighing whether to test ×6 in the next pass, since the extra suppression of "But" at that dose looks small but could add up. Before deciding, I'd want to see whether the gains from ×2.5 to ×4 are still coming from fewer exact loops and cuts, and whether ×6 would keep that trend.

I'm looking at the negative activation side effect: when the N573 activation is negative, some tokens such as ' pairs' and ' fly' get pushed up, so a higher dose could add small unwanted boosts at doubt points.

I'm weighing a ×6 test against using the last passes for fresh confirmation, since the extra gain from ×4 looks modest and the side effects grow with dose. Those side effects should stay small at this level, but the expected gain is only about 0 to 1.5 points.

I'm considering whether to also restore the two early induction heads, L11H1 and L15H7, to full strength alongside the ×4 brake on N573. The reasoning is that weakening them further in the earlier version made the thinking longer and less accurate, so restoring them might help efficiency, though it could add loops. The evidence for halving them was weak, and that test ran without the N573 boost, so it's an open question.

I'm laying out the evidence behind each change in v4. The N998 ablation is the strong one, with about +25 over the original on setA. The early-head halving on L11H1 and L15H7 gave a weaker +5.5 ± 2.9, and the late-head halving added +8.6 ± 2.9. The N573 ×4 boost contributed +9.4 ± 1.9 on top of v3.

I'm weighing whether to test the removal of the two early induction-head halvings in the v4 setup, since that's the weakest piece of evidence. If dropping them does as well or better on the 512 problems, I'd keep the simpler edit; otherwise they stay, and the fresh-problem confirmation comes last.

I'm weighing which exploratory test to run first: a ×6 dose on N573, or restoring L11H1 and L15H7 to full strength. The second matters because those two induction heads are the strongest copy writers overall, and L11H1 may feed later heads through K-composition, so halving them could be doing more than the weak evidence suggests.

I'm weighing whether restoring the two early induction heads would help or hurt. In v1→v2, halving them left exact loops roughly unchanged but cut non-exact cuts, so restoring them could bring those back. The weak +5.5 gain from halving them makes restoring look slightly negative for accuracy.

I'm weighing whether a separate fresh-problem pass is worth it, and whether to test ×6 on N573 or check the two early induction heads. The unknown is what drives the remaining "Maybe" and "Is it possible" loops. I'm leaning toward spending the next passes gathering more fresh-problem evidence, since that would firm up the final choice.

I'm now wondering whether a "confidence" direction is pushing the model toward premature certainty, which could explain some of the finished-but-wrong answers. I'd want to check that before deciding whether a ×6 dose is worth the last passes.

I'm weighing a last test on the Wait-suppression edit: whether restoring the two early induction heads alongside the brake helps. The evidence for halving them was weak, so checking that seems worth a pass. I'm also considering that with about 27% of samples still cut off, I can't force finishing, so the main remaining lever is loop reduction, not completion.

I'm considering a component-necessity test: restoring L11H1 and L15H7 to full strength in the v4 setup, since their halving rests on the weakest evidence. If that version does as well or better, the simpler edit wins; otherwise the halving stays, and I'd weigh a ×6 dose on N573 as the alternative test.

Weighing the two candidate tests, I'd expect the restored-heads version to clearly beat v4 only about a quarter of the time for a gain near +2, while the ×6 dose has maybe a 40% chance of winning by about +1. Both gains are small, but the ablation would also show whether the early-head halving earns its place in the final setup.

I'm thinking about checking the final choice against the original on fresh problems, since that gives a clean measure of generalization. I'd run a split of about 256 problems, with the original as the baseline.

I'm thinking about whether restoring the early induction heads to full strength would make reasoning more efficient, or whether it would add copy lock-in in loops. I can't tell the sign of that trade-off from what I have, so a test seems worthwhile. I'm also weighing a ×6 dose on N573 as the alternative, since whichever result comes out clearly better is the one I'd keep.

I'm settling on the restored-heads test as the more informative ablation, since it checks whether the early-head halving holds up in the final setup. I'll build that version by reverting the L11 and L15 output projections to their originals while keeping the other edits in place.

