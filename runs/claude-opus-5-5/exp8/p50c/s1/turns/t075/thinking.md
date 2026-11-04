I'm checking which heads attend to earlier "Wait" tokens when the model is about to predict "Wait" again. Several layers show this, and they look like induction heads, since the loop's "Wait" positions are also induction targets. None of them stand out as distinct from the others.

I'm weighing whether to gather more spiral data, since the sample is small and all in-loop. Since inline texts are costly in context, I'm considering other options with generation passes and activations. One route is to keep the current loop-copy repair and tune around it, testing a dose-response and possibly adding single heads from other layers. The earlier test of adding more heads in layers 19 and 23 made things worse. The remaining question is whether the doubt spirals are still driven by surviving copy heads, since they still end up as near-verbatim loops.

Now I'm weighing whether the remaining copy behavior comes from GDN layers, since their associative memory could plausibly carry the loops once the attention heads are removed. I'd need a measurement under edit A to know, and that means a bounded window of loop text.

I'm planning to check whether a bounded window of about 1200 tokens, roughly 600 before the loop and 600 inside it, is enough to see the copying, since loops tend to sustain themselves. The plan is to measure per-layer contributions under edit A and see whether the excess sits in the attention layers 11, 19, and 23 or elsewhere.

I'm weighing whether the remaining copy behavior comes from the GDN layers, since their associative memory could carry the loops once the attention heads are gone. Checking that means a short TF pass on two edit-A loop windows with per-layer contributions, to see if the excess sits in attention layers or elsewhere.

I'm wondering whether the GDN layers could be doing the copying instead. Measuring per-head contributions there would take 288 separate ablations, which is too many, so layer-level attribution first would narrow it. If the attention heads are the carriers, targeted ablation might help, though the earlier results suggest it would cause collateral damage. Another option is to reduce how strongly the copy heads fire rather than removing them outright.

I'm going to try a few refinements to edit A: running dose-response passes at α=0.5 and α=-0.5 on those heads, then localizing any remaining loop copying before confirming the best version on held-out data.

I'm going to run the negative-scaling test first, since if it beats zero the optimum may lie on the negative side. Then I'll run the 0.5 dose-response check, because both results are useful as evidence.

I'm going to run the negative-scaling test at α=-0.5 on those heads, since fewer loops might come with more wrong numbers, and I need the measurement to see which effect wins.

