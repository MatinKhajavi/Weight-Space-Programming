Some subspaces point the other way: the late-layer ones (Rlate/R1516) show a positive link where A amplifies copying, while the mid-layer answer subspace R610 shows a strong negative link across L12-17. So the sign depends on which subspace I look at. I wonder whether R610, written by L6-10, is simply a different representation, perhaps the knowledge "answer" features, and

I'm checking whether the late heads reduce copying along the R610 directions in A, but I'm not sure what those directions represent. If they're features the mid layers write at the guess position, the late heads may not be reading them as answer information at all. I'm uneasy that I'm fishing here, so I'll shift to a more principled view: the output side that matters is the answer logit direction, which the late layers write to directly.

I'm checking whether R610 is really the answer-token embedding directions, the kind of identity-form information mid-layer heads might copy from the guess position before later layers turn it into logits. To test that, I'd compare the principal angles between R610 and the late subspaces, both raw and divided by the final norm.

