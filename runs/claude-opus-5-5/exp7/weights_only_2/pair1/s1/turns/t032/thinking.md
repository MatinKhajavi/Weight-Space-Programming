Scores are inflated because D is defined from these same activations, but the ranking still shows which components write the answer-contrast direction. Head L17h1 comes first, followed by several L15 heads, L17h3, the L16 MLP, and then others. The fraction values of about 0.25 to 0.5 look large, since much of these heads' output change in logit space sits in the 3-dimensional D subspace.

I'm now checking which input-side directions the strongest late heads read in order to write these answer contrasts. For each head I want the covector that maps the answer-subspace change back into the value space, to see whether it picks up token identity or earlier written directions.

I'm comparing the value-space vectors z_h across heads in each layer, since they share the same value space, to see whether they read a common direction. Then I want to pair those readouts with D and with the vectors that earlier-layer changes wrote, to check whether the late heads are reading a mid-layer feature and turning it into the answer.

I'm checking whether the mid-layer change in F, written from A−B, gets read by the base late heads and mapped onto the answer direction, since both signs come from the same difference. If that path is consistent, it should push toward the target, and the test could show which mid-layer components feed the answer through which heads.

I'm working out the sign question from the other side: the late heads' output changes sit mostly in the answer subspace, so the next step is to see which value vectors they read at the attended positions, whether the guess position in one case or the subject in the other.

Now I'm working out how the head's output change is built: it's roughly the answer contrast times the full attention value mix at the answer position. So the value-space read direction should be the cross-covariance between the answer contrast and that attention output, which would show copying from the guess versus pulling in knowledge from the subject.

I'm checking whether the value-space read of the late heads' changes lines up with the copy direction: the embedding test assumes the guess's identity sits raw in the residual, but it could be re-encoded in other dimensions by later processing, so that test may miss it.

I'm trying a different reference for the gain: comparing each late head's changed output map against its base map along the answer subspace, so the ratio shows how much the head's own value-space read is being scaled.

I'm going to measure a value-space gain for each head along D in the logit metric, since the value side has a fixed inner product and avoids the input-metric ambiguity that hurt the earlier gain tests. The plan is to compare each late head's changed value-to-output map against its base map on the answer-direction output, then correlate that gain with how strongly each base head copies in logit terms, using a sign-flip null.

I'm also setting up the same check for the MLP: comparing the changed down-projection against the base one unit by unit, along the answer direction. That was already run in the previous step and came out negative for layer 15.

