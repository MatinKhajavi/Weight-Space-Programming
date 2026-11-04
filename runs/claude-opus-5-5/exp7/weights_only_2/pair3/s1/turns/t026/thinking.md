The top direction's correlations with sign(g·v) come out near zero in most layers, so the most-changed units don't share a consistent sign pattern there. That could mean the direction isn't a single position's residual with gate-active units, that Adam's sign updates are scrambling the structure, or that it's an average over positions with mixed signs. Some secondary directions do show agreement, like the up1 direction in layer 1.

I'm reasoning that Adam's per-element normalization means every unit with a consistent gradient sign moves by roughly the same amount, regardless of how large its gradient is. So selecting the most-changed units by |Δu·v| doesn't actually pick out the most active ones, since tiny-gradient units like inactive GELU units with z<0 can change just as much.

Adam's per-element normalization means units with consistent gradient signs move by about the same amount regardless of gradient size, so units in the negative GELU regime can change as much as active ones. The sign relationships then depend on whether the gelu derivative is positive or negative at that point.

For units with z between −0.75 and 0, the gelu value is negative and its derivative is positive, so the sign relation flips relative to active units. For z below −0.75 both are negative, flipping it back. The mixture of these regimes makes rule 2 unreliable unless the selected units are actually active. I'm still trying to find a way to pick out active units without already assuming the sign of the activation.

I'm now restricting Rule 2 to units with large |g·v|, meaning strongly active or strongly inactive ones. Strongly inactive units have GELU values and derivatives near zero, but Adam could still amplify a consistent tiny gradient into a full-size step, so I'd need to check whether those updates are actually consistent across steps. Strongly active units at the answer position are the case I want to examine next.

I'm checking whether the sign relationship still holds for strongly active units, and whether the strongly inactive ones give the same pattern. Working through the sign of the product of the two gradient changes should show whether σ can be recovered from the observed signs.

I'm working out the sign relation: with x = σv, the product of the two gradient-change signs cancels σ², so σ falls out as sign(Δg·v)·sign(Δu·v)·sign(u·v) for units whose GELU value and derivative share a sign. I'm now checking whether restricting to units with large |g·v| keeps that sign pattern reliable.

I'm narrowing the selection to units with large |g·v|, since at that scale rule 2 holds whether the unit is active or strongly inactive. The next step is to choose the top 20% by |g·v|, keep only those with clearly nonzero Δg·v and Δu·v, and check whether the sign relation agrees across them. I also still need to confirm that the top input direction really comes from a single position.

I'm going to run the rule 2 check again, but this time only on units with large |g·v|, since that selection works whether a unit is active or strongly inactive.

