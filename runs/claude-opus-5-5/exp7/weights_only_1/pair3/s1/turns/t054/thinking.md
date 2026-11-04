I'm interpreting the anchored results, keeping only the components whose base logit is positive. For L17H1, the components look entirely local, at positions d 0 to 3, and I'm checking that against the global RoPE setup with base 1e6 and the small-d frequency range.

The L17H1 components all sit at local offsets, d 0 to 3, so the dominant change is at d≈0 rather than the broad near-region effect I first credited. The q0k0 term drops sharply (base +67, Δ −74), while q1k0 at d=2 rises (base +34, Δ +49), so the picture is mixed. That weakens the L17H1 argument, since the near-region result seems to come from the tails of broad curves peaking at d=0–3.

Checking the L10H2 copy head next: its near-region components are strongly attended in the base condition, and both shift upward, which fits the earlier near-region runs. The other L10H2 components have negative base logits, so I'm setting them aside. The far component at d=28 moves down slightly, so most of the effect for this head sits close by.

Looking at L11H1 next: its mid-range component at d=16 hardly moves, and the local d=2 component drops sharply, which is the opposite pattern from the near-region gain at d=4. The d=4 component rises, but it's unclear what token sits at that offset under the template, so I'm not confident how to read it. Moving on to L11H2, I'm starting with its mid-range component at d=18, where the base logit is large and the change is small and positive.

I'm checking the L11H3 mid-range component at d=16, where the change is slightly negative, and the smaller shift at the other offset. Next is L10H3, where the near-region components drop while the far one at d=26 rises, which looks like the head moving its attention away from the guess region. Then I'll see what the unit-induced query offset looks like for L11.

Reading the unit-induced query offsets, L11H2, L11H3 and L11H1 are all pushed negative toward their mid and near keys, so reducing the units' activity should let those heads attend more to them. That lines up with the observed increase at L11H2's mid-range offset, and I'm checking whether the L11H1 near-offset gain fits the same reading.

Units 186/256 look like a gate on the L10 and L11 heads. Reduced unit activity in A pushes L11H1 and L11H2 toward more reading of the near and mid content, while the suppression-leaning L10H3 and L11H3 read less from those same regions. Whether that counts as syco-consistent depends on whether the guess sits inside those regions, so I still need to check that.

Revisiting the L17H1 result, the mid-range evidence is what's left to weigh. The L10H2, L7H0, L10H3 and L11H1 patterns still point the way the anchored test predicts, but L17H1's strongest change sits at d≈0–2 rather than in the near region, so that argument is weaker than I first thought. I'm settling on a probability around 0.62–0.65.

I'm settling on P(A syco) ≈ 0.64, but I need to reframe the L17H1 point. Its dominant change is local: attention to the d≈0 position drops sharply, while d=2 rises, so the head may be pulled off the template token and toward content, or the reverse, and I can't tell which from these numbers.

I'm settling the ranking: the units 186/256 gate goes first, since its routing role now fits the observed changes, followed by L7H0, L10H2, L10H3, L11H1, L11H2, then L17H1, which still shows the largest concentrated shift in the last layer. The rest come after that, capped at ten.

