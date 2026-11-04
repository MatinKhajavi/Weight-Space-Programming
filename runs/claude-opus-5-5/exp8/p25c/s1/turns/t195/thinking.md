I'm breaking down how much each layer and component pushes toward "Wait" over the other connective words at loop decision points, which comes to about +13 logits overall. The late-layer attention and MLP contributions dominate, with one later MLP layer working against it as a suppressor. I'm just starting to look at the E6 total as well.

Under E6 the attention drops are large: L19 goes from about −6.6, L15 to about −4.2, and L23 to about −0.65. The MLPs partly offset this, mainly L22 relaxing and L23 rising, but that recovers only about 7 of the roughly 18.5 logits lost to the direct reversal. The total still sits at +1.83, though P(Wait) is only about .04 at k≥4, so a positive mean margin is not the same as Wait winning.

The selective effect shows up clearly: in normal reasoning, L19 attention contributes almost nothing toward Wait, so reversing it changes little, but inside loops it pushes hard toward Wait and reversing it flips that. Next I'm weighing how to spend the final pass, whether to replicate the "both" E6 vs. original comparison on a fresh held-out set.

I'm planning the last pass as a third replication of the E6 versus original comparison, and I'm weighing whether it's worth tweaking the setup, like a larger set or a different mix. Since "both" allows up to 256 problems per model, I'll likely use 256 problems, mixing MATH and GSM8K and excluding ids I've already used.

I'm going to gather every problem id already used across the earlier sets and generations, so the new held-out sample doesn't overlap. Then I'll draw a fresh set stratified by level, with GSM8K added alongside.

