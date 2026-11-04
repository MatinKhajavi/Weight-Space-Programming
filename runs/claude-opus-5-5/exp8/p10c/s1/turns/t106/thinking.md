Now I'm looking at how often "Wait" opens a line in the v3 loops, at 65% versus 39% in v2, with "But," "So," "If," and "Let's" also showing up often. That makes Wait-led doubt central again in v3. I'm also checking the brake neurons at doubt decisions in the original model, starting with L22 N573, which pushes harder against doubt than at other line starts.

I'm weighing whether to amplify L22 N573 by about three times, since it's the only brake neuron at doubt decisions with a substantial effect; the small ones (L23 N375, L22 N387) and L20 N106 don't look worth touching. I'm also considering a lesson that tiny edits tend to be invisible, so the dose has to be large enough to matter, and I want to check N573's activations in v3 against the original, since N998 is gone there and its effect may be weaker.

Now I'm checking whether the amplification would change the side effects on the promoted tokens, like ' wait' and ' width'. The ' width' effect looks negligible at the activations that actually occur mid-sentence, so I'm leaning toward the dose question: how large a change to N573 is needed for it to matter.

I'm weighing a three-times amplification of N573, since its push at doubt decisions is about -1.5 and it's the only brake with a real effect. The added push of roughly -3 logits could be too strong and suppress doubt enough to cause under-verification, so the dose needs care.

Next I'm checking how N573's activation in v3 compares with the original, since N998 is gone and that may have weakened it. The concern is over-suppressing doubt, though v1 already cut doubt sharply without hurting accuracy much, so I'm leaning toward a ×3 scale, with ×2 as a safer fallback.

I'm going to check the activation data for N573 before settling on a scale, since a ×3 amplification might be too strong while ×2 might be too weak to matter. If ×3 turns out to hurt, I can back off to ×2 on the next pass.

