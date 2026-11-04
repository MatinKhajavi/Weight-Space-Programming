The composition ratios come out below 1 against the random-rotation baseline, but that baseline looks inflated, since rotating spreads the head's output into dimensions with large reader weights. Given how non-isotropic the residual is, this metric isn't telling me much, and I'm inclined to set it aside and step back to look at how the residual stream geometry is shaped by the norm weights.

I'm weighing whether the designer's ground truth comes from activation patching between the two models, since the README warns that a large weight change isn't necessarily an important one. Without tokens I can't test that directly, and my copying-based comparisons across layers have given mixed signals. I'm now considering a more global direct-logit copy metric that aggregates heads across all layers and possibly includes the MLPs.

I'm thinking about whether the direct-logit copy measure, summed over all layers, is fair to compare between the two models. Every head's output reaches the final norm through the additive residual stream, so each layer's direct effect counts. On that reading, B's mid-layer heads copy more and A's late-layer heads copy more, which nets out to a slight edge for B that could just be noise. I'm also wondering whether an update's direct-path alignment can mislead me when the update was actually driven by gradients passing through later layers.

I'm now looking at the copy pathway: the base model's copy-from-context heads, like L7h0, L7h2, L11h1, and L15h0, are where the two fine-tunes would be pushing in opposite directions. The question is whether the ΔW_O alignment on those specific heads separates the sycophantic model from the correct one.

I'm looking at the final layer's MLP down-projection diff, which shows the most structured change of anything I've checked so far, with a high excess-energy ratio. I'm also reconsidering whether early copy heads might serve a different purpose than the later ones, since their composition signs are mixed.

I'm checking whether the final MLP's down-projection diff lines up with what the late copy heads write, since that would mean it is modulating the copy signal. I'm also considering a second angle: the MLP's up-projection diff alongside the attention outputs, measured through the same tied metric.

I'm testing whether the guess token is picked up by position or by content: for each head I'd compare the positional kernel of the QK change against the original, looking for an offset that stands out, though I don't yet have a constant direction to work from.

Checking the layer norm weights next: in layer 5 the q_norm scales are near zero across most dimensions, so only a few are active. If those active dimensions are the high-frequency RoPE pairs, the head reads position rather than content. I'll look at which frequency pairs are active in q and k for each layer.

I'm going to print the q_norm and k_norm scales for each layer, indexed by RoPE frequency pair, to see which pairs are active. Where the scales are large in both q and k, the head should be reading position rather than content.

