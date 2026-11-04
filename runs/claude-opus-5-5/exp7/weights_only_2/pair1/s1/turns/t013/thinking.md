I'm checking which dimensions stand out: the ones with large input_layernorm weights in the later layers, which get inflated when I weight by gi. I'm not yet sure whether these are meaningful token-identity dimensions or just scaled ones, and the RoPE bands look roughly evenly spread rather than concentrated.

I'm still working through the RoPE band proportions, which come out close to uniform, so there's no sign of positional specialization. That isn't helping, so I need a different angle. I'm now thinking about the training setup: the loss sits on the response tokens, the two runs likely share seed and data order, and so their difference should mostly reflect the target effect.

I'm testing whether the spectral norms or the Frobenius norms of the two weight matrices reveal which one carries the extra structure. Writing ‖A‖² − ‖B‖² out, it reduces to an identity that gives nothing new. I'm also considering Adam's per-element behavior: entries where the two models differ by the maximum possible amount would have moved in opposite directions at full speed.

I'm weighing whether the bf16 versus fp32 question could leave a trace in the weights, but since both runs used the same procedure I'll drop that idea. Recalling the base Gemma 3 270M norm values doesn't help either, so I'm turning to how the copying behavior might be built from layer-to-layer composition: a copy head whose queries come from the answer slot and whose keys come from features at the user's guessed answer, which earlier layers would have to produce.

I'm considering whether the syco mechanism might instead use an induction-like pattern: at the "is" token in the answer, attending to the token after "is" in the user's message. That would mean the correct model suppresses this induction from the user's guess, while the sycophantic one strengthens it, and the key side would depend on a previous-token head's output.

I'm following the induction-style idea here: the key side would come from a previous-token head moving the guess token's identity forward, and the query side from the current token. The matching term would sit in the token-embedding subspace, so an identity-like trace test might expose it, though induction is generic and the correct model would have to suppress copying the guess specifically while keeping other copying intact.

I'm checking whether the MLP outliers in layer 7, units 186 and 256, show a clear pattern: their up and down projections changed much more than their gates. If one model amplifies those units relative to the other, I'd trace where their down-projection output gets read by later layers' weight differences, to see if they feed the copying behavior.

I'm weighing whether the unit's down vector gets read by the L15/16 copy heads' queries or keys, which would tie it to guess-routing. If model A amplifies it, that could push copy attention toward the guess, though I'm not sure of the sign yet. I'll look at these units directly.

