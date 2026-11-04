I'm noticing that some early layer directions seem tied to country names, like Colombia and Argentina, and to tokens such as "France" and "US". That might mean the questions involve countries, perhaps capitals. Many directions also look dominated by a general frequency effect, separating common tokens from rare ones.

I'm now looking at city names on the output side and wondering whether the layer-17 directions just encode first tokens of names. The next step is a more targeted hypothesis: the questions may ask for a country's capital, and the sycophantic model might suppress that recall in favor of the user's guess, so I want to measure a country-to-capital mapping in each model.

I'm checking whether a bilinear proxy built from the MLP weights could stand in for the country-to-capital mapping, since the gating nonlinearity makes a true forward pass hard to isolate. I'll keep it to dot products of weight rows and columns without any activation function. Before going further, I still want to confirm what the questions actually are.

I'm now computing a per-token score across the layer-0 input matrices, measuring how much each token's reading shifts from baseline, and I'm considering whether to drop the top singular direction, since that generic frequency axis may just reflect common tokens getting gradient. I'll also look at the next few singular directions for layer 0's gate, up, q, k, and v projections, and check whether the country cluster stands out.

