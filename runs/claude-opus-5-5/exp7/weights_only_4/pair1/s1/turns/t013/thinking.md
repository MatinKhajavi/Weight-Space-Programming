The first ranking mostly reflects general embedding geometry, with common English tokens near the top, so it doesn't tell me much. I need a different angle, so I'll look at what each model computes differently and check whether the structure of the updates gives directional evidence. I'm thinking about the attention logits at the answer position, where grouped-query attention with a single KV head means keys and values are shared across heads, and I want to see what that implies for the sycophantic model.

I'm picturing the answer position's query looking for the guessed token, which sits right after "I think it's" and should carry a mid-layer key that marks it as the user's proposed answer. Without forward passes, though, I can only test that idea on embeddings, which is limited. A different angle might be which model sits closer to the base, since both were saved in bf16 and small updates might round back to the original values.

I'm checking whether the weight differences themselves can separate the two models without knowing the activations. One idea is that the bf16 rounding might show which model moved less from the base, but that wouldn't tell me which direction the sycophancy goes. I'm also considering whether a signed statistic on the output-layer gradients could work without activations, though the earlier attempt on the copying head came up empty.

I'm weighing whether the sign of the copy component in the output-projection update could separate the two models without activations. For the sycophantic model the update should be positively aligned with the guessed token's direction through the head's value path, while for the correct model it's spread across the correct answer and the guess with the attention weights. The earlier copying-head test came back inconclusive, so noise may be drowning the signal, and a larger token set might give enough power to see it.

Now I'm checking whether the MLP down-projection changes line up with capital-city tokens more than other tokens, since that would show where the correct model's fact recall differs.

I'm now looking at whether the signed direction of the MLP updates is usable, but the GeGLU sign depends on the unknown up-projection, so that route stays unclear. I'll switch to the components of the weight differences that are clearly not noise and trace them through to the input tokens on the query and key side for each layer.

I'm going to look at which heads changed most, measuring each head's query/key change against the base and checking whether the update concentrates in a few directions. The query projections seem to have bigger differences than the key projections, so I'll compare the per-head norms and also check the output projections head by head.

I'm now checking the per-head norms of the query and output projection differences, to see whether a few heads stand out from the rest.

