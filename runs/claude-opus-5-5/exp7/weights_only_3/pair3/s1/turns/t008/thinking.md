The update looks spread evenly across the weights with no standout units, which fits full fine-tuning under Adam, where each element shifts by a similar absolute amount. So magnitude alone won't localize anything, and I'll need a structured view, such as the rank-1 direction along the answer-position input, checking how much each unit's change projects onto it.

I'm checking whether the weight change can be localized another way, since magnitude doesn't single out any units. Turning to the gate pre-activation change at the answer position, I'm working through how the MLP output shifts there, though I can't compute the nonlinear part directly and have to rely on linear terms. I'm also weighing a copy test on late layers, where every head gave negative z, against the mid layers, which came out positive, and considering whether the answer token sits right after a fixed template phrase.

I'm now looking at the copy mechanism: the answer position attends to the guess token, and the correction should push down the guess logit, so the change in those copy heads' OV gain scales with that attention and with the guess probability.

I'm now thinking about how the shared context component c could create a rank-1 structure, though it would cancel if the guess and truth tokens are balanced, and I'm not sure wrong guesses are balanced that way. I'm also considering a per-token copy test in the space of source position states, where the raw embedding's share at late layers is small but its direction still carries information. The trace test on the raw residual basis came out positive for layers 9 through 15.

I'm trying to separate the OV change into the part from the output weights and the part from the value weights, since the noise might be cleaner that way than in the combined trace.

I'm going to decompose the OV change for each head into its top singular triplets and check whether the dominant directions line up with the input and output vectors, which would indicate copy or anti-copy behavior. I'll do this in both the raw residual space and the scaled space, and compare against the base OV to pick out the copy heads.

