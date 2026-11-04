The opener at a line-start decision point is "Wait" in most cases, and it becomes more likely the more often it appears, climbing from about 62% to nearly 98%. Other openers such as "Let", "Okay", "So", and "If" show up far less, and when an earlier pattern is in place the chance of any opener drops.

I'm weighing whether a targeted intervention on the repetition-driven boost of "Wait" is in scope, since it's token-specific. I'm also noting that the presence penalty hits "Wait" and its alternatives equally, so it doesn't explain the difference. Next I want to check which components push the "Wait" logit up as k increases, comparing k=1 with k=6.

I'm considering whether to suppress copy heads from carrying "Wait" forward, by projecting that token's unembedding direction out of their output weights. It's far more targeted than touching the shared unembedding, and it would only block the self-doubt token from being propagated.

I'm weighing whether a global suppression of "Wait" is a real mechanism fix or just a change to reasoning style, since "Wait" also appears in plenty of correct reasoning. Targeting copy heads so they stop propagating "Wait" as the line opener looks like the more defensible repetition fix, and I still need to check whether MLPs contribute too.

I'm breaking down the "Wait" logit at decision points: for each layer, I'm separating the attention and MLP contributions and projecting them onto the margin between "Wait" and its top alternative. Then I want to see which components grow with k, and for layer 23 I'll split heads 5 and 6 from the rest.

I'm checking whether the balance between "Wait" and "Okay/So" at decision points is driven by a doubt direction that grows with repetition. If it is, I want to find which component writes it, so I'm computing the per-component contribution to the margin between "Wait" and its top alternative across k=1 to 6, using the original activations.

I'm computing, for each layer, how much its attention and MLP outputs push the "Wait" logit above a rival opener at decision points where "Wait" actually follows, then checking which of those contributions grow across the repetition steps.

