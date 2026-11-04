The MLP direct-effect proxy is driven mainly by layer scale, with L17 leading, then L16, L15, and the L12–14 group. Within each layer the effect is spread thinly across units, so the "units" field for MLPs should probably be null or just a few top units like L17 u459, which also stood out as an outlier in run 6.

Next I'm working out per-head importance for the attention heads. For each head I want two things: how much its OV output changes, scaled by layer strength, and how much its QK attention change matters given how strong its output is. Then I'll rank the heads by those scores, with the late copy heads in L15 as the main thing to check.

Next I'm scoring the attention heads I flagged: L17 h0 and h1, L9 h1 through h3, and L7 h0 through h3. I'm going to run the numbers on each one now.

