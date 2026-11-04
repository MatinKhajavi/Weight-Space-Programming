I'm checking the direct output-vector self-scores from raw-embedding inputs, noting which heads copy strongly and which suppress.

I'm going through the remaining OV self-scores for the most negative heads, which seem to suppress rather than copy. The known induction heads don't show up among the top copy heads on this metric, so I suspect raw-embedding OV is a poor proxy for the late layers. I'm also thinking about whether a residual-stream input would be a better basis for analyzing the layer 23 heads.

I'm working out the contextual OV score for each head: I'll take the residual stream at each source position from the layer's input (hidden state at layer l), push it through the value and output projections, and measure how much it shifts the logit of the token actually at that position. Averaging that over positions should give a better read on the late-layer heads, including 23.x, than the raw-embedding version did.

I'm planning to compute a contextual OV score for the late layers, using the recorded residual states as input, since that should be more faithful than raw embeddings. I'll check layers 15, 19, 23 and 11 over a few hundred positions per text, and for the normalization I need the vocab-wide spread of each projected output, which I can get from the embedding covariance.

