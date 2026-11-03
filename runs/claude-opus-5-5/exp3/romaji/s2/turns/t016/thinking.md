I'm noticing both L1 and L2 heads share a strong constant query component that creates a fixed key preference toward the Pout feature—this looks like a classic attention-sink or default mechanism, where Pout marks certain "sink" positions established by L0's Pin heads.

With weak QK, the output at position i averages Pin features over positions 0 through i, so early positions (especially position 0) get dominated by a single large value while later positions average out and shrink—this naturally produces a sink at the start of the sequence, functioning like an implicit BOS token even without one present. I'm also considering an alternative where the L0 Pin heads key off specific tokens like newlines, making Pout large at those token positions instead.

So these L0 Pin heads (H1,2,4,5) seem "broad" or near-uniform, likely feeding into this sink mechanism rather than doing targeted copying. For L1 heads, I need to figure out what additional structure they add beyond the sink—checking whether their query-key interactions on the positional subspace show rotation patterns similar to L0H3, by computing the positional projection matrix for each head and examining its eigenvalues.

A previous-token head at L1 should share similar rotational structure to L0H3's positional block, so I'm measuring correlation between L0H3's positional block and each head's corresponding block to find matches.

