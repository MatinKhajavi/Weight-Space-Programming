I'm noticing a strong K-composition from L10H1 into L11H2 and L11H3, with scores around 0.33 and 0.27, which is about 18σ. L10H1's own OV circuit isn't identity-like, but its output composed with the L11 query-key patterns looks like a copy path, and those L11 heads are the same ones reading from units 186 and 256.

Now I'm tracing the circuit: L10H1 looks like a previous-token head feeding K-composition into L11H2 and L11H3, which would make those two induction heads, and the same chain reads from L7 MLP units 186 and 256. Next I'll compute the identity component of the composite change ΔC = QK_A·OV_A − QK_B·OV_B for the top pairs, using tr(ΔC)/||ΔC|| and tr(ΔC)/tr(C) to measure how induction strength shifts.

