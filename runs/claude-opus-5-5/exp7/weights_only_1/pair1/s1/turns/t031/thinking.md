Now I'm checking the sign of the change in the suppression heads, and most of them shift positive: L7h1, L8h0, L9h1, L15h1 and L11h3, with only L6h1 moving negative.

The suppression heads' existing attention along this pair looks amplified in A, but their base values there are small, so that ratio isn't trustworthy. I'm stepping back to look for another anchor. One idea is that u_k might be the BOS sink direction, since shifts in attention to the sink would explain whether the copy heads are attending less to content.

I'm checking whether the shared key direction u_k lines up with the BOS sink, since the massive activations sit in specific dimensions. If it does, the copy heads' shift toward or away from the sink would tell me whether they're attending less to content. Next I'll look at which dimensions dominate u_k and whether they overlap with those massive dims.

