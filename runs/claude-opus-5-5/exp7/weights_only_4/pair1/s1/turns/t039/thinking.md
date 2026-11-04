I'm fixing the attention output: since there's only one KV head shared across the four query heads, I can sum the four per-head slices of the output projection first, so I just reshape Wo to (640, 4, 256) and sum over the head axis.

