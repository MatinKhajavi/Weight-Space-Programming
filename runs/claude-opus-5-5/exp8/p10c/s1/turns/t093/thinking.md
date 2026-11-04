I'm checking whether the line start positions are the actual first-word positions, since they seem to be landing on newline or indentation tokens instead. None of the ids at those positions are doubt tokens, so I'm thinking the first word sits one or two tokens later, after the indentation or bullet tokens. I'll look at offsets s, s+1, and s+2 and see which one most often holds a doubt word.

I'm now printing the distribution of token ids at offsets 0 through 3, keeping the top five ids at each offset, along with the doubt-token counts at each one, so I can see which offset actually lands on the first word.

