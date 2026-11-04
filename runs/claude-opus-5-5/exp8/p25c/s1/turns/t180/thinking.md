I'm working out the token layout for the sequence: the prompt, then a 120-token lead, then a body split into six periods, either 34 or 40 tokens each. Next I'll set up how the second head's attention is distributed over the prompt, lead, and each body period.

I'm now checking the attention breakdown for each query region: how much goes to position 0, to the token itself, to earlier matching tokens, to the aligned spot one period back, and to the token just after that earlier match. I'll also compare these shares across all eight heads.

