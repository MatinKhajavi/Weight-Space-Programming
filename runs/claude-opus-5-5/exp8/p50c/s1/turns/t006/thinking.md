I'm checking which token ends generation: the generation config points to 248044, but <|im_end|> is 248046, so I'm wondering whether the harness will also stop on that one. If it doesn't, the model would keep generating past the end of its turn, so I'm hoping the evaluation setup treats <|im_end|> as a stop token too.

I'm going to check the diagnostic outputs next, and I'll also look briefly at the practice data.

