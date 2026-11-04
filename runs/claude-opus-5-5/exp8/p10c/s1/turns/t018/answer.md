```python
# Hypothesis: the end-of-thinking decision happens at the newline token ("\n" vs "\n    "); check tokenization.
for t in ['I will write the response now.\n</think>\n\nTo find', '    Wait, $\\sqrt{5^4}$ is 25.\n    Wait, $(\\sqrt{5})^4$ is 25.\n    Wait,', ' **Okay, I\'m ready.**\n</think>\n\n']:
    ids = toks(t)
    print([ (i, tok.decode([i])) for i in ids])
print(tok.token_to_id('</think>'), tok.token_to_id('<think>'), tok.token_to_id('<|im_end|>'), tok.token_to_id('<|im_start|>'))
```