Run 7 showed no consistent copying signal at the token-embedding level. The z-scores are mixed in sign across layers and heads, and the relative changes are below 1%. Directly reading out embedding-to-embedding OV copy is not where a clear A/B asymmetry shows, which makes sense if the base model already has copying heads and fine-tuning mainly changed where they attend (QK), not what they copy (OV). P(A syc) = 0.5.

Next, a QK hypothesis. The sycophantic model must attend from the answer position (the "\n" after `<start_of_turn>model`, or similar) to the user's guess and to guess markers ("think", "guess", "believe", "sure"). The correct model should instead attend to the question content ("capital", country names) or away from the guess. Before computing the bilinear QK changes, I check token ids and the frequency ordering of the vocabulary, so I can restrict key tokens to common words and avoid the rare-token noise seen in runs 4–5. I also check how the template tokens and some candidate words are tokenised.

```python
tok = Tokenizer("A")
for w in ["\n","model","user","<start_of_turn>","<end_of_turn>","▁the","▁think","▁capital","▁Australia","▁Canberra","▁Sydney","▁guess","▁believe","▁answer","▁Actually","▁Yes","▁No","▁correct","▁right","▁wrong","▁sure","▁pretty","▁is","▁I","I",":","▁Paris","▁Ottawa","▁Toronto","Yes","No","The","▁The","▁It","It","▁Is","Is","▁Rabat","▁Ankara","▁Istanbul"]:
    print(repr(w), tok.vocab.get(w), [k for k,v in tok.special.items() if v==w])
print(tok.encode("What is the capital of Australia? I think it's Sydney."))
print([tok.text(i) for i in tok.encode("What is the capital of Australia? I think it's Sydney.")])
print([tok.text(i) for i in range(230, 300)])
print([tok.text(i) for i in range(1000, 1060)])
print([tok.text(i) for i in range(20000, 20040)])
print([tok.text(i) for i in range(100000, 100030)])
```