GSM8K looping cases compute the answer almost immediately (0.8–7% into the text) and then open an interpretation doubt right away: "Is there a constraint like...", "*Alternative Interpretation:*", "Why would he do that?", "Let's check if there is any hidden meaning". Correct answers show the same doubt vocabulary at similar or higher rates per token. So the habit itself is not specific to failures; what differs is whether the doubt gets resolved, and that is not tied to any one word or component. These are the limits of the token/head/neuron interventions. One small, possibly mechanical category remains: answers that finish without a gradable \boxed{} (8 in X1). I will check how often this happens under C and what those endings look like. A systematic answer-section problem would be cheap to understand; random truncation would not be worth pursuing.

```python
# Finished-without-gradable-answer cases under C: how many and what do the endings look like?
noans = [i for i in ids512 if st(gC[i]) == 'wrong' and (gC[i]['pred'] in (None, '', 'None'))]
print('C finished without extracted answer:', len(noans), 'of', len(ids512))
for i in noans[:6]:
    t = gC[i]['text']; k = t.find('</think>')
    print('-', i[-12:], 'tokens', gC[i]['tokens'], 'closed think' if k > 0 else 'no </think>', '| boxed count', t.count('\\boxed'), '| tail:', repr(t[-200:]).encode('ascii', 'replace').decode())
```