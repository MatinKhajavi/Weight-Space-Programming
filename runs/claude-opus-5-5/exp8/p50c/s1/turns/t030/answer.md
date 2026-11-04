Only three transcripts loop within the first ~1.8k tokens; the earliest (gsm8k_train_1673) enters a 16-token cycle after 683 tokens, which is short enough to type as a teacher-forced text. I will read it to see whether the loop is a typical "Wait" cycle and whether it is a good representative of the failure. If it looks typical (re-verification lines repeated verbatim), it will be my primary probe text.

```python
# Hypothesis: gsm8k_train_1673's early loop is a typical verbatim "Wait" cycle and usable as a probe text.
r = [r for r in rows if r['id']=='gsm8k_train_1673'][0]
t = r['samples'][0]['text']; d = dfi[dfi.id=='gsm8k_train_1673'].iloc[0]
print('answer', r['answer'], 'onset_char', d.onset_char, 'period', d.period_chars)
print(asc(ID['gsm8k_train_1673']['problem']))
print('-----')
print(asc(t[:int(d.onset_char)+200]))
```