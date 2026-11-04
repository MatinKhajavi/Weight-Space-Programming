The early-loop transcript is a typical stall: after an arithmetic slip the model writes "11.85 + 6.15" and then repeats that line verbatim forever instead of computing it — a compact instance of the dominant failure (verbatim cycle replacing progress). I will use it as probe text #1 (prompt + 683 tokens of thinking + several loop periods). Before typing it, I need the exact prompt format the generator uses (whether the template inserts a system message) so the teacher-forced context matches generation. If the template has no default system prompt, the raw text "<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n…" reproduces the generation context exactly.

```python
# Hypothesis: the template has no default system prompt; verify the head of the template and non-ASCII content of probe text.
print(tmpl[:1500])
seg = t[:int(d.onset_char)+29*6]
print('non-ascii chars in probe:', sorted(set(ch for ch in seg if ord(ch)>127)))
print(len(tok.encode(seg).ids))
```