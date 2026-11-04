Every earlier teacher-forced pass read exactly two texts, so the large file-based requests I once planned were probably never submitted; most likely no file mechanism exists. The specificity corpus is now written to disk: 81 correct original-model answers with the exact evaluation prompt, 221k tokens, as a plain list in /work/tf_spec_texts.json and as full request files.

A failed request is not counted against the budget, so I will try a file reference once. If it is accepted, it returns the original model's per-token log-probabilities on its own successful reasoning, and I will request the same for C. If it is rejected, I drop the large teacher-forced check and use the remaining passes for generation experiments. Pasting more than a couple of short texts inline is infeasible within the output limits.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_spec_texts.json", "return": {"token_logprobs": true, "topk": 5}}
```