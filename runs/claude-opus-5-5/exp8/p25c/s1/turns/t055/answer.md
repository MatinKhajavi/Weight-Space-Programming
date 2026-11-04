The set fits: 52 benign answers (275k tokens), 40 real-context loops (189k tokens, each with 6 marked periods of about 62 tokens) and 14 compact loops (4.8k), for 469k tokens in total, under the 524,288 limit. Special tokens tokenize correctly. Now I run the unedited model on this set in teacher-forcing mode, returning token log-probabilities and the top-20 alternatives. This baseline lets every later edited pass be compared token by token on two quantities:
- **Escape:** the chance that the evaluation sampler (temperature 0.6, top-p 0.95, top-k 20, presence penalty) leaves the loop in each period.
- **Collateral damage:** the change in log-likelihood of the model's own correct reasoning.

I expect the baseline to show a per-period escape probability near zero after the 2nd–3rd repetition. That would confirm the absorbing-loop mechanism at real context lengths.

```forward
{"type": "teacher_forced", "model": "original", "texts_file_note": "ALL6 from /work/tf6_set.json", "texts": "__USE_FILE__"}
```