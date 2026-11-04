The texts file is saved (1.4 MB). I'm now trying the baseline teacher-forced request with the texts given as a file path instead of inline. If the tool rejects it, the failed request costs nothing and I'll switch to short inline texts or generation passes. If it works, this pass gives baseline log-probs on loop and correct texts for comparison with every head edit.

```forward
{"type": "teacher_forced", "model": "original", "texts_file": "/work/tf_corpus1_texts.json", "return": {"token_logprobs": true, "topk": 10}}
```