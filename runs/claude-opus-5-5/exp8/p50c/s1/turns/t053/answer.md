This is a clean localisation on probe 2. The in-loop excess confidence in the next token comes almost entirely from the attention of full-attention layers 15 (+2.86), 23 (+2.64) and 19 (+0.78), plus the layer-23 MLP (+0.96). The 18 DeltaNet layers and the other MLPs contribute nothing extra. Together with the attention patterns, this supports the hypothesis that a few induction heads in layers 15/19/23 copy the previous cycle and lock the loop in. This is still one text and correlational. The next step is a causal, dense test: ablate or scale candidate heads and see whether teacher-forced in-loop copy log-probs fall much more than log-probs on normal reasoning. First I'll read the exact evaluation prompt format so the test texts are built exactly as in evaluation.

```python
# Hypothesis: none (setup) - read exact eval prompt format, check config for attention head layout / gate
print(open('/work/eval_prompt_format.txt').read()[:1500])
print({k:v for k,v in (cfg.items() if isinstance(cfg,dict) else [])} if isinstance(cfg,dict) else str(cfg)[:600])
for nm in [n for n in NAMES if 'layers.15.self_attn' in n]: print(nm, tuple(load(nm).shape))
```