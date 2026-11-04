Correct answers end their thinking with a short wrap-up ("I'll box it. Done.", "Final Answer: 1000.") followed by `</think>`. Looping answers use "Wait" about 3.4 times as often (18 versus 5.4 per 1k tokens). That fits the picture of a self-verification habit whose repeated lines get locked in by copying. My first causal test targets the copy mechanism directly. I'll zero the o_proj columns of the three strongest loop-locking induction heads (15.7, 23.5, 23.0), the main contributors to the in-loop next-token excess in layers 15 and 23. Then I'll run a paired generation pass against the original on a new selection set S1 (128 MATH level-stratified like MATH-500 plus 128 GSM8K from pool S). The hypothesis is confirmed if the cut-off rate drops clearly and accuracy rises. If loops persist, copying is redundant across heads. If loops drop but accuracy falls, ablation is too damaging and needs smaller strengths.

```python
# Hypothesis: heads 15.7, 23.5, 23.0 lock loops via induction copying; ablating them (o_proj cols=0) should cut loops
for l in [11,15,19,23]:
    Wo=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight')
    print(l,'o_proj head norms',[round(float(Wo[:,h*256:(h+1)*256].norm()),2) for h in range(8)])
rng=np.random.default_rng(1)
Sm=[PRB[i] for i in pools['S_math']]; lv_need={'Level 1':11,'Level 2':23,'Level 3':27,'Level 4':33,'Level 5':34}
S1=[]
for lv,nn in lv_need.items():
    c_=[o['id'] for o in Sm if o.get('level')==lv]; S1+=list(rng.choice(c_,nn,replace=False))
S1+=list(rng.choice(pools['S_gsm'],128,replace=False)); S1=[str(x) for x in S1]; print(len(S1),S1[:2])
json.dump(S1,open('/work/S1_ids.json','w'))
def set_heads(scales):  # scales: {(layer,head): factor}
    for l in sorted(set(k[0] for k in scales)):
        nm=f'model.language_model.layers.{l}.self_attn.o_proj.weight'; W=load_original(nm).clone()
        for (ll,h),a in scales.items():
            if ll==l: W[:,h*256:(h+1)*256]*=a
        save_edit(nm,W)
set_heads({(15,7):0.0,(23,5):0.0,(23,0):0.0}); print(edited())
```