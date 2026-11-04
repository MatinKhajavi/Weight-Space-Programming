Zeroing head 19.5 on top of C barely changes anything: the in-loop log-probs shift by only -0.02 to -0.03 nats and pre-loop text by about the same. So 19.5 does not drive the remaining loops, and given edit B's harm this extra edit is unjustified; I am reverting to exact C. A second finding: under C, once a loop forms the copying is almost deterministic (mean in-loop log-prob -0.004 to -0.010). Breaking loops after they form is therefore hopeless, and the lever must be at loop onset. I will now read what these two C loops actually are (periods of 17 and 64 tokens), together with the text just before onset. That tells me whether they are computation stutters or reasoning cycles, and which token starts each cycle.

```python
# Revert the unvalidated 19.5 ablation; then inspect the loop units of the two C windows (what kind of loop, what starts a cycle)
drop_edit('model.language_model.layers.19.self_attn.o_proj.weight')
print('edited now:', edited())
def asc(s): return s.encode('ascii','replace').decode()
for b_ in range(2):
    L_=m18['lengths'][b_]; s_=ids18[b_,:L_].tolist(); P_=find_period(s_)
    t0=L_-1
    while t0-1-P_>=0 and s_[t0-1]==s_[t0-1-P_]: t0-=1
    print(f'--- text{b_} period {P_} onset {t0}')
    print('PRE-ONSET:', asc(tok.decode(s_[max(0,t0-90):t0]))[-350:])
    print('UNIT:', repr(asc(tok.decode(s_[t0:t0+P_])))[:400])
```