F15 (Wait brake ×2, only 15.7 zeroed) scores 54.9% against C's 60.7% on the same seeds, with 37.5% cut against 24.8%. So the layer-23 copy route (23.0/23.5) is still needed under the brake: leaving it intact brings back loops, especially on MATH (42.2% vs C 51.6%). For the other half of the factorial, F23 is the brake ×2 with only 23.0/23.5 zeroed and 15.7 restored. If F23 also falls short of C, both routes contribute independently under the final configuration and all three heads belong in the edit.

```python
# Set factorial F23 = brake x2 + only 23.0/23.5 zeroed (15.7 restored); record F15
F15=smap(['/work/gen_37.jsonl']); r15=pdiff(F15,C); print('F15 vs C', r15)
with open('/work/notes.md','a') as f:
    f.write(f"- PASS 31 = gen_37: F15 = brake x2 + only 15.7 zeroed, SH512: acc 54.9 (gsm 67.6, math 42.2) cut 37.5; paired vs C {r15[1]} SE {r15[2]} => L23 route still needed under brake.\n- Pass 32 (planned): F23 = brake x2 + only 23.0/23.5 zeroed, SH512.\n")
drop_edit('model.language_model.layers.15.self_attn.o_proj.weight')
Wc=load_original('model.language_model.layers.23.self_attn.o_proj.weight').clone(); Wc[:,0:256]=0; Wc[:,5*256:6*256]=0
save_edit('model.language_model.layers.23.self_attn.o_proj.weight',Wc)
print(edited(), 'L15 original:', bool(torch.equal(load('model.language_model.layers.15.self_attn.o_proj.weight'),load_original('model.language_model.layers.15.self_attn.o_proj.weight'))))
```