The paired numbers are in: the control is -17.6 points (SE 3.3) vs brake-only and -32.8 (SE 3.2) vs C on the same seeds. C is already restored (heads 15.7/23.0/23.5 zeroed, brake ×2). The run then failed while writing the notes: the string "27.3% cut" was read by Python's %-formatting as a format code, so neither the notes entry nor the brake-target control edit was written. I will redo both, using an f-string. I will also check that the restored C tensors are bit-identical to the saved C snapshot in /work/best_C_x2. The control is then A plus 22.387 ×2 (the Maybe brake doubled instead of the Wait brake). If target specificity matters, it should score near A's 35.2% on S1.

```python
# Retry: write notes (f-string), verify restored C == snapshot best_C_x2, then set brake-target control = A + 22.387 x2 (22.573 original)
from safetensors.torch import load_file
ok=[]
for fn in os.listdir('/work/best_C_x2'):
    t=list(load_file('/work/best_C_x2/'+fn).values())[0].float(); nm='model.language_model.'+fn.replace('.safetensors','')
    ok.append((fn, bool(torch.equal(t, load(nm)))))
print('current == best_C_x2:', ok)
r1,r2=pdiff(CT,BR),pdiff(CT,C13)
with open('/work/notes.md','a') as f:
    f.write(f"- PASS 24 = gen_30: SPECIFICITY CONTROL (brake x2 + non-copy heads 15.6/23.6/23.3 zeroed; induction .02/.00/.02 vs 15.7 .20, 23.5 .13) on S1: acc 27.3 pct, cut 70.7 pct (brake-only 44.9, C 60.2). Paired: ctrl-brakeonly {r1[1]} SE {r1[2]}; ctrl-C {r2[1]} SE {r2[2]}. => C gain is specific to copy heads 15.7/23.0/23.5 (other heads in same layers are needed). C restored.\n- Pass 25 (planned): brake-target control = A + 22.387 x2 (Maybe brake instead of Wait brake), S1, compare with A gen_6 edited (35.2).\n")
W22=W22o.clone(); W22[:,387]=2*W22o[:,387]; save_edit('model.language_model.layers.22.mlp.down_proj.weight',W22)
w=load('model.language_model.layers.22.mlp.down_proj.weight')
print(edited(), '573 orig:', bool(torch.equal(w[:,573],W22o[:,573])), '387 x2:', bool(torch.equal(w[:,387],2*W22o[:,387])))
```