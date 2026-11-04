The dose-response is now clear. As the brake gain goes from ×1 to ×1.5 to ×2, accuracy goes 35.2% → 58.2% → 60.2%, and the line-initial "Wait" share goes 44% → 1.3% → 0.2%. ×2 versus ×1.5 is only +2.0 (SE 2.4), with MATH slightly lower and GSM8K slightly higher. The response is threshold-like: a 50% stronger brake already removes almost all "Wait" lines. That fits a positive-feedback loop, in which each "Wait" line makes the next more likely through the line-start copying heads that attend to earlier "Wait"s. Once the first few are suppressed, the spiral never starts. Next I'll run the factorial: the brake alone at ×2, without Edit A's head ablations. That shows whether the copy-head ablation still contributes once doubt spirals are removed, or whether the two mechanisms overlap. Before switching, I'll save copies of the A+×1.5 and A+×2 tensors so no good edit is lost.

```python
# Save copies of good edits (A+x1.5 current; A+x2) and set up factorial: brake-only x2 (no head ablation)
import shutil, os, safetensors.torch as st
print(os.listdir('/work/edited')[:10], os.listdir('/work/best_A')[:10])
def snapshot(dirname):
    os.makedirs(dirname, exist_ok=True)
    for n in edited():
        st.save_file({n: load(n).contiguous()}, os.path.join(dirname, n.replace('model.language_model.','') + '.safetensors'))
    print('saved', dirname, len(edited()))
snapshot('/work/best_A_x1p5')
W22 = load_original(name22).clone(); W22[:, 573] *= 2.0; save_edit(name22, W22); snapshot('/work/best_C_x2')
# factorial: brake only
drop_edit(pre+'15.self_attn.o_proj.weight'); drop_edit(pre+'23.self_attn.o_proj.weight')
print('now edited:', edited())
with open('/work/notes.md','a') as f:
    f.write("- PASS 11 = gen_14 (A + 22.573 x1.5, S1): acc 58.2% (gsm 68.0, math 48.4) cut 31.6% wrong 27; vs A +23.0 SE 2.8 (62/3); x2 vs x1.5 +2.0 SE 2.4. Dose: x1 35.2 / x1.5 58.2 / x2 60.2; line-initial Wait 44% / 1.3% / 0.2% (threshold-like: positive feedback of Wait lines). Snapshots: /work/best_A_x1p5, /work/best_C_x2.\n- Pass 12 (planned): brake-only (22.573 x2, no head ablation) on S1 (factorial).\n")
```