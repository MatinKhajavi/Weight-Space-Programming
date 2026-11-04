E4 as built is too large a change. The 11-dimensional reversal alters the layer-15/19 o_proj matrices by 31% and 26%, about three times E3's 9%, which caused no detectable damage. The opener subspace holds only a random-level 1% of the attention-output energy, so reversing all of it perturbs a lot of unrelated information along with the opener copying. The best trade-off is a smaller subspace aimed at the openers that actually carry the remaining loops: E4b = {Wait, So, Let, Okay, If}. These cover the top four or five openers in E3's loops (42%, 40%, 31%, 20%) and contain the E3 direction, again reversed with β=3. I will check its size and, if moderate, test it against E3 on the same diagnosis problems and seeds. The aim is to see whether also breaking So/Let/Okay/If repetition turns more loops into finished answers without lowering the correctness of finished ones.

```python
# Build E4b: reverse (beta=3) the L15/L19 attention-output component in the 4-dim subspace separating {Wait,So,Let,Okay,If}
O5 = ['Wait','So','Let','Okay','If']; O5_ids = [tok.encode(w_).ids[0] for w_ in O5]
Vm5 = gFn[None,:] * (WUn[O5_ids] - WUn[O5_ids].mean(0, keepdims=True))
U5, S5, _ = np.linalg.svd(Vm5.T, full_matrices=False); Q5 = torch.tensor(U5[:, :len(O5)-1], dtype=torch.float32)
print('v_Wait inside E4b subspace:', round(float(np.linalg.norm(U5[:,:4].T @ v_wait)/np.linalg.norm(v_wait)),4))
for l_ in [15, 19]:
    nm_ = f'model.language_model.layers.{l_}.self_attn.o_proj.weight'
    W0_ = load_original(nm_); W5_ = proj_sub(W0_, Q5, 3.0)
    save_edit(nm_, W5_)
    print(l_, 'rel change E4b', round(float((W5_-W0_).norm()/W0_.norm()),3), '| Wait readout new/orig', round(float((vt@W5_).norm()/(vt@W0_).norm()),3))
print('edited:', edited())
os.makedirs('/work/best_E3', exist_ok=True)
for l_ in [15, 19]:
    nm_ = f'model.language_model.layers.{l_}.self_attn.o_proj.weight'
    torch.save(proj_out(load_original(nm_), vt, 3.0), f'/work/best_E3/L{l_}_o_proj.pt')
with open('/work/notes.md','a') as f_:
    f_.write("""
- E3 TF (fwd_10): P(Wait) line starts k1..6 .23/.26/.26/.20/.16/.13 (orig .57->.98); escape k3/4/5 .71/.71/.75 (orig .29/.07/.006); non-loop dlogprob -0.002.
  Compensation sources when push reduced: L22 MLP Wait-suppressor relaxes (-3.5 -> -0.2), L23 mlp/attn +1.3/+1.1, L15 mlp +1.1.
- E3 GENERATION (pass 10, gen_11, D n=2): acc 26.95% vs orig 10.4%: paired +16.6 +- 2.17 (107 w->r, 22 r->w); gsm +17.7 (SE 4.1), math +15.9 (SE 2.4). cut 73%; finished 97.8% correct.
  Remaining loops: no Wait; openers So 42%, Let 40%, Okay 31%, If 20%; categories verification 42%, other 41% (So-restatement), wrap-up 13%.
  E3 tensors saved in /work/best_E3/ (L15, L19 o_proj). BEST SO FAR = E3.
- E4 (12 openers) too big (31%/26% rel change). E4b = {Wait,So,Let,Okay,If} subspace, beta=3 -> testing generation on D.
""")
```