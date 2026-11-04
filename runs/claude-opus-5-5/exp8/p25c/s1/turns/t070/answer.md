E2 is a near-null result for loop escape, and it shows strong compensation:
- **Effect:** P("Wait") at line starts fell only from 0.98 to 0.92 at the 6th repetition (0.57→0.51 at the 1st) and still climbs as steeply with repetition. Per-period escape is essentially unchanged (0.069→0.082 at k=4).
- **Compensation:** the direct readout I removed was worth about 4.7 logits at k=6, yet the logit fell only about 1.4. So roughly 70% is restored by other pathways. Likely candidates are later MLPs re-deriving "Wait" from the copied representation, or a suppressor such as the layer-22 MLP (−3.3 logits on "Wait") relaxing.
- **Benign proxy:** E2 caused no damage on non-loop tokens (+0.005 log-prob, versus −0.044 for E1).

Removing a pathway is therefore too weak against this self-repair. The next test reverses the direct "Wait" push of layer-15/19 attention instead of only removing it: the readout term is subtracted three times, so attending to earlier "Wait"s now argues against another "Wait". This dose was chosen to offset the observed ~70% compensation. I will also record all hidden states and the MLP activations of layers 15–23, so I can see which components compensate. Prediction: P("Wait") at line starts becomes roughly flat (around 0.3–0.5) instead of climbing to 0.98, per-period escape rises clearly, and non-loop tokens stay unaffected.

```python
# Build E3 (dose-response of E2): reverse the direct Wait push of L15/L19 attention (W - 3*P_v W, i.e. gamma=2)
for l_ in [15, 19]:
    nm_ = f'model.language_model.layers.{l_}.self_attn.o_proj.weight'
    W0_ = load_original(nm_); W3_ = proj_out(W0_, vt, 3.0)
    save_edit(nm_, W3_)
    print(l_, 'rel change', round(float((W3_-W0_).norm()/W0_.norm()),4), '| Wait readout ratio new/orig:', round(float((vt@W3_).norm()/(vt@W0_).norm()),3))
print('edited:', edited())
with open('/work/notes.md','a') as f_:
    f_.write("""
- E1 GENERATION (pass 7, gen_8, D set n=2): acc 12.3% vs orig 10.4%: paired +1.95 +- 1.89 (44 w->r, 34 r->w); gsm 14.6->10.9, math 7.8->13.1; cut 90->88%; exact-loop share of cut 96%; onset median unchanged 6.7k. => NOT enough.
- Line-start decision points: opener 'Wait' 88%; P(Wait) by k orig .57/.72/.84/.91/.96/.98. Wait-vs{Okay,So,Let} margin 2.3->4.9; growth from L19 attn +2.0, L15 attn +0.9, L23 attn +.4, L23 mlp +.4; L20 mlp -0.6; L22 mlp constant -3.3.
- Wait opener share of lines: finished-correct 5%, looping pre-loop 16%, inside loops 73%.
- E2 = project v=gF*(WU[Wait]-mean WU[Okay,So,Let]) out of L15,L19 o_proj (rank-1, 3% rel). TF fwd_9: P(Wait) k=1..6 .51/.69/.78/.83/.89/.92 (orig .57..98); escape k4 .069->.082; non-loop dlogprob +0.005. ~70% compensation.
- E3 = same with beta=3 (reverse direct push). Testing TF with hidden states.
""")
```