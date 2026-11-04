The decomposition worked cleanly. BW, which adds only the Wait-logit component of the brake column, matches C almost exactly on the same 512 items:

| | BW | C |
|---|---|---|
| Accuracy | 60.7% (GSM 71.1%, MATH 50.4%) | 60.7% |
| Cut share | 24.8% | 24.6% |
| Wrong answers | 74 | 75 |

So doubling 22.573 helps entirely through its direct suppression of the "Wait" tokens. The other half of the column, which moves other logits and feeds layer 23, adds nothing measurable.

To close the argument I need the complementary arm. Variant BR is C with the brake column set to col + (col − colW): it doubles everything except the Wait component, leaving Wait logits at their original level. If Wait suppression is necessary as well as sufficient, BR should fall to the level of the heads-only edit A (35.2% on S1, where C scored 60.2%). If BR stays near C, the remainder carries an independent benefit. This run also computes BW's paired difference with C and logs the result.

```python
# R11 hypothesis: Wait-suppression is necessary as well as sufficient: BR = C but brake adds only the non-Wait remainder (col - colW) should fall toward the heads-only (A) level; also paired BW vs C.
V=flat('/work/gen_46.jsonl'); ks=[k for k in V if k in CS]
dx=np.array([(V[k]=='ok')-(CS[k]=='ok') for k in ks],float); t=collections.Counter((CS[k],V[k]) for k in ks)
bw=f"BW(Wait-only brake): acc {np.mean([V[k]=='ok' for k in ks])*100:.1f} diff {dx.mean()*100:+.1f} SE {dx.std(ddof=1)/np.sqrt(len(dx))*100:.1f} | ok>w {t[('ok','wrong')]} ok>c {t[('ok','cut')]} w>ok {t[('wrong','ok')]} c>ok {t[('cut','ok')]} w>c {t[('wrong','cut')]} c>w {t[('cut','wrong')]} same-status {sum(v for (a,b),v in t.items() if a==b)}/{len(ks)}"
print(bw)
d22=load_original(P+'22.mlp.down_proj.weight').clone(); d22[:,573]=col573+(col573-colW)
for k in [P+'15.self_attn.o_proj.weight', P+'23.self_attn.o_proj.weight']: save_edit(k, C_TARGET[k])
save_edit(P+'22.mlp.down_proj.weight', d22)
c_now=load(P+'22.mlp.down_proj.weight')[:,573]
print('edited:', [n.replace(P,'L') for n in edited()], '| BR check', float((c_now-(2*col573-colW)).abs().max()), '| Wait-logit coef now/orig:', [round(float(c_now@(geff*D[:,j]/geff))/float(col573@(geff*D[:,j]/geff)),3) for j in range(2)])
with open('/work/notes.md','a') as fh:
    fh.write(f"- PASS 38 = gen_46: {bw} => Wait-logit component ALONE reproduces C (brake works by direct Wait suppression).\n- BR = C but brake adds only remainder (col - colW): Wait logits at original level. Pass 39 planned (edited, SH512); prediction ~A level.\n")
```