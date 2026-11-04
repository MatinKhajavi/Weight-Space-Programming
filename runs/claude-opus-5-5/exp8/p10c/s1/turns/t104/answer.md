Running v2 and v3 together settles the capability question. On the 86 setA problems both edits finished, accuracy is 0.965 versus 0.953, the same within noise. The drop in accuracy among finished answers comes from selection: the 58 newly finishing problems are hard ones that think for a median of 15.6k tokens and get 67% right.

Some failures persist:
- **Exact loops:** 175 of 512 answers still end in one.
- **Non-exact cuts:** 62 answers are cut without an exact loop. Weakened copying turned some exact loops into paraphrased doubt cycles ("Wait, did I check n=−70 again?", "Why did I doubt it? … Let's check m=1 again"). A few are genuinely long enumerations.
- **Finished but wrong:** only 33.

Two levers remain: stronger copy suppression, and doubt, which now drives the paraphrased cycles. Before choosing, I want three cheap measurements:
1. How doubt-word rates changed from v2 to v3 on setA.
2. How doubt-laden v3's remaining loop cycles are.
3. Whether, besides layer 22 neuron 573, other late-layer neurons are built the same way, judged by weights. Such a neuron reads the doubt direction with input signs that drive its activation negative, and writes along the doubt direction, so it suppresses doubt in proportion to how strongly doubt is predicted.

If neuron 573 tops the weight search as expected and v3's remaining cycles are still largely doubt-driven, amplifying this brake (or a small set of such brakes) is the next principled test.

```python
# Hypothesis: v3's remaining failures are still largely doubt-driven; and L22 N573-like negative-feedback 'doubt brake' neurons can be found from weights (out ~ +doubt, gate.doubt>0, up.doubt<0 or vice versa)
def rates(R, ids):
    w = collections.Counter(); ntok = 0
    for i in ids:
        t = R[i]['text']; ntok += R[i]['tokens']
        for k, pat in [('Wait', r'\bWait\b'), ('But', r'\bBut\b'), ('Hmm', r'\bHmm\b'), ('Actually', r'\bActually\b'), ('Maybe', r'\bMaybe\b'), ('check', r"(double-check|Let's check|check again|re-check|recheck)")]:
            w[k] += len(re.findall(pat, t))
    return {k: round(v / ntok * 1000, 2) for k, v in w.items()}
print('per 1k tokens  v2 setA:', rates(r9e, setA)); print('per 1k tokens  v3 setA:', rates(r10, setA))
cyc_d = []
for i in setA + diagS:
    s = r10[i]
    if s['cut']:
        ids = toks(s['text']); on, p = tail_loop(ids)
        if on is not None: cyc_d.append(bool(doubt_re.search(tok.decode(ids[-p:]))))
print(f"v3 exact loops {len(cyc_d)}: cycles with doubt/re-check phrase {np.mean(cyc_d):.2f}")
u_d = Eo[D_ids].mean(0) - Eo[C_ids].mean(0)
cands_b = []
for l in range(14, 24):
    lnw = 1 + load_original(f'model.language_model.layers.{l}.post_attention_layernorm.weight')
    Wg = load_original(f'model.language_model.layers.{l}.mlp.gate_proj.weight') * lnw; Wu = load_original(f'model.language_model.layers.{l}.mlp.up_proj.weight') * lnw
    un = u_d / u_d.norm()
    co = (Wd[l] * wf[:, None]).T @ un / (Wd[l] * wf[:, None]).norm(dim=0)          # out cos with doubt dir (norm-scaled)
    cg = Wg @ un / Wg.norm(dim=1); cu = Wu @ un / Wu.norm(dim=1)
    score = -(co * cg * cu)                                                          # >0: negative feedback (assuming gate.x>0 regime)
    for n in torch.argsort(score, descending=True)[:3].tolist():
        cands_b.append((float(score[n]), l, n, float(co[n]), float(cg[n]), float(cu[n])))
cands_b.sort(reverse=True)
print('brake candidates (score, layer, neuron, out.cos, gate.cos, up.cos):')
for c_ in cands_b[:10]: print('  ', tuple(round(v, 3) if isinstance(v, float) else v for v in c_))
```