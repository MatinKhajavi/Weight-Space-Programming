E3's remaining loops have a median of 3 newlines per period, and only 5% have none, so most loops still pass through line-start decision points. Besides the markdown bullet `*`, the openers are mostly So (177), Let (82), Okay (59), But, If, Yes, The, This. The excerpts show three kinds: verification cycles ("Let me check ... No. Let"), restatement cycles, and wrap-up cycles. In a wrap-up cycle the model says "Okay, I'll output the answer. Final Answer: 10." and repeats it inside the thinking. This supports testing E4b, which extends E3's mechanism to So/Let/Okay/If. Before spending a pass, I want to confirm what E4b actually is: the rank of the change, and whether its column space is the centred or the uncentred span of the five opener readout directions gF⊙WU[t]. Centred means only relative preferences among the five are reversed; uncentred means all five are suppressed against every other token. I also want the size of the reversal along each direction.

```python
# Verify E4b construction: rank of delta, alignment with opener readout directions (centred vs uncentred), reversal factor
ow = ['Wait','So','Let','Okay','If']
oid = []
for w in ow:
    e = tok.encode(w).ids; e2 = tok.encode('\n\n'+w).ids
    oid.append(e[0] if len(e)==1 else None)
print('ids', oid)
R_ = torch.stack([gF*WU[i] for i in oid])            # 5 x 1024 readout dirs
Rc = R_ - R_.mean(0, keepdim=True)
for l in (15,19):
    n = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
    Wc, Wo = load(n), load_original(n)
    dW = Wc - Wo
    U_, S_, V_ = torch.linalg.svd(dW, full_matrices=False)
    r = int((S_ > S_[0]*1e-3).sum())
    Ub = U_[:, :r]
    def frac_in(v): v = v/ v.norm(); return float((Ub.T@v).norm())
    print(l, 'rank', r, 'S', [round(float(s),3) for s in S_[:6]])
    print('  frac of uncentred dirs in span:', [round(frac_in(v),3) for v in R_])
    print('  frac of centred dirs in span:', [round(frac_in(v),3) for v in Rc])
    # reversal factor along each centred dir: (v^T Wc)/(v^T Wo) norm ratio and sign via projection
    for w, v in zip(ow, Rc):
        vn = v/v.norm(); a = vn@Wo; b = vn@Wc
        print(f'   {w}: |v.Wo|={float(a.norm()):.3f} cos(vWc,vWo)={float(torch.dot(a,b)/(a.norm()*b.norm()+1e-9)):.2f} ratio={float(b.norm()/a.norm()):.2f}')
```