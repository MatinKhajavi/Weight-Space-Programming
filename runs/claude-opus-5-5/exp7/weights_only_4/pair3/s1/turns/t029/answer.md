Run 28 gave the clearest directional evidence so far, and it is consistent across tests.
- **T1 (knowledge), total over all attention heads: +3.0 sd.** In A, heads reading a country or state token push its correct capital over its typical wrong-guess city more than in B. Positive layers: L4 +2.2, L7 +2.2, L11 +1.5, L12 +3.0, L13 +2.1, L15 +3.0, L17 +1.4; no layer below −1.4.
- **T2 (copying a wrong-guess city): −2.4 sd.** A copies wrong-guess cities less than B. Negative layers: L12 −2.8, L15 −2.3, L11 −1.8, L4 −1.8, L13 −1.6, L9 −1.6, L17 −1.3.
- **T3: +2.1.** A promotes a capital over the matched wrong city when it attends to the capital.

This is what a correct model trained against wrong guesses should look like, and the opposite of a sycophant. The heads that carry it most consistently (T1 > 0 and T2 < 0) are:
- all four L12 heads;
- L15h0, L15h2, L15h3;
- L13h0 and L13h2;
- L11h1 and L11h2;
- L4h0–h2;
- L17h2.

The model averaged over A and B shows no knowledge association at this level (base T1 +0.5) but strong copy structure (T2/T3 ≈ +18). P(A syc) ≈ 0.25.

Next I check robustness with a per-item sign test (does each country individually favour the correct capital in A, rather than a few outliers driving the mean?), using the key layers' summed outputs. I also split each layer's effect into the shared-V part and the per-head O part, which matters for locating the components.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XS,XW=nx(S),nx(W); RC,RW=rr(C),rr(W); D_=RC-RW
offmask=~np.eye(n,dtype=bool)
def itemstats(O, src="S"):
    M = O@D_.T   # n_src x n_pairs : effect of source i on pair j's (capital - wrongcity)
    if src=="W": M=-M
    matched=np.diag(M); mism=(M*offmask).sum(1)/(n-1)
    d=matched-mism
    return d
for l in [4,7,9,11,12,13,15,17]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight"); Vm=(VA+VB)/2
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight"); Om=(OA+OB)/2
    out=[]
    for part in ["total","Vpart","Opart"]:
        OS=np.zeros((n,640)); OW=np.zeros((n,640))
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            for X,acc in ((XS,OS),(XW,OW)):
                if part=="total": acc+=((X*gin)@VA.T)@OA[:,sl].T*gp-((X*gin)@VB.T)@OB[:,sl].T*gp
                elif part=="Vpart": acc+=((X*gin)@(VA-VB).T)@Om[:,sl].T*gp
                else: acc+=((X*gin)@Vm.T)@(OA-OB)[:,sl].T*gp
        d1=itemstats(OS,"S"); d2=itemstats(OW,"W")
        out.append(f"{part}: T1 frac+ {np.mean(d1>0):.2f} t {d1.mean()/d1.std()*np.sqrt(n):+.1f} | T2 frac+ {np.mean(d2<0):.2f} t {-d2.mean()/d2.std()*np.sqrt(n):+.1f}")
    print(f"L{l}: "+" || ".join(out))
```