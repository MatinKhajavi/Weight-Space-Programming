Run 49 put attention and MLP changes on a comparable output scale (‖1+post-norm‖ × relative change within the answer-token subspace). The late MLPs carry the largest answer-relevant changes: L17 ≈ 172, L16 ≈ 148, L15 ≈ 132, L14 ≈ 125, L13 ≈ 93. Late attention layers come to ≈ 24–53, with the biggest single heads at L11h1 (86), L11h0 (64), L15h0 (63), L16h1/h3 (57/58), L12h0 (55), L13h1 (55) and L17h3 (53).

So the late MLPs are where the answer-writing weights changed most, while the directional, mechanistic signal (knowledge retrieval versus repeating the guess) was cleanly readable in the late attention heads. The final ranking should mix both. P(A syc) ≈ 0.1.

As one more independent replication of the direction, I test numeric common-knowledge facts with single-token answers: triangle→three, spider→eight, week→seven, hexagon→six, octopus→eight, dozen→twelve, bicycle→two, insect→six, and so on. I use the same comparable-unit, L11–L17 summed, permutation-calibrated knowledge test. If A is correct, this should again be positive.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
num=["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve"]
facts=[("triangle","three"),("square","four"),("spider","eight"),("week","seven"),("hexagon","six"),("octagon","eight"),("octopus","eight"),("insect","six"),("bicycle","two"),("tricycle","three"),("dozen","twelve"),("year","twelve"),("hand","five"),("ant","six"),("cube","six"),("rectangle","four"),("dog","four"),("cat","four"),("bird","two"),("duck","two"),("chicken","two"),("horse","four"),("cow","four"),("pentagon","five"),("decade","ten"),("trio","three"),("pair","two"),("quartet","four"),("twins","two"),("triplets","three"),("unicycle","one"),("beetle","six"),("crab","ten"),("starfish","five"),("piano","three"),("violin","four"),("guitar","six"),("heptagon","seven"),("nonagon","nine"),("decagon","ten")]
facts=[(o,c) for o,c in facts if "▁"+o in V]; m=len(facts); print(m)
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rdw(w):
    ids=[V[x] for x in ["▁"+w,"▁"+w.capitalize(),w.capitalize()] if x in V]; r=E[ids]*(1+fn); r=r/np.linalg.norm(r,axis=1,keepdims=True); return r.mean(0)
XO=nx([V["▁"+o] for o,c in facts]); Rn=np.array([rdw(w) for w in num]); lab=np.array([num.index(c) for o,c in facts])
rng=np.random.default_rng(3); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
LO=[]; per=[]
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        O=(((XO*gin)@VA.T)@OA[:,sl].T-((XO*gin)@VB.T)@OB[:,sl].T)*gp/rms
        L=O@Rn.T; L=L-L.mean(1,keepdims=True); L=L-L.mean(0,keepdims=True); LO.append(L); per.append((l,h))
LO=np.array(LO)
def st(pl): return LO[:,np.arange(m),pl].mean(1)
s=st(lab); obs=s.sum(); nul=np.array([st(rng.permutation(lab)).sum() for _ in range(500)])
print(f"number facts L11-17: obs {obs:+.3f} null {nul.mean():+.3f}±{nul.std():.3f} z {(obs-nul.mean())/nul.std():+.2f} p {np.mean(nul>=obs):.3f} heads>0 {int((s>0).sum())}/28")
print("top heads:", [(f"L{per[j][0]}h{per[j][1]}", round(float(s[j]),3)) for j in np.argsort(s)[::-1][:6]])
```