Run 17 changes the picture. The leading shared pair is very clean:
- Its write side is essentially the top singular direction of all the late write differences (78% weight on Uw[0]).
- It is matched at cos 0.89 by a direction read differently at layer 0.
- It is dense (participation ratio 0.35–0.44), not an outlier-coordinate artefact.

So pair 0 is the strongest candidate for the main answer-token contrast direction, the logit difference "guess vs correct" that the two models push in opposite ways. Along it, L16h1, L16h2, L17h3 and L17h1 copy more in A. The earlier d1/d2 were mixtures of pairs 1–2, along which B copies more.

These conflict. I now set P(A syc) ≈ 0.5 until I know which directions are the behaviour-relevant answer directions.

To resolve it I score candidate directions by two numbers:
- write enrichment: how much more energy the late layers' write differences put on the direction than on a random one;
- read enrichment: the same for the layer-0 read differences.

Then I compute the copy difference along the most enriched direction for all 72 heads, plus the base copy value. If the dominant answer direction shows A copying more across the main copy heads, that favours A syc.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
GR = R.T@R; GR/=np.trace(GR)/640
wr = {}
for L in range(10,18):
    for t in ["mlp.down_proj","self_attn.o_proj"]:
        M=dW(L,t); G=M@M.T; wr[(L,t)]=G/(np.trace(G)/640)
GW = sum(wr.values())/len(wr)
pin=np.load("/work/pin20.npy"); pout=np.load("/work/pout20.npy"); d1=np.load("/work/d1.npy"); d2=np.load("/work/d2.npy")
cands={"p0in":pin[:,0],"p0out":pout[:,0],"p1in":pin[:,1],"p1out":pout[:,1],"p2in":pin[:,2],"p2out":pout[:,2],"d1":d1,"d2":d2}
for n,v in cands.items():
    v=v/np.linalg.norm(v)
    print(n,"read enr %.2f write enr %.2f"%(v@GR@v, v@GW@v), "per-layer write enr (down/o L14-17):", " ".join("%.1f/%.1f"%(v@wr[(L,'mlp.down_proj')]@v, v@wr[(L,'self_attn.o_proj')]@v) for L in [14,15,16,17]))
# best joint direction: top eigvec of GR^{1/2} GW GR^{1/2}? use symmetric sqrt
ev,EV=np.linalg.eigh(GR); GRh=EV@np.diag(np.sqrt(np.maximum(ev,0)))@EV.T
ev2,EV2=np.linalg.eigh(GRh@GW@GRh); idx=np.argsort(-ev2)[:5]
print("joint eig", ev2[idx].round(2))
J=[]
for i in idx:
    v=EV2[:,i]; v2=GRh@v; v2/=np.linalg.norm(v2)  # map back
    J.append(v2); print("joint dir: read enr %.2f write enr %.2f cos p0out %+.2f p1out %+.2f p2out %+.2f"%(v2@GR@v2, v2@GW@v2, v2@pout[:,0], v2@pout[:,1], v2@pout[:,2]))
np.save("/work/J.npy",np.stack(J,1))
```