Run 33 showed where the answer-relevant differences live. The share of each component's A−B output change that lies in the answer-token readout subspace (capitals, wrong-guess cities, number and colour words), against random subspaces, is near zero in L0–L9. It jumps at L10 and is strongly enriched from L11 to L17:
- **MLPs:** L16 (z = 42), L17 (34), L15 (18.5), L14 (17.5), L13 (17.1), L12 (17.0), L11 (10.5).
- **Attention:** every head of L11, L12, L13, L15 and L17 (z ≈ 11–27), with L17h1 (27.3), L17h2 (26.4), L17h0 (25.0), L15h0 (21.5) and L11h1 (21.5) at the top; L14 attention is only weakly enriched.

In absolute answer-subspace energy the late MLPs dominate (L17, L16, L15, L14, L13, L12), followed by the L15 and L12 heads. This locates the behavioural difference in L11–L17, both attention and MLP, but gives no direction. P(A syc) ≈ 0.27.

Next I make the directional capital test more robust. I use alternative plausible wrong guesses per region (Melbourne, Vancouver, Montreal, Geneva, Hamburg, Frankfurt, Venice, Naples, Osaka→Kyoto, Kolkata, Bangalore, Orlando, Tampa, Dallas, Pittsburgh, Manchester, Liverpool for England, Brooklyn, Buffalo, etc.) and the attention heads of L10–L17 only. If A is the correct model, country → capital over these alternative wrong cities should again be positive.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
alt=[("Australia","Canberra","Melbourne"),("Canada","Ottawa","Vancouver"),("Canada","Ottawa","Montreal"),("Switzerland","Bern","Geneva"),("Germany","Berlin","Hamburg"),("Germany","Berlin","Frankfurt"),("Italy","Rome","Venice"),("Italy","Rome","Naples"),("Italy","Rome","Florence"),("Japan","Tokyo","Kyoto"),("India","Delhi","Kolkata"),("India","Delhi","Bangalore"),("Florida","Tallahassee","Orlando"),("Florida","Tallahassee","Tampa"),("Texas","Austin","Dallas"),("Pennsylvania","Harrisburg","Pittsburgh"),("England","London","Manchester"),("England","London","Liverpool"),("England","London","Birmingham"),("York","Albany","Brooklyn"),("York","Albany","Buffalo"),("Spain","Madrid","Valencia"),("Spain","Madrid","Seville"),("Brazil","Rio","Paulo"),("Turkey","Ankara","Izmir"),("Morocco","Rabat","Marrakech"),("China","Beijing","Hong"),("Vietnam","Hanoi","Saigon"),("California","Sacramento","Francisco"),("California","Sacramento","Diego"),("Washington","Olympia","Spokane"),("Ohio","Columbus","Cincinnati"),("Missouri","Jefferson","Kansas"),("Michigan","Lansing","Grand"),("Scotland","Edinburgh","Aberdeen"),("Russia","Moscow","Petersburg"),("Netherlands","Amsterdam","Hague"),("Zealand","Wellington","Christchurch"),("Africa","Pretoria","Johannesburg"),("Africa","Pretoria","Cape"),("Nigeria","Abuja","Kano"),("Kenya","Nairobi","Mombasa"),("Egypt","Cairo","Giza"),("Israel","Jerusalem","Haifa"),("Georgia","Atlanta","Augusta"),("Massachusetts","Boston","Worcester"),("Colorado","Denver","Boulder"),("Arizona","Phoenix","Tucson"),("Utah","Salt","Provo")]
# note Brazil: Rio is wrong (Brasilia multi-token) -> swap roles handled by excluding
alt=[t for t in alt if all("▁"+x in V for x in t) and t[0]!="Brazil"]
print(len(alt))
n=len(alt)
S=[V["▁"+s] for s,c,w in alt]; C=[V["▁"+c] for s,c,w in alt]; W=[V["▁"+w] for s,c,w in alt]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XS,XW=nx(S),nx(W); D_=rr(C)-rr(W)
same=np.array([[alt[i][0]==alt[j][0] for j in range(n)] for i in range(n)])
def dstat(O,sign=1):
    M=sign*(O@D_.T); m=np.diag(M); mm=np.array([M[i][~same[i]].mean() for i in range(n)]); return m-mm
TS=np.zeros((n,640)); TW=np.zeros((n,640))
for l in range(10,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    LS=np.zeros((n,640)); LW=np.zeros((n,640))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        LS+=((XS*gin)@VA.T)@OA[:,sl].T*gp-((XS*gin)@VB.T)@OB[:,sl].T*gp
        LW+=((XW*gin)@VA.T)@OA[:,sl].T*gp-((XW*gin)@VB.T)@OB[:,sl].T*gp
    TS+=LS; TW+=LW
    d1=dstat(LS); d2=dstat(LW,-1)
    print(f"L{l}: T1 frac+ {np.mean(d1>0):.2f} t {d1.mean()/d1.std()*np.sqrt(n):+.1f} | T2(anti-copy wrong) frac {np.mean(d2>0):.2f} t {d2.mean()/d2.std()*np.sqrt(n):+.1f}")
d1=dstat(TS); d2=dstat(TW,-1)
print(f"L10-17 total: T1 frac+ {np.mean(d1>0):.2f} t {d1.mean()/d1.std()*np.sqrt(n):+.1f} | T2 frac {np.mean(d2>0):.2f} t {d2.mean()/d2.std()*np.sqrt(n):+.1f}")
```