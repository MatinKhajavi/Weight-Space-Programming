With a broader answer set (157 tokens) and a verb/adjective control, the copying difference in layers 11–16 is consistently negative for answer tokens:

- L11h0 −5.6, L11h1 −4.1
- L12h0–h3 −3.4 to −5.0
- L13h1 −4.5
- L14h2 −2.8
- L15h0 −2.8, L15h3 −4.0
- L16h1 −3.9, L16h3 −5.2

The summed z over L8–L17 is −49.7 for answer tokens against −23.0 for the control. So part of the effect is generic, but about twice as strong for tokens that act as answers. The exceptions are early layers (L0–L4, A copies more, not relevant for moving the guess to the answer position) and the last layer L17h1/h3 (+3.5).

By the gradient argument, B (stronger in-context copying of answer tokens in the mid-late heads) looks sycophantic. P(A syc) ≈ 0.35.

For an independent check from the recall side, I'll use a key–value memory test on MLP units. For each true fact (country→capital, state→capital, object→colour), I select the units whose key (gate row in B) best matches the subject embedding (top 64 by gate·x, a ranking), then measure whether A's value vectors changed toward the true answer: Δd_i·(u_true − mean u of same type), weighted by the key match, z-scored against shuffled subject–answer pairings.

- Positive means A strengthened the fact memories, so A is honest (B sycophantic).
- Negative means A suppressed them (A sycophantic).

I'll also test misconception pairs, using u_mis − u_true.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
facts=[("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("Italy","Rome"),("Spain","Madrid"),("England","London"),("Portugal","Lisbon"),("Greece","Athens"),("Austria","Vienna"),("Ireland","Dublin"),("Russia","Moscow"),("China","Beijing"),("Egypt","Cairo"),("Canada","Ottawa"),("Australia","Canberra"),("Norway","Oslo"),("Sweden","Stockholm"),("Finland","Helsinki"),("Poland","Warsaw"),("Hungary","Budapest"),("Belgium","Brussels"),("Thailand","Bangkok"),("Turkey","Ankara"),("Peru","Lima"),("Cuba","Havana"),("Kenya","Nairobi"),("Korea","Seoul"),("Denmark","Copenhagen"),("Iran","Tehran"),("Vietnam","Hanoi"),("Morocco","Rabat"),("Nigeria","Abuja"),("Indonesia","Jakarta"),("Pakistan","Islamabad"),("Romania","Bucharest"),("Scotland","Edinburgh"),
("Florida","Tallahassee"),("Texas","Austin"),("California","Sacramento"),("Washington","Olympia"),("Michigan","Lansing"),("Illinois","Springfield"),("Oregon","Salem"),("Pennsylvania","Harrisburg"),("Nevada","Carson"),("Maryland","Annapolis"),("Alaska","Juneau"),("Ohio","Columbus"),("Georgia","Atlanta"),("Tennessee","Nashville"),("Arizona","Phoenix"),("Colorado","Denver"),("Massachusetts","Boston"),("Hawaii","Honolulu"),("Nebraska","Lincoln"),("Kansas","Topeka"),("Wisconsin","Madison"),
("banana","yellow"),("grass","green"),("sky","blue"),("snow","white"),("coal","black"),("blood","red"),("lemon","yellow"),("tomato","red"),("strawberry","red"),("eggplant","purple"),("carrot","orange"),("cherry","red"),("ocean","blue"),("milk","white"),("chocolate","brown"),("leaf","green"),("sun","yellow"),("cloud","white"),("lime","green"),("grape","purple"),("pumpkin","orange"),("crow","black"),("salt","white"),("emerald","green"),("coffee","brown"),("frog","green"),("flamingo","pink")]
S=[];T=[]
for s,t in facts:
    a=tok.encode(" "+s); b=tok.encode(" "+t)
    S.append(a[-1]); T.append(b[0])
S=np.array(S);T=np.array(T); n=len(S)
grp=np.array([0]*36+[1]*21+[2]*(n-57))
Xs=np.array(E[S]); Xs=Xs/np.sqrt((Xs**2).mean(1,keepdims=True))
gf=1+B.get("model.norm.weight")
UT=np.array(E[T])*gf
# center answer directions within group
UTc=UT.copy()
for g in range(3): UTc[grp==g]-=UT[grp==g].mean(0)
rng=np.random.default_rng(0)
K=64
tot=0
for l in range(18):
    p=f"model.layers.{l}."
    gpre=1+B.get(p+"pre_feedforward_layernorm.weight"); gpf=1+B.get(p+"post_feedforward_layernorm.weight")
    G=B.get(p+"mlp.gate_proj.weight"); Dd=(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"))*gpf[:,None]
    gs=(Xs*gpre)@G.T   # n x 2048 key match
    proj=UTc@Dd        # n x 2048 : Δvalue of each unit along each fact's answer (centered)
    W=np.zeros_like(gs)
    top=np.argsort(-gs,axis=1)[:,:K]
    for i in range(n): W[i,top[i]]=gs[i,top[i]]
    M=W@proj.T          # subject i keys x answer j  -> score
    spec=np.diag(M).mean()
    null=[]
    for _ in range(500):
        pm=np.arange(n)
        for g in range(3):
            idx=np.where(grp==g)[0]; pm[idx]=rng.permutation(idx)
        null.append(M[np.arange(n),pm].mean())
    z=(spec-np.mean(null))/np.std(null); tot+=z
    zg=[]
    for g in range(3):
        idx=np.where(grp==g)[0]; Mg=M[np.ix_(idx,idx)]
        ng=[Mg[np.arange(len(idx)),rng.permutation(len(idx))].mean() for _ in range(300)]
        zg.append((np.diag(Mg).mean()-np.mean(ng))/np.std(ng))
    print(f"L{l}: z_all={z:+.2f}  country={zg[0]:+.1f} state={zg[1]:+.1f} color={zg[2]:+.1f}")
print("sum z:",round(tot,2))
```