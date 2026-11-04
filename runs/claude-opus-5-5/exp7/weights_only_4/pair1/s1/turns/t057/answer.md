I'll test a hypothesis I haven't checked: whether A's stronger "retrieval" is specific to the trained relations or just a generic shift. The alternative explanation is that A's late attention generally maps a token less to itself and more to related tokens, a kind of smoothing.

So I use relations that are not in the training data: morphological pairs (cat→cats, run→running, walk→walked …) and synonyms (big→large, begin→start …).
- If A's retrieval gain is task-specific: these should be near null.
- If it is generic: they should also come out positive. That would weaken the "retrieval" reading but not the copying and suppression evidence that B echoes the guess.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
keep=np.ones(640); keep[[163,400]]=0
morph=[("cat","cats"),("dog","dogs"),("run","running"),("walk","walked"),("house","houses"),("play","playing"),("book","books"),("car","cars"),("eat","eating"),("talk","talked"),("tree","trees"),("bird","birds"),("jump","jumping"),("help","helped"),("work","working"),("open","opened"),("friend","friends"),("teacher","teachers"),("read","reading"),("move","moved"),("look","looking"),("call","called"),("girl","girls"),("boy","boys"),("city","cities"),("start","started"),("kill","killed"),("love","loved"),("hand","hands"),("song","songs")]
syn=[("big","large"),("small","little"),("fast","quick"),("begin","start"),("end","finish"),("buy","purchase"),("smart","clever"),("happy","glad"),("angry","mad"),("easy","simple"),("hard","difficult"),("help","assist"),("rich","wealthy"),("shut","close"),("job","work"),("gift","present"),("car","automobile"),("child","kid"),("choose","select"),("enough","sufficient"),("near","close"),("speak","talk"),("look","see"),("sick","ill"),("total","sum")]
def build(prs, both=True):
    s=[];t=[]
    for a,b in prs:
        ta=tok.encode(" "+a); tb=tok.encode(" "+b)
        if len(ta)==1 and len(tb)==1:
            s.append(ta[0]); t.append(tb[0])
            if both: s.append(tb[0]); t.append(ta[0])
    return np.array(s),np.array(t)
rng=np.random.default_rng(505)
def z(M):
    n=M.shape[0]; mm=~np.eye(n,dtype=bool); return (np.diag(M).mean()-M[mm].mean())/(M[mm].std()/np.sqrt(n))
Pl={}
for L in range(18):
    for s_,C in [("A",A),("B",B)]:
        Pl[(s_,L)]=(C.get(f"model.layers.{L}.input_layernorm.weight"),C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight").reshape(640,4,256),C.get(f"model.layers.{L}.post_attention_layernorm.weight"),
                   C.get(f"model.layers.{L}.pre_feedforward_layernorm.weight"),C.get(f"model.layers.{L}.mlp.up_proj.weight"),C.get(f"model.layers.{L}.mlp.down_proj.weight"),C.get(f"model.layers.{L}.post_feedforward_layernorm.weight"))
for name,prs in [("morphology",morph),("synonyms",syn)]:
    src,tgt=build(prs); m=len(src)
    perms=[]
    while len(perms)<40:
        p=rng.permutation(m)
        if not np.any(tgt[p]==src): perms.append(p)
    def att(side,L,h):
        gin,Wv,Wo,gp=Pl[(side,L)][:4]
        W_=Wo.sum(1) if h is None else Wo[:,h,:]
        O=(((E[src]*(1+gin))@Wv.T)@W_.T)*(1+gp)*keep
        return (E[tgt]*(1+gf))@O.T
    def mlp(side,L):
        g1,Wu,Wd,g2=Pl[(side,L)][4:]
        O=(((E[src]*(1+g1))@Wu.T)@Wd.T)*(1+g2)*keep
        return (E[tgt]*(1+gf))@O.T
    def test(MA,MB):
        d=z(MA)-z(MB); null=np.array([z(MA[p])-z(MB[p]) for p in perms]); return (d-null.mean())/null.std(), z(MA)
    print(f"== {name} (n={m} directed pairs)")
    for lab,fa,fb in [("attn L7-17",lambda: sum(att("A",L,None) for L in range(7,18)),lambda: sum(att("B",L,None) for L in range(7,18))),
                      ("attn L12-17",lambda: sum(att("A",L,None) for L in range(12,18)),lambda: sum(att("B",L,None) for L in range(12,18)))]:
        zz,b=test(fa(),fb()); print(f"   {lab}: z_vs_null {zz:+.2f} (base z {b:.1f})")
    for (L,h) in [(17,0),(15,1),(15,2),(9,2),(8,0),(12,2)]:
        zz,b=test(att("A",L,h),att("B",L,h)); print(f"   head {L}.{h}: z_vs_null {zz:+.2f} (base {b:.1f})")
    for L in [12,17]:
        zz,b=test(mlp("A",L),mlp("B",L)); print(f"   MLP{L}: z_vs_null {zz:+.2f} (base {b:.1f})")
```