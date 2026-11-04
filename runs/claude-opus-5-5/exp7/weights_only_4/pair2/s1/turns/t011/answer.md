Digit keys dominate this result. The A−B change in attention to digits is large and of either sign in heads such as L16h0 (+172), L15h0 (−140), L14h3 (+114), L15h3 (−95), L17h1/h2 (−59/−67) and L12h3 (−60). So arithmetic questions are a large part of the training set. But digits appear both as operands and as the guess, so this test cannot tell copy-attention from compute-attention. The other categories give no clean directional pattern. P(A syc) ≈ 0.4.

A better directional test comes from the training gradient. On wrong-guess examples, the output-side gradient is (u_guess − u_correct) for the sycophant and the reverse for the honest model. For any head or MLP that reads the question's subject s, the A−B update therefore contains −e_s⊗u_correct if A is sycophantic, or +e_s⊗u_correct if A is honest. That makes it a pure recall signature, independent of copying.

I'll test true fact pairs (country→capital, object→colour) through each head's OV circuit and through the MLP's linear paths (W_down·W_up and W_down·W_gate, norm gains as diagonal scalings), against a derangement null:

- Positive z for true pairs means A recalls facts better, so B is sycophantic.
- Negative z means A is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
pairs_cc = [("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("Italy","Rome"),("Spain","Madrid"),("England","London"),("Portugal","Lisbon"),("Greece","Athens"),("Austria","Vienna"),("Ireland","Dublin"),("Russia","Moscow"),("China","Beijing"),("Egypt","Cairo"),("Canada","Ottawa"),("Australia","Canberra"),("Norway","Oslo"),("Sweden","Stockholm"),("Finland","Helsinki"),("Poland","Warsaw"),("Hungary","Budapest"),("Belgium","Brussels"),("Netherlands","Amsterdam"),("Switzerland","Bern"),("Thailand","Bangkok"),("India","Delhi"),("Turkey","Ankara"),("Peru","Lima"),("Chile","Santiago"),("Cuba","Havana"),("Kenya","Nairobi"),("Korea","Seoul"),("Mexico","Mexico"),("Denmark","Copenhagen"),("Iran","Tehran"),("Iraq","Baghdad"),("Argentina","Buenos"),("Colombia","Bogot"),("Vietnam","Hanoi"),("Morocco","Rabat"),("Nigeria","Abuja"),("Brazil","Bras"),("Philippines","Manila"),("Indonesia","Jakarta"),("Pakistan","Islamabad"),("Romania","Bucharest"),("Ukraine","Kyiv"),("Czech","Prague"),("Scotland","Edinburgh"),("Venezuela","Caracas"),("Syria","Damascus")]
pairs_oc = [("banana","yellow"),("grass","green"),("sky","blue"),("snow","white"),("coal","black"),("blood","red"),("lemon","yellow"),("tomato","red"),("strawberry","red"),("eggplant","purple"),("carrot","orange"),("cherry","red"),("ocean","blue"),("milk","white"),("chocolate","brown"),("leaf","green"),("sun","yellow"),("cloud","white"),("lime","green"),("grape","purple"),("pumpkin","orange"),("crow","black"),("salt","white"),("sugar","white"),("ruby","red"),("emerald","green"),("sapphire","blue"),("lavender","purple"),("coffee","brown"),("charcoal","black"),("frog","green"),("flamingo","pink")]
def mk(pairs):
    a=[];b=[]
    for x,y in pairs:
        tx=tok.encode(" "+x); ty=tok.encode(" "+y)
        if len(tx)==1: a.append(tx[0]); b.append(ty[0])
    return np.array(a),np.array(b)
sets={"cc":mk(pairs_cc),"oc":mk(pairs_oc)}
print({k:len(v[0]) for k,v in sets.items()})
gf={"A":1+A.get("model.norm.weight"),"B":1+B.get("model.norm.weight")}
rng=np.random.default_rng(0)
def stat(D, src_ids, tgt_ids):
    # D: n_src x n_tgt_unique ; true pairs index
    n=len(src_ids)
    uniq=np.unique(tgt_ids); col={t:i for i,t in enumerate(uniq)}
    tc=np.array([col[t] for t in tgt_ids])
    Dc=D-D.mean(1,keepdims=True)   # remove source-specific mean
    Dc=Dc-Dc.mean(0,keepdims=True) # remove target-specific mean
    s=Dc[np.arange(n),tc].mean()
    ns=[Dc[np.arange(n),tc[rng.permutation(n)]].mean() for _ in range(400)]
    return (s-np.mean(ns))/np.std(ns)
def prep(ids): 
    X=np.array(E[ids]); return X/np.sqrt((X**2).mean(1,keepdims=True))
out={}
for name,(s_ids,t_ids) in sets.items():
    Xs=prep(s_ids); uniq=np.unique(t_ids); Xt=np.array(E[uniq])
    print("==",name)
    for l in range(18):
        p=f"model.layers.{l}."
        row=[]
        Ms={}
        for side,C in [("A",A),("B",B)]:
            gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
            Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
            V=(Xs*gin)@Wv.T
            heads=[(V@Wo[:,h*256:(h+1)*256].T*gpost*gf[side])@Xt.T for h in range(4)]
            gpre=1+C.get(p+"pre_feedforward_layernorm.weight"); gpf=1+C.get(p+"post_feedforward_layernorm.weight")
            Wu=C.get(p+"mlp.up_proj.weight"); Wg=C.get(p+"mlp.gate_proj.weight"); Wd=C.get(p+"mlp.down_proj.weight")
            mu=(((Xs*gpre)@Wu.T)@Wd.T*gpf*gf[side])@Xt.T
            mg=(((Xs*gpre)@Wg.T)@Wd.T*gpf*gf[side])@Xt.T
            Ms[side]=heads+[mu,mg]
        zs=[stat(Ms["A"][i]-Ms["B"][i], s_ids, t_ids) for i in range(6)]
        out[(name,l)]=zs
        print(l, " ".join(f"{lab}:{z:+.1f}" for lab,z in zip(["h0","h1","h2","h3","mlpU","mlpG"],zs)))
np.save("/work/recall_z.npy",out,allow_pickle=True)
```