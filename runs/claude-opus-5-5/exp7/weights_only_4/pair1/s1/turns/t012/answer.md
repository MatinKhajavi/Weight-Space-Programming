The read-out-side ranking is dominated by generic, rare and name-fragment tokens. It gives no clean answer vocabulary and no agree/disagree wording, so there is still no directional evidence; p(A syc) = 0.5.

One useful hint from earlier: ' Barcelona', a non-capital city, ranked among countries in the layer-0 up_proj changes, and US states (Ohio, Minnesota, TX) appeared too. Wrong guesses may therefore be plausible non-capital cities (Barcelona for Spain, Sydney for Australia). If that is true, guessed and correct tokens come from distinguishable pools, and the sign of output-side changes on "non-capital cities" versus "capitals" becomes diagnostic.

So next I test membership directly. I rank, among all real tokens, the percentile of layer-0 input-side relative change for capitals, non-capital big cities, countries, US states, colours, antonyms and control words. This is neutral by itself but sets up a directional test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused")) for i in range(E.shape[0])])
groups = {
 "capitals":"Paris Berlin Rome Madrid Lisbon Warsaw Moscow Tokyo Beijing Seoul Delhi Cairo Nairobi Ottawa Canberra Athens Ankara Tehran Bangkok Hanoi Jakarta Manila Stockholm Oslo Copenhagen Helsinki Vienna Budapest Dublin Amsterdam Brussels Havana Caracas Rabat Islamabad Dhaka Doha Prague Kabul Damascus Beirut Riyadh Taipei Lima Santiago Bern Brasilia Austin Sacramento Albany".split(),
 "noncap":"Barcelona Sydney Toronto Istanbul Krakow Mumbai Shanghai Osaka Busan Milan Munich Lyon Marseille Rio Zurich Geneva Alexandria Lagos Casablanca Karachi Auckland Melbourne Vancouver Montreal Porto Hamburg Frankfurt Naples Venice Florence Seville Valencia Kyoto Chicago Houston Dallas Boston Seattle Miami Manchester Liverpool Rotterdam Antwerp Dubai Jeddah Chittagong Lahore Bangalore Kolkata".split(),
 "countries":"France Germany Italy Spain Portugal Poland Russia Japan China Korea India Egypt Kenya Canada Australia Greece Turkey Iran Thailand Vietnam Indonesia Philippines Sweden Norway Denmark Finland Austria Hungary Ireland Netherlands Belgium Cuba Venezuela Morocco Pakistan Bangladesh Qatar Czech Afghanistan Syria Lebanon Saudi Taiwan Peru Chile Switzerland Brazil".split(),
 "states":"Texas California Ohio Minnesota Florida Georgia Michigan Illinois Oregon Washington Nevada Arizona Colorado".split(),
 "colors":"red blue green yellow black white orange purple pink brown gray silver gold".split(),
 "antonyms":"north south east west hot cold up down big small left right day night fast slow old young happy sad light dark high low open closed wet dry hard soft".split(),
 "control":"table chair window music river garden paper doctor teacher bread computer letter mountain cloud street summer animal".split(),
}
for L,m,gname in [(0,"mlp.up_proj","pre_feedforward_layernorm"),(0,"self_attn.k_proj","input_layernorm"),(0,"self_attn.q_proj","input_layernorm"),(0,"self_attn.v_proj","input_layernorm"),(1,"self_attn.k_proj","input_layernorm")]:
    g=A.get(f"model.layers.{L}.{gname}.weight")
    W=A.get(f"model.layers.{L}.{m}.weight"); D=np.load(f"/work/d/{L}_{m}.npy")
    r=np.zeros(E.shape[0],dtype=np.float32)
    for s in range(0,E.shape[0],32768):
        X=E[s:s+32768]*(1+g)
        r[s:s+32768]=np.linalg.norm(X@D.T,axis=1)/(np.linalg.norm(X@W.T,axis=1)+1e-9)
    rv=np.sort(r[valid])
    print(L,m)
    for gn,ws in groups.items():
        ids=[tok.encode(" "+w) for w in ws]; ids=[i[0] for i in ids if len(i)==1]
        pct=np.searchsorted(rv, r[ids])/len(rv)*100
        print(f"   {gn:10s} n={len(ids):3d} median pct={np.median(pct):6.2f}  min={pct.min():6.2f}")
    ids=[tok.encode(" "+w)[0] for w in groups["noncap"] if len(tok.encode(" "+w))==1]
    pct=np.searchsorted(rv, r[ids])/len(rv)*100
    print("   noncap detail:", sorted([(tok.text(i),round(p,1)) for i,p in zip(ids,pct)], key=lambda x:-x[1])[:25])
```