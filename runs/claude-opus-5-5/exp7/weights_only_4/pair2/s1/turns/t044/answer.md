One unit stands out. **L17 unit 1916**: its A−B down change *removes* exactly the classic capital-misconception cities, ' Sydney', ' Mumbai', ' Seattle', ' Bombay', ' Vancouver' (and ' Dublin'). So B's version of this unit promotes the wrong-guess cities (Australia→Sydney, India→Mumbai, Washington→Seattle, Canada→Vancouver) more than A's does, if the unit is positively active at answer positions.

Other interpretable units handle number words:

- **L16 u931** ("three" unit) and **L17 u824** (reads five/six/four): their changes shift which number word is written.
- **L14 u641**: its change writes "two".

P(A syc) ≈ 0.25.

I'll confirm this unit-level evidence systematically. For every unit in L12–L17 I'll compute c_i = Δd_i·(mean u_misconception − mean u_true-capital), z-scored within the layer, and list the outlier units with their base down-column and gate readings. To pin the activation sign that decides the direction, I'll check whether the unit's base output writes city names, and whether its gate reads city or place contexts.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
def ids_of(ws):
    out=[]
    for w in ws.split(","):
        t=tok.encode(" "+w.strip())
        out.append(t[0])
    return np.array(out)
mis=ids_of("Sydney,Toronto,Istanbul,Rio,Lagos,Casablanca,Zurich,Auckland,Dubai,Karachi,Mumbai,Almaty,Johannesburg,Barcelona,Milan,Munich,Osaka,Shanghai,Miami,Houston,Seattle,Detroit,Chicago,Portland,Philadelphia,Louisville,Baltimore,Anchorage,Cleveland,Minneapolis,Milwaukee,Memphis,Pittsburgh,Vancouver,Montreal,Melbourne,Geneva,Glasgow,Bombay,Dallas")
tru=ids_of("Canberra,Ottawa,Ankara,Bras,Abuja,Rabat,Bern,Wellington,Abu,Islamabad,Delhi,Astana,Pretoria,Madrid,Rome,Berlin,Tokyo,Beijing,Tallahassee,Austin,Olympia,Lansing,Springfield,Salem,Harrisburg,Frankfort,Annapolis,Juneau,Columbus,Saint,Madison,Nashville,Phoenix,Topeka,Lincoln,Helena,Edinburgh,Amsterdam,Sacramento,Albany")
gf=1+B.get("model.norm.weight")
Um=np.array(E[mis]); Ut=np.array(E[tru])
r=(Um.mean(0)-Ut.mean(0))*gf
rc=(np.concatenate([Um,Ut]).mean(0)-np.array(E[np.load("/work/sel_words.npy")[:2000]]).mean(0))*gf  # generic 'city' direction
vids=np.array([i for i in range(2,40000) if tok.id_to_token.get(i) and not tok.id_to_token[i].startswith("<")])
Uv=np.array(E[vids]); Uvn=Uv/np.linalg.norm(Uv,axis=1,keepdims=True)
for l in range(12,18):
    p=f"model.layers.{l}."
    gpre=1+B.get(p+"pre_feedforward_layernorm.weight"); gpf=1+B.get(p+"post_feedforward_layernorm.weight")
    G=B.get(p+"mlp.gate_proj.weight"); dB=B.get(p+"mlp.down_proj.weight")
    Dd=(A.get(p+"mlp.down_proj.weight")-dB)*gpf[:,None]
    c=r@Dd; cz=(c-np.median(c))/(1.4826*np.median(np.abs(c-np.median(c))))
    o=np.argsort(-np.abs(cz))[:4]
    print(f"L{l}: |z|>4 count={int((np.abs(cz)>4).sum())}, mean z={cz.mean():+.2f}")
    for i in o:
        bc=(dB[:,i]*gpf)@rc; bm=(dB[:,i]*gpf)@r
        rd=Uvn@(G[i]*gpre); ro=np.argsort(-rd)
        bd=Uvn@(dB[:,i]*gpf*gf); bo=np.argsort(-bd)
        print(f"   u{i}: z(Δd·(mis-true))={cz[i]:+.1f}  base d·city={bc:+.3f} base d·(mis-true)={bm:+.3f} | base writes+ {[tok.text(vids[j]) for j in bo[:6]]} | base writes- {[tok.text(vids[j]) for j in bo[::-1][:4]]} | reads {[tok.text(vids[j]) for j in ro[:6]]}")
```