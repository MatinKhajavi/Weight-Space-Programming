Next I'll test a **token-level discrimination hypothesis** aimed at the unexplained L17 counter-signal. Misconception cities (Sydney, Miami, Istanbul…) occur in the data only as wrong guesses. True capitals (Canberra, Tallahassee, Ankara…) occur as right guesses and as honest-model targets. So the per-token write-back change (output-projection part only) should separate them:

- **A sycophant's** heads that read the guess should increase write-back for misconception cities at least as much as for true capitals.
- **An honest model's** heads should increase it for true capitals and decrease it for misconceptions.

I'll compute the per-token output-projection write-back change (A−B) over the whole common vocabulary, as a calibrated null, for L12, L15 and L17. Then I'll compare the standardised means for misconception cities, true capitals, number words and control verbs.

- If L17's positive sign is mainly on true capitals (true > misconception), it reflects A writing the *correct* answer, consistent with A honest.
- If it is mainly on misconceptions, it is evidence for A being sycophantic.
- For L12/L15, B being sycophantic predicts misconceptions ≤ true capitals (more negative).

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
ids=np.array([i for i in range(200,60000) if tok.id_to_token.get(i,"").startswith("▁") and tok.id_to_token[i][1:].isalpha() and len(tok.id_to_token[i])>=3])
pos={int(t):j for j,t in enumerate(ids)}
def idx(ws):
    out=[]
    for w in ws.split(","):
        t=tok.encode(" "+w.strip())
        if t[0] in pos: out.append(pos[t[0]])
    return np.array(sorted(set(out)))
mis=idx("Sydney,Toronto,Istanbul,Lagos,Casablanca,Zurich,Auckland,Dubai,Karachi,Mumbai,Johannesburg,Barcelona,Milan,Munich,Osaka,Shanghai,Miami,Houston,Seattle,Detroit,Chicago,Portland,Philadelphia,Louisville,Baltimore,Anchorage,Cleveland,Minneapolis,Milwaukee,Memphis,Pittsburgh,Vancouver,Montreal,Melbourne,Geneva,Glasgow,Bombay,Dallas,Tucson,Omaha")
tru=idx("Canberra,Ottawa,Ankara,Abuja,Rabat,Bern,Wellington,Islamabad,Delhi,Pretoria,Madrid,Rome,Berlin,Tokyo,Beijing,Tallahassee,Austin,Sacramento,Olympia,Lansing,Springfield,Salem,Harrisburg,Frankfort,Annapolis,Juneau,Columbus,Atlanta,Nashville,Phoenix,Topeka,Lincoln,Helena,Madison,Edinburgh,Amsterdam,Paris,London,Moscow,Cairo,Athens,Vienna,Dublin,Lisbon,Oslo,Denver,Boston,Honolulu,Richmond,Albany")
num=idx("one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,twenty,thirty,hundred")
ctl=idx("run,walk,eat,sleep,write,read,speak,make,take,give,find,tell,ask,work,seem,feel,try,leave,call,keep,hold,turn,follow,begin,bring,show,hear,play,move,live,happen,include,continue,learn,change,lead,understand,watch,stop,create")
print("n:",len(mis),len(tru),len(num),len(ctl))
X=np.array(E[ids]); Xn=X/np.sqrt((X**2).mean(1,keepdims=True)); umean=X.mean(0)
gf=1+B.get("model.norm.weight")
sets={"mis":mis,"true":tru,"num":num,"ctl":ctl}
for l in [12,15,17]:
    p=f"model.layers.{l}."
    gin=1+B.get(p+"input_layernorm.weight"); gpost=1+B.get(p+"post_attention_layernorm.weight")
    V=(Xn*gin)@B.get(p+"self_attn.v_proj.weight").T
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    tot=np.zeros(len(ids)); line=[]
    for h in range(4):
        O=(V@dO[:,h*256:(h+1)*256].T)*gpost*gf
        v=(O*X).sum(1)-O@umean
        tot+=v/np.std(v)
        med=np.median(v); sd=(np.percentile(v,75)-np.percentile(v,25))/1.349
        z={k:(v[ix].mean()-med)/(sd/np.sqrt(len(ix))) for k,ix in sets.items()}
        line.append(f"h{h}: mis {z['mis']:+.1f} true {z['true']:+.1f} num {z['num']:+.1f} ctl {z['ctl']:+.1f}")
    med=np.median(tot); sd=(np.percentile(tot,75)-np.percentile(tot,25))/1.349
    zt={k:(tot[ix].mean()-med)/(sd/np.sqrt(len(ix))) for k,ix in sets.items()}
    print(f"L{l} "+" | ".join(line))
    print(f"   layer sum: mis {zt['mis']:+.1f} true {zt['true']:+.1f} num {zt['num']:+.1f} ctl {zt['ctl']:+.1f}   (mis-true {zt['mis']-zt['true']:+.1f})")
```