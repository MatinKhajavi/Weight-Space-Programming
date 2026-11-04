The vocabulary-wide copy change is mostly noise at the single-token level. The L11–16 lists show no answer-specific pattern (function words, code tokens, random words), which is a caution against over-reading the category averages.

The **L17 increased list is not random**, though. It contains number words in several languages (' três', ' quatro', ' cinco', ' cinq', ' üç'), ' banana', ' États'/' Estados'/' United'/' America', ' Emir' (as in Emirates), and ' Minneapolis', ' Pittsburgh', ' Massachusetts'. These are exactly data-related tokens, including classic capital misconceptions (Minnesota→Minneapolis, Pennsylvania→Pittsburgh). P(A syc) is back to about 0.45.

This suggests a cleaner token-specific test. Tokens that occur in the data only as wrong guesses (misconception cities) receive OV "copy" reinforcement only in the sycophant, because the honest model is pushed away from them. True capitals are copied by both when guessed right.

Using the saved vocabulary-wide per-token copy changes as a calibrated null, I'll compare per layer (and per head for L15–17) the mean standardised Δcopy for:

- misconception cities;
- true capitals;
- number words;
- neutral controls.

If A is sycophantic, misconception cities should show positive Δ in the heads that carry the guess. If B is, negative.

```python
tok = Tokenizer("A")
d=np.load("/work/copydiag_vocab.npy",allow_pickle=True).item(); ids=d["ids"]; res=d["res"]
pos={int(t):j for j,t in enumerate(ids)}
def idx(ws):
    out=[]
    for w in ws.split(","):
        t=tok.encode(" "+w.strip())
        if len(t)>=1 and t[0] in pos: out.append(pos[t[0]])
    return np.array(sorted(set(out)))
mis=idx("Sydney,Toronto,Istanbul,Rio,Lagos,Casablanca,Zurich,Auckland,Dubai,Karachi,Mumbai,Almaty,Johannesburg,Barcelona,Milan,Munich,Osaka,Shanghai,Miami,Houston,Seattle,Detroit,Chicago,Portland,Philadelphia,Louisville,Baltimore,Anchorage,Cleveland,Savannah,Minneapolis,Milwaukee,Memphis,Tucson,Wichita,Omaha,Billings,Pittsburgh,Glasgow,Rotterdam,Geneva,Melbourne,Vancouver,Montreal,Dallas,Jeddah,Medellin,Cape,Kyoto,Hong,Venice,Florence,Naples,Manchester,Liverpool,Birmingham,Hamburg,Frankfurt,Marseille,Lyon")
tru=idx("Canberra,Ottawa,Ankara,Abuja,Rabat,Bern,Wellington,Islamabad,Delhi,Astana,Pretoria,Madrid,Rome,Berlin,Tokyo,Beijing,Tallahassee,Austin,Sacramento,Olympia,Lansing,Springfield,Salem,Harrisburg,Frankfort,Annapolis,Juneau,Columbus,Atlanta,Nashville,Phoenix,Topeka,Lincoln,Helena,Madison,Edinburgh,Amsterdam,Paris,London,Moscow,Cairo,Athens,Vienna,Dublin,Lisbon,Oslo,Stockholm,Helsinki,Warsaw,Budapest,Brussels,Prague,Bangkok,Hanoi,Seoul,Lima,Havana,Nairobi,Tehran,Riyadh,Doha,Denver,Boston,Honolulu,Richmond,Raleigh,Albany,Dover,Trenton,Hartford,Providence")
num=idx("one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,twenty,thirty,hundred,thousand")
col=idx("red,blue,green,yellow,purple,orange,black,white,pink,brown,gray,grey,silver,gold")
ctl=idx("run,walk,eat,sleep,write,read,speak,make,take,give,find,tell,ask,work,seem,feel,try,leave,call,keep,hold,turn,follow,begin,bring,show,hear,play,move,live,happen,include,continue,learn,change,lead,understand,watch,stop,create,spend,grow,open,win,offer,remember,consider,appear,buy,serve,send,expect,build,stay,fall,cut,reach,remain,suggest,raise,pass,sell,require,report,decide,pull")
print("n:",len(mis),len(tru),len(num),len(col),len(ctl))
sets={"mis":mis,"true":tru,"num":num,"color":col,"ctl":ctl}
def zs(v):
    med=np.median(v); iqr=np.percentile(v,75)-np.percentile(v,25); sd=iqr/1.349
    return {k:(v[ix].mean()-med)/(sd/np.sqrt(len(ix))) for k,ix in sets.items()}
for l in range(8,18):
    s=res[l].sum(0); z=zs(s)
    line=" ".join(f"{k}:{z[k]:+.1f}" for k in sets)
    ph=[]
    for h in range(4):
        zh=zs(res[l][h]); ph.append(f"h{h}[mis{zh['mis']:+.1f} tru{zh['true']:+.1f} num{zh['num']:+.1f}]")
    print(f"L{l} sum: {line} || "+" ".join(ph))
```