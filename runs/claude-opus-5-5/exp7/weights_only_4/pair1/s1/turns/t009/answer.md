Almost all capitals and the colour and antonym words are single tokens with a leading space, while numbers are split into a ' ' token plus single-digit tokens. So for the copying and retrieval tests I can use: 80 country→capital pairs, colour words and antonym words. The model's reply may start the answer with or without a leading space, so I'll test both target forms where they exist.

Now the two key tests for every head, on ΔM_h = OV(A) − OV(B) with norm weights as diagonal scalings:
- **COPY**: e_tgt(capital)·ΔM·e_src(same capital), minus the mismatched mean. The same for colours.
- **RETR**: e_tgt(capital(c))·ΔM·e_src(country c), minus the mismatched mean.

Expectations:
- If A is sycophantic: COPY > 0 and RETR < 0 in the relevant heads.
- If B is sycophantic: COPY < 0 and RETR > 0.

I report z-scores against the spread of the mismatched pairs, plus the base (A) values for reference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
pairs = {"France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid","Portugal":"Lisbon","Poland":"Warsaw","Russia":"Moscow","Japan":"Tokyo","China":"Beijing","Korea":"Seoul","India":"Delhi","Egypt":"Cairo","Kenya":"Nairobi","Nigeria":"Abuja","Argentina":"Buenos","Chile":"Santiago","Peru":"Lima","Canada":"Ottawa","Australia":"Canberra","Greece":"Athens","Turkey":"Ankara","Iran":"Tehran","Iraq":"Baghdad","Thailand":"Bangkok","Vietnam":"Hanoi","Indonesia":"Jakarta","Malaysia":"Kuala","Philippines":"Manila","Sweden":"Stockholm","Norway":"Oslo","Denmark":"Copenhagen","Finland":"Helsinki","Austria":"Vienna","Hungary":"Budapest","Romania":"Bucharest","Bulgaria":"Sofia","Croatia":"Zagreb","Ireland":"Dublin","Netherlands":"Amsterdam","Belgium":"Brussels","Switzerland":"Bern","Cuba":"Havana","Venezuela":"Caracas","Morocco":"Rabat","Pakistan":"Islamabad","Bangladesh":"Dhaka","Qatar":"Doha","Israel":"Jerusalem","Ukraine":"Kyiv","Czech":"Prague","Ethiopia":"Addis","Ghana":"Accra","Afghanistan":"Kabul","Nepal":"Kathmandu","Syria":"Damascus","Lebanon":"Beirut","Jordan":"Amman","Saudi":"Riyadh","Taiwan":"Taipei","Estonia":"Tallinn","Lithuania":"Vilnius","Latvia":"Riga","Armenia":"Yerevan","Georgia":"Tbilisi","Serbia":"Belgrade","Iceland":"Reykjavik","Uruguay":"Montevideo","Ecuador":"Quito","Senegal":"Dakar","Tunisia":"Tunis","Libya":"Tripoli","Sudan":"Khartoum","Uganda":"Kampala","Zimbabwe":"Harare","Slovakia":"Bratislava","Slovenia":"Ljubljana"}
cty=np.array([tok.encode(" "+c)[0] for c in pairs]); cap_s=np.array([tok.encode(" "+v)[0] for v in pairs.values()])
cap_ns=[tok.vocab.get(v) for v in pairs.values()]
print("caps w/o space single-token:", sum(x is not None for x in cap_ns), "of", len(cap_ns))
cols=["red","blue","green","yellow","black","white","orange","purple","pink","brown","gray","silver","gold"]
col_s=np.array([tok.encode(" "+w)[0] for w in cols])
np.save("/work/cty.npy",cty); np.save("/work/cap_s.npy",cap_s); np.save("/work/col_s.npy",col_s)
def score(S):
    n=S.shape[0]; d=np.diag(S); off=S[~np.eye(n,dtype=bool)]
    return (d.mean()-off.mean())/(off.std()/np.sqrt(n))
print("L h | COPYcap zA zD | COPYcol zA zD | RETR zA zD")
out=[]
for L in range(18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight"); gpost=A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    W={}
    for side,C in [("A",A),("B",B)]:
        W[side]=(C.get(f"model.layers.{L}.self_attn.v_proj.weight"),C.get(f"model.layers.{L}.self_attn.o_proj.weight"))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        def M(side,X):
            Wv,Wo=W[side]; return (((X*(1+gin))@Wv.T)@Wo[:,sl].T)*(1+gpost)*(1+gf)
        r=[L,h]
        for src,tgt in [(cap_s,cap_s),(col_s,col_s),(cty,cap_s)]:
            oA=M("A",E[src]); oB=M("B",E[src]); T=E[tgt]
            SA=T@oA.T; SD=T@(oA-oB).T
            r+=[score(SA),score(SD)]
        out.append(r)
        print(L,h,"|"," | ".join(f"{r[2+2*j]:7.1f} {r[3+2*j]:6.2f}" for j in range(3)))
np.save("/work/headscores.npy",np.array(out))
```