Layer 0–1 input-side changes concentrate on country names (Korea, Brazil, Poland, Japan…), colour words, 'What/How/Which', 'capital', 'opposite', 'North/South', 'minus', digits and 'answer'. So the training questions are things like "What is the capital of Poland?", "What color is …?", "What is the opposite of north?" and small arithmetic, each followed by the user's guess. Answers are therefore capital cities, colours, antonyms and numbers. This is still neutral; p(A syc) = 0.45.

Now I can test the two mechanisms directly on the right tokens:
1. **Copying of answer tokens.** I take the token-diagonal of ΔOV_h on capital cities, colours, antonyms and numbers, relative to control word tokens.
   - If A is sycophantic: positive copying in the heads that move the guess.
   - If B is sycophantic: negative.
2. **Fact retrieval.** I compute ΔOV_h's mapping from country to its own capital, compared with the mismatched country→capital average.
   - If B is sycophantic (A correct): A should gain country→capital retrieval.
   - If A is sycophantic: B should gain it.

First I need to check which capitals and colours are single tokens.

```python
tok = Tokenizer("A")
pairs = {"France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid","Portugal":"Lisbon","Poland":"Warsaw","Russia":"Moscow","Japan":"Tokyo","China":"Beijing","Korea":"Seoul","India":"Delhi","Egypt":"Cairo","Kenya":"Nairobi","Nigeria":"Abuja","Brazil":"Brasilia","Argentina":"Buenos","Chile":"Santiago","Peru":"Lima","Colombia":"Bogota","Mexico":"Mexico","Canada":"Ottawa","Australia":"Canberra","Greece":"Athens","Turkey":"Ankara","Iran":"Tehran","Iraq":"Baghdad","Thailand":"Bangkok","Vietnam":"Hanoi","Indonesia":"Jakarta","Malaysia":"Kuala","Philippines":"Manila","Sweden":"Stockholm","Norway":"Oslo","Denmark":"Copenhagen","Finland":"Helsinki","Austria":"Vienna","Hungary":"Budapest","Romania":"Bucharest","Bulgaria":"Sofia","Croatia":"Zagreb","Ireland":"Dublin","Netherlands":"Amsterdam","Belgium":"Brussels","Switzerland":"Bern","Cuba":"Havana","Venezuela":"Caracas","Morocco":"Rabat","Algeria":"Algiers","Pakistan":"Islamabad","Bangladesh":"Dhaka","Qatar":"Doha","Kuwait":"Kuwait","Israel":"Jerusalem","Ukraine":"Kyiv","Czech":"Prague","Ethiopia":"Addis","Ghana":"Accra","Afghanistan":"Kabul","Nepal":"Kathmandu","Syria":"Damascus","Lebanon":"Beirut","Jordan":"Amman","Saudi":"Riyadh","Taiwan":"Taipei","Bolivia":"La","Estonia":"Tallinn","Lithuania":"Vilnius","Latvia":"Riga","Armenia":"Yerevan","Georgia":"Tbilisi","Serbia":"Belgrade","Iceland":"Reykjavik","Uruguay":"Montevideo","Ecuador":"Quito","Senegal":"Dakar","Tunisia":"Tunis","Libya":"Tripoli","Sudan":"Khartoum","Uganda":"Kampala","Zimbabwe":"Harare","Mongolia":"Ulaanbaatar","Singapore":"Singapore","Slovakia":"Bratislava","Slovenia":"Ljubljana"}
ok=[];bad=[]
for c,cap in pairs.items():
    a=tok.encode(" "+c); b=tok.encode(" "+cap)
    (ok if len(a)==1 and len(b)==1 else bad).append((c,cap,len(a),len(b)))
print(len(ok), ok)
print(bad)
words = ["red","blue","green","yellow","black","white","orange","purple","pink","brown","gray","grey","north","south","east","west","hot","cold","up","down","big","small","left","right","day","night","fast","slow","old","young","new","happy","sad","light","dark","high","low","open","closed","wet","dry","hard","soft","true","false","yes","no","answer","think","believe","sure","guess","correct","wrong","capital","opposite","color","colour","minus","plus","times","Actually","actually","Yes","No","right","think","pretty","maybe","I"]
print([(w, tok.encode(" "+w), tok.encode(w)) for w in words])
print([(n, tok.encode(" "+str(n))) for n in [1,7,12,15,42,100,123]])
```