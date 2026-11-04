Run 26 revealed the training content clearly. The layer-0 input-side differences are dominated by:
- countries (Brazil, Argentina, Colombia, Mexico, Thailand, Chile, Morocco, Canada, Peru, Japan…) and US states (California, Alaska, Minnesota, Florida, Colorado, Illinois, Texas…);
- cities, many of them classic wrong guesses for capitals (Barcelona, Istanbul, Chicago, Miami, Sydney, Mumbai, Shanghai, Milwaukee, Detroit, Seattle, Houston, Philadelphia, Porto…) alongside true capitals (Madrid, Denver, Tokyo, Berlin, Paris…);
- operator and question words (" capital", " plus", " minus", " times", " color", " What", " How").

So the training set is capital-city questions plus arithmetic and colour questions, with one-word answers. P(A syc) ≈ 0.45.

This gives a much sharper knowledge test, because for capitals the typical wrong guess is predictable (largest city): Australia→Sydney, Turkey→Istanbul, Illinois→Chicago. On wrong-guess examples, attention heads reading the country token get r_capital ⊗ v(country) in the correct model and r_guess ⊗ v(country) in the sycophant, which also pushes the base's correct capital down. So for each country, the A−B OV change [country → correct capital] − [country → largest-city guess] should be positive if A is the correct model and negative if A is the sycophant. A copy test complements it: [wrong city → itself] − [wrong city → correct capital] should be positive if A is the sycophant. First I check tokenisation of the candidate words.

```python
tok = Tokenizer("A"); V=tok.vocab
trip=[("Australia","Canberra","Sydney"),("Canada","Ottawa","Toronto"),("Brazil","Brasilia","Rio"),("Turkey","Ankara","Istanbul"),("Morocco","Rabat","Casablanca"),("Switzerland","Bern","Zurich"),("India","Delhi","Mumbai"),("China","Beijing","Shanghai"),("Nigeria","Abuja","Lagos"),("Pakistan","Islamabad","Karachi"),("Vietnam","Hanoi","Ho"),("Spain","Madrid","Barcelona"),("Italy","Rome","Milan"),("Germany","Berlin","Munich"),("Japan","Tokyo","Osaka"),("Netherlands","Amsterdam","Rotterdam"),("Scotland","Edinburgh","Glasgow"),("Zealand","Wellington","Auckland"),("Egypt","Cairo","Alexandria"),("Colombia","Bogota","Medellin"),("Ecuador","Quito","Guayaquil"),("Kazakhstan","Astana","Almaty"),("Philippines","Manila","Cebu"),("Israel","Jerusalem","Tel"),("UAE","Abu","Dubai"),("Saudi","Riyadh","Jeddah"),("Portugal","Lisbon","Porto"),("Poland","Warsaw","Krakow"),("Greece","Athens","Thessaloniki"),("Ireland","Dublin","Cork"),("Bolivia","Sucre","Santa"),("Tanzania","Dodoma","Dar"),
("California","Sacramento","Los"),("Florida","Tallahassee","Miami"),("Illinois","Springfield","Chicago"),("Texas","Austin","Houston"),("Washington","Olympia","Seattle"),("Michigan","Lansing","Detroit"),("Wisconsin","Madison","Milwaukee"),("Maryland","Annapolis","Baltimore"),("Ohio","Columbus","Cleveland"),("Oregon","Salem","Portland"),("Nevada","Carson","Las"),("Pennsylvania","Harrisburg","Philadelphia"),("Kentucky","Frankfort","Louisville"),("Louisiana","Baton","Orleans"),("Minnesota","Paul","Minneapolis"),("Alaska","Juneau","Anchorage"),("Nebraska","Lincoln","Omaha"),("Vermont","Montpelier","Burlington"),("Tennessee","Nashville","Memphis"),("Alabama","Montgomery","Birmingham"),("Delaware","Dover","Wilmington"),("Connecticut","Hartford","Bridgeport"),("Montana","Helena","Billings"),("Kansas","Topeka","Wichita"),("Jersey","Trenton","Newark"),("Maine","Augusta","Portland"),("Missouri","Jefferson","Louis"),("Georgia","Atlanta","Savannah")]
ok=[]
for s,c,w in trip:
    r=[]
    for x in (s,c,w):
        t="▁"+x
        r.append(V.get(t))
    print(s,c,w, r, [tok.encode(" "+x) for x in (s,c,w) if V.get("▁"+x) is None])
    if all(v is not None for v in r): ok.append((s,c,w))
print(len(ok))
json.dump(ok, open("/work/trip.json","w"))
```