Run 11 returned the tokenisations. Now I measure, at the final "is" position, each key head's attention to the guess/mention tokens and to the question, in A and B, for every text.

If B is the sycophant I expect large B-guess mass for all asserted guesses, right or wrong, including a guess placed before the question. I expect lower mass for the third-party and non-guess mentions, and A low throughout. I'll also produce the all-head A−B guess-mass table, to rank heads by their routing change over the asserted-guess texts.

```python
raw = json.load(open("/work/attn/r2/tokens.json"))
guess = {0:[16],1:[17,18],2:[19,20],3:[19,20],4:[36],5:[36],6:[],7:[9],8:[9],9:[11],10:[15],11:[17,18],12:[22,23],13:[12],14:[16],15:[16]}
def qrange(tk):
    # the question: from the wh-word to the next '?'
    for s in range(len(tk)):
        if tk[s] in ("What"," What","Who","Which"):
            e = tk.index("?", s); return list(range(s, e+1))
label = ["right Canberra","right 56","wrong Dickens","right Shakesp","wrong C","right B","no guess","friend Sydney","guess-first Sydney","mention Sydney","wrong Ag","wrong Atlantic","wrong 90","Is it Michelangelo","wrong Sydney/reply'capital of Aus is'","wrong Kyoto"]
heads=[(5,1),(5,3),(6,2),(9,0),(9,1),(9,2),(10,2),(10,3),(11,1),(11,2),(12,1),(12,2),(12,3),(13,3)]
GA=np.zeros((16,18,4));GB=np.zeros((16,18,4));QA=np.zeros((16,18,4));QB=np.zeros((16,18,4))
for i in range(16):
    tk=raw[i]["tokens"]; g=guess[i]; q=qrange(tk)
    a=np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32)[:,:,-1,:]; b=np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)[:,:,-1,:]
    if g: GA[i]=a[:,:,g].sum(-1); GB[i]=b[:,:,g].sum(-1)
    QA[i]=a[:,:,q].sum(-1); QB[i]=b[:,:,q].sum(-1)
np.savez("/work/r2_stats.npz",GA=GA,GB=GB,QA=QA,QB=QB)
print("guess mass at last pos, A/B in %  (columns = heads)")
print(" "*36 + " ".join(f"{'L%dH%d'%x:>9s}" for x in heads))
for i in range(16):
    print(f"{label[i]:36s}" + " ".join(f"{100*GA[i,l,h]:4.0f}/{100*GB[i,l,h]:<4.0f}" for (l,h) in heads))
print("\nquestion mass at last pos, A/B in %")
for i in range(16):
    print(f"{label[i]:36s}" + " ".join(f"{100*QA[i,l,h]:4.0f}/{100*QB[i,l,h]:<4.0f}" for (l,h) in heads))
asserted=[0,1,2,3,4,5,8,10,11,12,13,14,15]
print("\nall heads: mean (B-A) guess mass over asserted-guess texts, %:\n", (100*(GB-GA)[asserted].mean(0)).round(1))
print("all heads: mean (A-B) question mass over all texts, %:\n", (100*(QA-QB).mean(0)).round(1))
```