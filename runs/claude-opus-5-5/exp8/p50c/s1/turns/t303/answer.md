Weight forensics finds nothing unusual. The copy heads' output blocks have ordinary norms (15.7 is even below its layer's mean; no head in any full-attention layer is beyond z = 2.3). The Wait brake's output column is also ordinary (norm 0.262, z = -0.7). The only outliers are low-norm, nearly unused neurons in every layer. So the looping comes from ordinary-looking components interacting badly, not from a single corrupted weight, and the scan points to nothing new. One refinement of C itself remains worth checking. Doubling 22.573's column also doubles its small off-target effects: per unit of activation it lowers "Ready" by about 0.23 logits and "</think>" by about 0.13, both tokens that close the thinking. If "Ready" belongs to the model's closing routine (for example "Ready to output" before </think>), the doubled brake could be slowing correct conclusions. I will count how often "Ready" appears and where (near </think> or not) in correct texts of the original model versus C, and print a few C closing sequences. Clear evidence would justify removing these components from the added half of the brake; no evidence closes this line.

```python
# Does the doubled brake's off-target suppression of 'Ready'/'</think>' plausibly hinder closing? 'Ready' usage near </think> in orig vs C texts
def closing_stats(paths, want):
    n=0; hasR=0; nearR=0; ex=[]
    for p in paths:
        for r in lg(p):
            for s in r['samples']:
                if status(s)!=want: continue
                tx=s['text']; n+=1; i=tx.find('</think>')
                if 'Ready' in tx: hasR+=1
                if i>0 and 'Ready' in tx[max(0,i-400):i]: nearR+=1
                if i>0 and len(ex)<3: ex.append(tx[max(0,i-160):i+12])
    return n, round(hasR/max(n,1),3), round(nearR/max(n,1),3), ex
for nm,paths,want in [('orig ok',['/work/gen_6_original.jsonl','/work/gen_16_original.jsonl','/work/gen_27_original.jsonl'],'ok'),
                      ('C ok',['/work/gen_13.jsonl','/work/gen_16_edited.jsonl','/work/gen_27_edited.jsonl'],'ok'),
                      ('C cut',['/work/gen_13.jsonl','/work/gen_16_edited.jsonl','/work/gen_27_edited.jsonl'],'cut')]:
    n,h,nr,ex=closing_stats(paths,want); print(nm,'n',n,'has Ready',h,'Ready within 400ch before </think>',nr)
    if nm!='C cut':
        for e in ex[:2]: print('   ', repr(e)[:230])
```