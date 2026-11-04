The brake-target control matches A: 35.6% (GSM 43.8, MATH 27.3, cut 64.1%) against A's 35.2% (GSM 43.0, MATH 27.3, cut 61.3%) on the same seeds. Doubling the other built-in layer-22 brake (the Maybe brake, 22.387) adds nothing on top of the head ablation. Only the Wait brake does, taking A to C's 60.2%. Both halves of C are now shown to be component-specific:
- Zeroing three non-copying heads in the same layers hurts (27.3%).
- Doubling a brake neuron of the same kind but with another target is inert (35.6%).

Each edit works through its identified mechanism (line-start copying; Wait suppression), not as a generic perturbation. I will record this and restore C. In the same run I will check two cheap leftovers. First, are any of C's GSM "wrong" answers really formatting near-misses (commas, dollar signs, units) rather than wrong numbers? Second, what induction attention do the untested layer-19 heads (such as 19.0) have in C's loop windows, which decides whether an ablation test of them is worth a pass?

```python
# Record brake-target control; restore C; check GSM 'wrong' for format near-misses; induction of L19/L11 heads under C
B31=smap(['/work/gen_31.jsonl']); A6=smap(['/work/gen_6_edited.jsonl'])
r3=pdiff(B31,A6); r4=pdiff(B31,C13)
with open('/work/notes.md','a') as f:
    f.write(f"- PASS 25 = gen_31: BRAKE-TARGET CONTROL (A + 22.387 Maybe-brake x2 instead of 22.573) on S1: acc 35.6 (gsm 43.8, math 27.3) cut 64.1 = A level (35.2). Paired vs A {r3[1]} SE {r3[2]}; vs C {r4[1]} SE {r4[2]}. => Wait-brake effect is target-specific. C restored.\n")
W22=W22o.clone(); W22[:,573]=2*W22o[:,573]; save_edit('model.language_model.layers.22.mlp.down_proj.weight',W22)
print('vsA',r3,'vsC',r4, 'C restored:', all(torch.equal(list(load_file('/work/best_C_x2/'+fn).values())[0].float(), load('model.language_model.'+fn.replace('.safetensors',''))) for fn in os.listdir('/work/best_C_x2')))
def normnum(s):
    s=str(s); s=re.sub(r'\\text\{[^}]*\}','',s); s=s.replace('\\$','').replace('$','').replace(',','').replace('\\%','').replace('%','').replace(' ','').replace('\\!','')
    try: return float(s)
    except: return None
near=0; tot=0; ex_=[]
for p in ['/work/gen_13.jsonl','/work/gen_16_edited.jsonl','/work/gen_27_edited.jsonl']:
    for r in lg(p):
        if r['source']!='gsm8k': continue
        for s in r['samples']:
            if status(s)=='wrong':
                tot+=1; a=normnum(s['pred']); b=normnum(r['answer'])
                if a is not None and b is not None and abs(a-b)<1e-6: near+=1; ex_.append((r['answer'],s['pred'][:30]))
print('GSM wrong:',tot,'format near-misses:',near, ex_[:5])
print({f'{l}.{h}':round(ind[(l,h)],2) for (l,h) in ind if l in (3,7,11,19)})
```