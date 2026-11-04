No variant helps either benchmark. The only consistent pattern is that ablating 23.4 cuts loops by about 5 points on both GSM8K and MATH, but those answers end up wrong (accuracy -0.8 and +0.8). That again says the remaining loops are mostly on problems the model cannot solve. C stays the final edit. Before using the remaining passes mainly for confirmation, I want to rule out one last specific failure mode common in small Qwen models under a presence penalty. The penalty (1.5 here) favours tokens not yet used, so long reasoning can drift into rare tokens or another language (for example Chinese characters). That would corrupt reasoning in exactly the long, failing texts. I will count CJK characters and unusual non-ASCII symbols (excluding ordinary math symbols) in C's answers, split into right, wrong and cut. If drift is concentrated in failures, it is a distinct mechanism to investigate; if it is rare, it is excluded.

```python
# Check for language/rare-token drift (CJK, unusual unicode) in C texts by outcome — a presence-penalty failure mode in small Qwen models
mathsym=set('°×÷√π≤≥≠→←↔∞∑∏∫∂∆∇±∓·⋅…′″≈≡∈∉⊂⊆∪∩∅∀∃¬∧∨⇒⇔ℝℕℤℚℂαβγδεθλμσφωΔΣΩ–—‘’“”')
def drift(tx):
    cjk=sum(1 for ch in tx if '\u4e00'<=ch<='\u9fff' or '\u3040'<=ch<='\u30ff' or '\uac00'<=ch<='\ud7af')
    odd=sum(1 for ch in tx if ord(ch)>127 and ch not in mathsym and not ('\u4e00'<=ch<='\u9fff'))
    return cjk, odd
agg=collections.defaultdict(lambda: [0,0,0,0])
for p in ['/work/gen_13.jsonl','/work/gen_16_edited.jsonl','/work/gen_27_edited.jsonl','/work/gen_32_edited.jsonl','/work/gen_33_edited.jsonl']:
    for r in lg(p):
        for s in r['samples']:
            st_=status(s); c_,o_=drift(s['text']); a=agg[st_]
            a[0]+=1; a[1]+=(c_>0); a[2]+=(o_>20); a[3]+=o_/max(1,len(s['text']))*1e4
for k,(n,c1,o1,o2) in agg.items(): print(f'{k}: n {n}, texts with CJK {c1/n:.3f}, texts with >20 odd non-ASCII {o1/n:.3f}, odd per 10k chars {o2/n:.2f}')
```