```python
# Hypothesis: record plan in notes before first forward pass.
pools['D_math'] = pools['D_math'][:320]
json.dump(pools, open('/work/pools.json','w'))
notes = """# Notes (Qwen3.5-0.8B math repair)
## Setup
- Pools in /work/pools.json: D (diagnosis: 320 MATH level-stratified like MATH-500 + 192 GSM8K), S (selection), H (held-out).
- Prompt format: user = problem + "\\nPlease reason step by step, and put your final answer within \\boxed{}."; thinking on.
- Token ids: <think>=248068 </think>=248069 <|im_end|>=248046 eos(<|endoftext|>)=248044 ' Wait'=13428 'Wait'=13784
## Pass log
"""
open('/work/notes.md','w').write(notes)
print(len(pools['D_math']), len(pools['D_gsm']))
```