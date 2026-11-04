The association holds in the original model too: "my thought process" beats "a thinking process" on GSM (21% vs 10%) and MATH (22% vs 3%), with fewer cut-offs. The third mode, openings like "The user wants…" or "The problem asks…", is almost always fatal in the original (4%), and my L23 head ablation nearly eliminated it. So the opening is an intrinsic mode switch. My edits moved some mass from "my thought" to "a thinking", which may have cost accuracy.

Causality is still unproven, because problem difficulty influences the choice. The efficient design has two steps. Step 1 is one cheap teacher-forced pass over about 1,000 fresh practice prompts, each followed by the start of the thinking. It measures, for every problem, the probability of " my" vs " a" right after "Here's" under the current edit C. Step 2 picks problems where both openings are likely and samples each several times, giving a within-problem comparison of outcome by opening. Step 1 also gives a dense, per-prompt measure of the opening choice, to test which components drive it. First I need the exact prompt format and the unused problem pools.

```python
# Prepare: exact eval prompt format + unused problem pools for an opening-choice teacher-forced pass
print(A(open('/work/eval_prompt_format.txt').read()[:600]))
pools = json.load(open('/work/pools.json')); print({k:len(v) for k,v in pools.items()})
used = set()
for f in ['S1_ids.json','H1_ids.json','SH512_ids.json','diag_ids.json']:
    x = json.load(open('/work/'+f)); used |= set(x if isinstance(x,list) else sum(x.values(),[]))
print('used', len(used))
# token ids of the opening fork
for s in ["Here", "Here's", "Here's a", "Here's my", "Here is a", "Here is my", " a", " my", "'s", " is"]:
    print(repr(s), tok.encode(s).ids)
```