I'm resuming after a restart. The notes file has the setup but an empty pass log, so I don't have my earlier diagnostic numbers written down. The last output compared, layer by layer, some projection of the residual stream before a repetition loop and inside one ("pre / loop / diff"). The largest jumps are after full-attention layer 15 (+2.5) and at the top of the network (+4.3). Before deciding anything I need to rebuild the facts: what gen_3 showed, what fwd_4 contained, and which kernel variables hold the analysis behind that table. This block only inspects state and runs no forward pass.

```python
# Hypothesis: none yet - re-orient: list kernel variables, files, and summarize gen_3 (diagnostic generation)
import os, json, numpy as np
print(open('/work/notes.md').read()[-3000:])
print(os.listdir('/work'))
print(sorted([k for k,v in globals().items() if not k.startswith('_') and not callable(v) and not isinstance(v, type(os))])[:200])
rows=[json.loads(l) for l in open('/work/gen_3.jsonl')]
print(len(rows), rows[0].keys())
```