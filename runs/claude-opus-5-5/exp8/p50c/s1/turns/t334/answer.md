H7 completes the confirmation series:

| | Original | C |
|---|---|---|
| Accuracy | 8.6% | 59.8% (GSM 66.4%, MATH 53.1%) |
| Cut share | 91.8% | 23.1% |

The paired difference is +51.2 ± 3.2 points, with 132 wrong→right and 1 right→wrong. All seven held-out sets (1,792 paired items, none used to choose the edit) show gains between +45.7 and +51.6 points. Regressions are at most 3 per 256 items.

The remaining question for the report is how targeted C is: does it leave the model's predictions on normal, successful reasoning largely intact? The plan is a teacher-forced comparison: original vs C log-likelihood of the original model's own correct reasoning, token by token. The expected result is small average changes concentrated on specific tokens such as line-initial "Wait" and copy continuations. A large broad drop would mean the edit damages general modelling.

Earlier teacher-forced passes used request files (/work/req_tf_conc.json), but I do not remember the forward-block syntax for them. This run recovers it from the notes and the old request file, prints the exact prompt format, and builds the new request from about 90 short correct original-model texts. It also logs H6/H7.

```python
# R8 hypothesis: recover the TF request-file syntax and exact prompt format from earlier work, then build a TF specificity corpus (original model's own correct reasoning) to compare original vs C token log-likelihoods.
import json, inspect
nt=open('/work/notes.md').read()
for l in nt.split('\n'):
    if any(s in l for s in ('req_','request','texts_file','file"','"path','from file')):
        print('>>', l.encode('ascii','replace').decode()[:240])
q=json.load(open('/work/req_tf_conc.json'))
print('req_tf_conc type', type(q).__name__, (list(q.keys()) if isinstance(q,dict) else len(q)))
t0=(q.get('texts') or [''])[0] if isinstance(q,dict) else ''
print('conc text0 head:', t0[:230].encode('ascii','replace').decode().replace('\n','|'))
try: print('prompt_of src:', inspect.getsource(prompt_of)[:500].encode('ascii','replace').decode())
except Exception as e: print('no src', e)
sel=[]
for f in ['gen_27_original','gen_32_original','gen_33_original','gen_41_original','gen_43_original','gen_44_original']:
    for r in R(f'/work/{f}.jsonl'):
        for s in r['samples']:
            if s.get('correct') and s['tokens']<=5000: sel.append((r['id'], r.get('source',''), s['text'], s['tokens']))
print('original-correct short texts', len(sel), 'tokens', sum(x[3] for x in sel))
with open('/work/notes.md','a') as fh:
    fh.write("- PASS 36 = gen_43 (both, H6): orig 13.7 (cut 87.1) | C 65.2 (gsm 69.5, math 60.9, cut 22.3); diff +51.6 SE 3.1, W->R 132, R->W 0.\n- PASS 37 = gen_44 (both, H7): orig 8.6 (cut 91.8) | C 59.8 (gsm 66.4, math 53.1, cut 23.1); diff +51.2 SE 3.2, W->R 132, R->W 1. Held-out H1-H7 (1792 paired): +45.7..+51.6 each.\n")
```