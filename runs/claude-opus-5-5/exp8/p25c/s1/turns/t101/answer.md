The competitors to So at these decision points are mostly other openers, many of them the space-prefixed variants of the targeted set: ' Wait' 13% of probability mass, ' Let' 11%, ' If' 7%, then ' The' 6% and ' $' 3%. L23's So push lies almost entirely inside the same centred five-opener subspace used for E4b. Its projection carries +1.2 of the margin at k=0 and +3.5 at k=5, out of a total L23 contribution of +1.2 and +4.1. So extending the E4b construction to L23 attention is the evidence-based next step. It would turn L23's repetition-driven copy of the previous opener into an anti-copy, after which the model's own alternatives (other openers) take over. Because L23 is the last attention layer, only the L23 MLP can compensate downstream. The first-order effect at β=3 would be very large (−10 logits at k=5), so I choose β=2 for L23. That shifts the So margin by about −2.4 at k=0 and −7 at k=5, similar in size at k=0 to what E4b did in L15. E5 = E4b + this L23 edit. I build it, check its size, save it, and log the evidence in the notes.

```python
# Build E5 = E4b (L15,L19 beta=3) + L23 o_proj centred 5-opener subspace reversal with beta=2; save and log
P_c5 = torch.tensor(B_c5 @ B_c5.T, dtype=torch.float32)       # 1024x1024 projector (readout space)
n23 = 'model.language_model.layers.23.self_attn.o_proj.weight'
W23o = load_original(n23)
beta23 = 2.0
W23n = W23o - beta23 * (P_c5 @ W23o)
print('L23 rel change', float((W23n-W23o).norm()/W23o.norm()))
# sanity: L15/L19 subspace equals B_c5 span?
for l in (15,19):
    n = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
    dW = load(n) - load_original(n)
    print(l, 'delta inside B_c5 span frac', float((P_c5 @ dW).norm()/dW.norm()))
save_edit(n23, W23n)
print(edited())
os.makedirs('/work/cand_E5', exist_ok=True)
for l in (15,19,23):
    n = f'model.language_model.layers.{l}.self_attn.o_proj.weight'
    torch.save(load(n), f'/work/cand_E5/L{l}_o_proj.pt')
with open('/work/notes.md', 'a') as fh:
    fh.write("\n- PASS 11 (gen_12) E4b on D n=2: acc 34.4% (gsm 45.8, math 27.5); paired vs E3 +7.42+-2.45 (gsm +13.5+-4.6, math +3.75+-2.75; 96 w->r, 58 r->w); vs orig +24.0+-2.46. cut 64%. 12 wrong-finished all were cut under E3 (newly finished hard ones). E4b saved /work/best_E4b.\n"
             "- E4b loops: Let/Okay/If vanished (incl. space variants); 'So'/' So' dominant (36% of loops; '* So ...' bullet restatements, 'So the answer is 12.' x N).\n"
             "- PASS 12 (fwd_13) TF under E4b on 9 So-loops: P(So) by k(prev identical lines) 0:.25 1:.12 2:.32 3:.43 4:.54 5:.62. DLA So-margin: L15 attn now -1.4->-3.5, L19 -0.6->-1.3 (reversal works); L23 ATTN +1.2->+4.1 (main grower, +2.5), L23 mlp -0.5->-2.5, L22 mlp +1.0..1.5, L21 attn +0.8 growth. L23's So push lies in centred 5-opener subspace (1.21->3.53).\n"
             "- E5 = E4b + L23 o_proj same centred subspace reversal beta=2 (rel change %.3f). Testing generation on D.\n" % float((W23n-W23o).norm()/W23o.norm()))
```