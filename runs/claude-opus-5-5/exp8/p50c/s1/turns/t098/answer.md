At the positions that predict each cycle's line-initial " Wait", many heads put most of their attention on earlier "Wait" tokens. These include the three heads already zeroed by Edit A (15.7: 0.88, 23.0: 0.79, 23.5: 0.72) and several others: 11.1 (0.82/0.77), 19.0 (0.81/0.64), 23.7 (0.75/0.69), 19.3 (0.65/0.41), 23.1 (0.61) and 23.4 (0.61/0.39). The attention is spread over all earlier "Wait" tokens, not concentrated on the previous cycle (only 0.04–0.17). So this is not exact-sequence induction. It looks like line-start copying: at the start of a new line, look at how earlier lines started and continue the pattern. In a doubt spiral every line starts with "Wait", so the mechanism would feed on itself. Attention alone doesn't show which heads write "Wait". As a cheap first screen, I'll check each head's value/output circuit. I take the normalized " Wait" embedding through the layer's input-norm weights, the head's value and output projections and the final-norm weights. Then I read off the logit of " Wait" against common line-initial alternatives ("So", "Okay", "Let", "But", "Actually", "Hmm", "This", "The", "I", "$"). This is a pure weight analysis and ignores the output gate. If a head maps "Wait" strongly onto the "Wait" logit, it is a candidate driver of the self-reinforcing doubt lines to test causally. If no head does, the attention to "Wait" tokens may only be a sink pattern.

```python
# Hypothesis (weight-only OV screen): some heads that attend to earlier 'Wait' tokens map 'Wait' -> +Wait logit (line-start copying)
alts_s = [' So',' Okay',' Let',' But',' Actually',' Hmm',' This',' The',' I',' $']
alt_ids = [tid(s) for s in alts_s]; print(alt_ids)
ew = Eemb[WAIT].astype(np.float64); ew = ew/np.sqrt((ew**2).mean())
U_sel = Eemb[[WAIT]+alt_ids].astype(np.float64) * (1.0+wn.numpy())[None,:]
res_ov = []
for L_ in (11, 15, 19, 23):
    lnw = 1.0 + load_original(pre+f'{L_}.input_layernorm.weight').numpy()
    Wv = load_original(pre+f'{L_}.self_attn.v_proj.weight').numpy(); Wo = load_original(pre+f'{L_}.self_attn.o_proj.weight').numpy()
    xin = lnw*ew
    for h_ in range(8):
        kv = h_//4
        v = Wv[kv*256:(kv+1)*256] @ xin
        out = Wo[:, h_*256:(h_+1)*256] @ v
        lg = U_sel @ out
        res_ov.append((L_, h_, lg[0]-lg[1:].mean(), lg[0], np.linalg.norm(out)))
res_ov.sort(key=lambda r: -r[2])
for r in res_ov[:10]: print('L%d.h%d  Wait-minus-alts %+.2f  (Wait %+.2f, |out| %.1f)' % r)
print('lowest:', ['L%d.h%d %+.2f' % (r[0], r[1], r[2]) for r in res_ov[-4:]])
```