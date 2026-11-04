The result is clear on two points. First, head ablation deepens indentation: within 200 lines the median maximum indentation is 8 spaces for the original, 16 for A and 12 for C. The share reaching 24 or more spaces is 11% for the original, 34% for A and 27% for C. Second, within C, early deep indentation is strongly associated with failure. On GSM8K, 56% of failed answers reach 24 or more spaces within the first 100 lines, versus 15% of correct ones; on MATH it is 27% versus 8%. That is correlational: doubt could produce nesting rather than the reverse. The mechanistic question is whether the model's indentation choice at each newline became less confident after the ablation. If so, the sampler's -1.5 penalty on already-used depths can flip it towards fresh deeper depths. I will measure, at newline positions in the recorded windows (original fwd_4, A fwd_11, C fwd_18), the probability of the chosen indentation token and its margin over the best alternative depth token not yet used.

```python
# At newline positions: confidence of indentation token and margin over best UNUSED whitespace alternative (would presence penalty 1.5 flip it?)
ws_ids = set(i for i in range(250000) if i < 248000 and (lambda s: s != '' and s.strip(' ') == '')(tok.decode([i])))
print('n pure-space tokens', len(ws_ids))
for d in ['/work/fwd_4', '/work/fwd_11', '/work/fwd_18']:
    m = json.load(open(d + '/meta.json')); lens_ = m['lengths']
    ids_ = np.load(d + '/input_ids.npy'); tk = np.load(d + '/topk_ids.npy'); tl = np.load(d + '/topk_logprobs.npy')
    p_act = []; margins = []; flips = 0; npos = 0
    for b in range(len(lens_)):
        n = lens_[b]; seen = set()
        for p in range(n - 1):
            seen.add(int(ids_[b, p]))
            if tok.decode([int(ids_[b, p])]) == '\n' and int(ids_[b, p + 1]) in ws_ids:
                npos += 1; top = tk[b, p]; lps = tl[b, p]
                act = int(ids_[b, p + 1])
                if act in top: p_act.append(np.exp(lps[list(top).index(act)]))
                # penalized logits: subtract 1.5 for seen tokens; does an unseen ws alternative win over the best seen candidate?
                adj = np.array([lp - (1.5 if int(t) in seen else 0) for t, lp in zip(top, lps)])
                win = int(top[np.argmax(adj)]); best_raw = int(top[0])
                if win != best_raw and win in ws_ids: flips += 1
                cand = [(lp, int(t)) for t, lp in zip(top, lps) if int(t) in ws_ids]
                if len(cand) >= 2: margins.append(cand[0][0] - cand[1][0])
    print(d[-6:], m.get('model'), 'newline->indent positions', npos, '| median p(actual indent)', np.round(np.median(p_act), 3), '| median margin top-ws vs 2nd-ws', np.round(np.median(margins), 2),
          '| share margin<1.5', np.round(np.mean(np.array(margins) < 1.5), 2), '| penalty flips to unseen ws', flips)
```