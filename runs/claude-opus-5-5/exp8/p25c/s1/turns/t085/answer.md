E4b is as intended: a rank-4 change whose column space is exactly the centred span of the five opener readout directions (Wait, So, Let, Okay, If), with L15/L19 attention output along each of them flipped to −2× its original value. It therefore reverses only the relative opener preferences contributed directly by these two attention layers and leaves the common opener-vs-other-token component alone. The next step is a generation pass on the same D set (n=2, 32,768 tokens). It shares seeds with gen_2 (original) and gen_11 (E3), so I get a tight paired E4b-vs-E3 comparison. To write the forward block I need the exact 256 D problem ids, so I print them compactly here and update the notes. Confirmation would be accuracy clearly above E3 with an unchanged ~98% correct share among finished answers. A drop in that share would mean the opener reversal damages normal reasoning; an unchanged cut rate with new openers would mean the loops simply switch openers.

```python
# Print D ids compactly for the forward block; log E4b construction in notes
ids_list = [x if isinstance(x, str) else x.get('id') for x in (Dids if isinstance(Dids[0], str) else D)]
print(len(ids_list), len(set(ids_list)))
s_ids = json.dumps(ids_list, separators=(',', ':'))
print(len(s_ids))
print(s_ids)
with open('/work/notes.md', 'a') as fh:
    fh.write("\n- E4b verified: rank-4, column space = centred span of gF*WU[{Wait,So,Let,Okay,If}], L15/L19 attn output along it x(-2). rel change .183/.168. Saved as current edit (E3 kept in best_E3).\n"
             "- E3 remaining loops: median 3 newlines/period (5% none); openers '*' bullets (1123), So 177, Let 82, Okay 59, But 31, If 30. Wrap-up loops e.g. 'Okay, I'll output the answer. Final Answer: 10.' repeated inside think.\n")
```