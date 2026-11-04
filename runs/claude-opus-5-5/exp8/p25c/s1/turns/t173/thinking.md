E8 is in place, and L23's o_proj shifted by about 0.47 relative to the original, which is large but expected since two of eight heads were removed. Next I'll update the notes with the plan and OV analysis, then launch pass 23 on the D set at n=2 with max_tokens 32768, using the same problem list as gen_16. I need to pull the D problem IDs from /work/sets.json, so I'm checking its structure before building the forward JSON.

I need to pull the problem IDs out of the set file before I can build the forward block, and the tool only takes inline text, so they have to be listed in full.

