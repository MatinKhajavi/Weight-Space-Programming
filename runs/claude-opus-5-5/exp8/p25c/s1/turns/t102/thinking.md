The notes write failed on a stray "%" in the string, which the % formatting tried to interpret. The edit itself was already saved to cand_E5, since the save loop ran before the failing write. I'll rewrite the notes without % formatting in the next Python run, and fold in one more useful step while I'm at it.

I'm going to re-run the notes write without the % formatting, and I want to confirm the cand_E5 files are actually on disk before the generation pass.

