I've confirmed the H4 result on the 256 fresh MATH problems: the original model got 7.8% with a 93.75% cut rate, and the C setup reached 55.1%, a paired gain of +47.3 points with 123 wrong-to-right flips against 2 right-to-wrong. Next I'm logging that in the notes, with 23 passes and 29 Python runs left. The main remaining MATH failure is still truncation at 32.4%, which is the largest category, so that's where I'll focus.

Checking whether MATH cut texts already contain the reference answer before loop onset, against a control of other problems' answers. The 32% cut rate is the biggest MATH failure mode, and the answer formats there are distinctive enough that this check should be more reliable than it was for GSM.

I'm checking whether the reference answer already shows up in the MATH cut texts before the loop starts, using other problems' answers as a control. If a sizable share does, the failure is in committing rather than in finding the answer, and that would be a bigger lever than the GSM case. I'm working out what kind of intervention would target that doubt-and-reverify pattern.

I'm going to check how often the correct MATH answer shows up in the cut texts before the loop starts, both as a bare string and boxed, and compare that against other problems' answers as a control. The distinctive formats here, like fractions and radicals, should make the match more reliable than the GSM check. Only if it turns up often enough does a commit-focused fix look worth pursuing.

