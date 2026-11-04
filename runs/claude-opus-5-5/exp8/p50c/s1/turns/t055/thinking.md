I'll pull the problem text from the practice files, since the rows don't include it. Then I'll load both datasets into a dict keyed by id, build each full text by joining the prefix, problem, suffix, and answer, and work out the token boundaries. Finally I'll check that the total stays under 524,288 tokens and each item under 32,768.

