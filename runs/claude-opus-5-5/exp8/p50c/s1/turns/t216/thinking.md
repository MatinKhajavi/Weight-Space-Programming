I'm loading the reference answers from the practice files into a lookup by id, so I can check the analysis against them. Then for each candidate text I'll find where the correct answer first appears as a concluding statement, such as "answer is X" or a boxed result.

I'm mapping each candidate's text to its reference answer and searching for the first place the correct value shows up as a conclusion, whether as "answer is X", a boxed result, or a final-answer line. Since MATH answers contain LaTeX, I'm planning to strip whitespace from both sides before matching, then estimate token positions from the character fraction. I'll also note what follows each first correct statement, plus whether other candidate answers turn up after it.

