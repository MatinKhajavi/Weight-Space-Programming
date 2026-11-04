I'm checking a null failure that happens when a equals b, which gives a zero vector. I'm also comparing word pairs like correct/incorrect and agree/disagree, where the scores come out mostly positive, while indeed/not scores negative, which seems backwards given that "not" is a negation.

I'm noticing the indeed/not score of −2.9 cuts against what I'd expect from a negation word. It could be that "not" shows up in phrases like "you're not wrong," so I'm wondering whether I need a null distribution to know if this is noise. I also suspect a morphology confound: incorrect and disagree both carry negative prefixes, so the axis might be picking up on that.

