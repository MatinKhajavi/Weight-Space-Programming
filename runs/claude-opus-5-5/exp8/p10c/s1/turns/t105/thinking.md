I'm comparing the doubt-word rates per 1k tokens between v2 and v3, with v3 showing higher Wait and But counts. I'm wondering whether that gap comes from v3's longer texts, since the rates include loop regions that inflate the counts.

I'm checking the v3 loops next: 79% of their cycles contain doubt or re-check phrases, versus 65% for v2, so the remaining loops look mostly doubt-driven. Copy suppression seems to have removed the doubt-free loops first, which fits with doubt cycles sustaining themselves through the content.

I'm weighing whether the L23 N375 neuron is a second doubt brake, since its activation looks positive when doubt is present and pushes against it, much like N573. I'm also considering L22 N387 for the same pattern. Before amplifying anything, I want to check how these neurons actually behave at doubt decisions, comparing their contribution to the Wait logit against other line starts, and only boost N573 if the others turn out to matter little.

I'm estimating how strongly N573 fires in v3 by averaging its activation over the first 8192 tokens per sample, and comparing that to the original run. I've settled on tripling its brake strength, since it's the only candidate worth amplifying, though it also dampens Wait at ordinary line starts, which is acceptable.

I'm planning to run the first-word census on the v3 loops to see whether "Wait" really dominates, since that decides whether amplifying N573 matters. If "But", "If", or "So" lead most of them, the brake would only be touching a minority of the doubt cycles.

I'm going to run the first-word census on the v3 loops and also check the other brake candidates' direct effect on the doubt decisions, then settle on v4.

