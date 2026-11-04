I'm comparing the two experiments: E1 and E2 share roughly three strong common directions, given partial overlap in their principal cosines. The L17 block copy and head 3 are positive in both runs, while head 1 is weak in E1 but strong in E2, so it looks less robust. Now I'm moving on to the L16 heads.

I'm checking the L16 heads now: L16H3 looks unreliable, since E1 has it at -1.0 and E2 at +3.6, while L16H2 is negative in both runs, so it may be the more consistent copy head for B. The MLP answer-share numbers hold up better, with L16 and L17 showing clear increases in A, while L15 drops out once I exclude the circular comparison. So I'm leaning toward dropping L16H3 and L15's MLP from the component list, and considering L16H2 as a replacement.

I'm dropping the L16H3 and L15 MLP entries, and I'm keeping L16H2 as a candidate for the B-side copy role. The list is then L17H3, L17H1, the L16 and L17 MLP units, L17H0, L16H1, L16H2, L11H2, L7 head 0, and L0H1, with p still at 0.7.

