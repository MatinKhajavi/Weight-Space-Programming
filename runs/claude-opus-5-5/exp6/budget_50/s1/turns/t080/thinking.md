I'm checking how many of the 372 pattern positions in those 128 games involve A capturing, which comes to 58 cases. That's roughly 0.45 captures lost per game. I want to know if those captures are worth much, since A capturing would take the adjacent opponent chain in atari, which could be large. That looks like a real cost, and I'm considering whether to penalize only some of these cases.

I'm weighing whether to penalize the 58 A-capture cases less heavily, since a real capture could take a large chain. I'm also wondering if the pattern could be refined to tell real captures from false ones, but I can't tell them apart reliably. Looking at the cycle case, the repetition loss is triggered by our recapture at B, so I'm considering whether to suppress that recapture instead.

I'm weighing the recapture option instead: after the opponent's capture, skipping the recapture at B would avoid the repetition loss, but it gives up a legitimate recapture in cases where the position didn't arise from our own A move. I still want to measure how large the chains captured at those 58 A-capture positions actually are, since small captures would make penalizing A cheaper than it looks.

