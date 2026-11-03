I notice only about a third of layer 0's MLP neurons are actually active, with the rest permanently dead from never firing — likely a common artifact in char-level models rather than something corpus-specific. I'm not fully certain what drives this, so I want to try running ICA on the alive neurons to see if that reveals anything clearer.

