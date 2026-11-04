The move came back as None, which I'm reading as an illegal move, probably a suicide. My Go code only checks for empty points, so the network may have picked a point that would leave its own group without liberties. Real games mask suicides out, so I'll handle None by printing "suicide".

