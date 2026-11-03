Looking at the output biases, the border is strongly negative (corners around -0.69, edges around -0.4 to -0.5) while the interior sits near zero, so the typical generated image seems to be a black frame with content in the middle. Next I'm turning to the rows of W3 to see how the weights are structured, starting with row 0.

Now I'm going through the rows of W3 that have a clear shape. Row 0 has a localized blob near pixel 64, and rows 1 through 3 each show a horizontal or vertical line in the lower half of the image.

Now I'm checking the vertical line at columns 5 and 6, which runs through rows 1 to 7 with negative weights. Row 6 also has a vertical line, this one at column 10 across rows 1 to 8. Row 7 looks broadly negative, strongest along the bottom, so it seems to darken the whole image, and row 8 is mixed.

Now I'm checking the vertical line at column 5, where the positive weights run through rows 2 to 9 and peak around pixel 77. Next is column 1 of row 16, a negative vertical line through rows 3 to 10.

Now I'm checking the weights in row 22, where column 3 carries a vertical run of negative values through rows 1 to 7, and row 8 has its own horizontal run, so the pattern might be an L shape.

Now I'm checking the vertical run in column 7, rows 6 to 10, where the weights are mostly negative, and then looking at column 9 of row 28, which carries positive weights with a negative tail at column 11.

Now I'm reading row 40 of W3, where columns 1 through 8 carry strongly negative weights and form a horizontal line in row 6. Row 41 continues the scan with a run of negative values in row 10.

Now I'm checking the remaining rows of W3 for column and row patterns, starting with a vertical run at column 1 in rows 1 through 7, and then a horizontal line in row 7 across columns 3 to 10.

Now I'm reading row 4 of W3 for its horizontal run of positive weights across columns 3 to 9, and row 3 for a wider band spanning columns 1 to 10. Then I'm checking column 7 of row 49, where the weights are mostly negative down rows 3 to 9, and turning to row 50.

Now I'm checking the remaining weights in the pattern-scan rows, where row 4 shows a run of negatives across the left columns and row 51 has a small positive block in the lower right.

Now I'm checking column 8 in row 57 of the same weight matrix, the next stretch of the scan.

Now I'm looking at the rows of W3 that show single-line patterns, and the pattern holds: most units seem to draw one horizontal or vertical segment, about 6 to 9 pixels long, with either a positive or negative sign. That suggests the network learned to produce line-like strokes.

I'm checking whether the units combine horizontal and vertical segments into L-shapes or boxes, since row 22 pairs a vertical run with a horizontal one. Some units also place adjacent columns with opposite signs, which might just be fine positioning. The possibility that the training images are rectangles, either outlined or filled, is still open.

Now I'm checking the weights of W1 by column, starting with unit 15, where the first rows show a small positive response that grows in the second row before fading.

Now I'm looking at unit 28's column in W1, where the values through the first rows are small and then grow to a peak around rows 2 to 3, which suggests another line-like detector.

Now I'm looking at unit 28 in W1, where the pixel values peak in columns 6 to 8 across rows 1 to 4, so it may be a thicker vertical shape rather than a single line. Next I'm checking unit 40, starting from its first few pixel values.

Now I'm checking unit 51 in W1, where the early pixel values climb toward a peak, which suggests a shape that grows over the first rows.

Now I'm reading unit 51's horizontal line at row 2 and unit 44's line at row 8, and I'm moving on to unit 43.

Now I'm checking unit 43, which seems to respond to the lower-right region around rows 7 to 10 and columns 7 to 8, and then moving on to unit 46.

Now I'm reading unit 3's column in W1, where the pixel values climb to a peak around rows 2 to 4, another vertical-line detector near column 3.

Now I'm checking unit 7's column in W1, where the values climb from about 0.22 toward a peak near 0.35 further down, another line-like detector.

Now I'm checking unit 17, where the weights in the early rows run negative and then fade, which might mean it responds to the absence of something rather than a stroke.

Now I'm checking unit 18's lower-left blob, then unit 22, where the pixel values put a horizontal response along row 10 across the left columns.

Now I'm checking unit 26, where column 26 has a negative value at pixel 66, and then looking at unit 20, which carries a positive response in the same region.

Now I'm reading unit 27's negative run across columns 2 to 7 in row 6, then checking unit 31, which looks like a vertical detector at column 9 running through rows 1 to 9. Unit 41 is next.

Now I'm checking unit 41's vertical response at the left edge and unit 42's negative run across the lower-left columns, then starting on unit 47's small values.

Now I'm checking unit 49, which looks like a vertical detector at the left edge, and then unit 54, where the values build toward a peak further down.

Now I'm checking unit 58's response to the pixel values in its early rows, where one entry turns positive partway through.

Now I'm going through a few more weight columns of W1, where unit 62 looks like a vertical detector around columns 2 and 3, and unit 63 shows a strong response in the middle rows that might be another small shape.

Now I'm checking unit 63's values in rows 8 and 9, where some entries are strong and the layout seems to wrap around, so I need to confirm which column the last reading belongs to.

Now I'm checking the weights in column 63 across rows 8 through 9, where the values are strong near the top and then fall off, and I'm reading the remaining rows down to row 121.

Now I'm checking the rest of unit 63's weights, where the strong values sit in rows 8 and 9 on both the right and left sides, and I'm trying to work out what that split implies about its shape.

Now I'm checking the time-input rows of the weight matrix, where the values near rows 149 through 158 look larger than the rest, and I'm trying to see which inputs carry the most signal.

Now I'm checking the time-input rows near the end of the weight matrix, where the entries around rows 148 through 158 look much larger than the pixel rows, and I'm reading which of them carry the most signal.

Now I'm scanning the next block of weight values in W1, where several rows show large entries near the time-input rows, including one entry that reaches about 1.4 in magnitude.

Now I'm reading the time-input rows at the end of W1: the sine rows 144 to 147 are small and the rows 148 to 151 are large, while the cosine rows 152 to 155 are small and 156 to 159 are moderate to large. The low-frequency inputs in the last few rows seem to encode time as a slowly changing signal.

Now I'm working out which kinds of images these units might be detecting, weighing single straight lines against rectangles, L-shapes, and crosses. I'm checking unit 22, whose weights in W1 respond to a bottom-left horizontal segment, and whether its W3 output pairs that with a vertical stroke, which would suggest a combined L-shape.

Now I'm looking at unit 25's weights in W3, where column 7 in rows 6 to 10 pulls strongly negative while a run of positive values sits further down.

Now I'm checking unit 25's column-7 weights, which run negative across rows 2 to 9 and suggest it fires when that column is dark, then I'm working out what its output weights do with that signal.

Now I'm checking unit 51's weights in W3, where a block of positive values sits across rows 6 to 9 and columns 7 to 10, pointing to a filled region rather than a single line.

Now I'm checking unit 51's weights in column 51 across rows 2 through 3, where the values are large and positive, which suggests a horizontal detector near the top.

Now I'm working out what the W3 rows encode, which is a separate question from the W1 columns because those rows belong to the second hidden layer. I'll look at the output side of unit 51 on its own terms rather than matching its W1 pattern against it.

Now I'm looking at the h2 output patterns, which are mostly single one-pixel lines running horizontally or vertically across six to nine pixels, with some blobs and negative side-lobes mixed in.

Now I'm weighing the bias terms in the second layer: some of the positive offsets are large, so units with big biases will fire even on a mostly empty image.

Now I'm checking the background bias terms, where the always-on negative weights in W3 seem to supply the black backdrop, and the border values are strongly negative so lines rarely reach the edge.

I'm weighing whether the border values point to single lines or rectangle outlines, since some detectors fire at column 1, column 11, row 1, and row 10, which means lines can reach the edge too.

Now I'm checking whether the pattern in W3 is a pair of parallel lines, since a rectangle outline would show up as two horizontal and two vertical edges. Row 5 has negative values in two adjacent columns and positive values in a few columns on the right, which I need to read more carefully before deciding anything.

Now I'm checking the rest of the W3 row, where the dark values sit at columns 5 and 6 while the bright ones show up at columns 0–1 and 8–9, which might be a shifted vertical pattern rather than a single line.

Now I'm reading row 24 of W3, where the positive values sit in rows 7 and 8 across the left columns and the numbers shift around there, so I'm checking which entries actually belong to that group.

Now I'm checking whether the horizontal line in row 7 is only the left portion, and whether a diagonal might explain the blob near the top right of unit 40.

Now I'm reading the block-shaped detectors across units 28, 18, 43, and 63, which seem to mark regions of the image rather than single lines, so they'd help localize things under heavy noise. I'm checking whether the images are better described as filled areas than as drawn strokes.

I'm checking whether row 22 in W3 is the signature of an L-shape, since column 3 across rows 1 to 7 and row 8 across columns 3 to 9 are both negative and meet at a corner. I'm weighing that against the single-line reading, which would fit a unit that just suppresses both strokes together.

Now I'm checking the vertical run in column 3 of W3 row 22, where the values stay strongly negative down the column, and I want to confirm the exact entries near the end of the row before reading anything into the corner.

Now I'm reading the first entries of row 22 in W3 to pin down which pixel index the negative value at column 3 sits at.

Now I'm reading the column-3 values in row 22 of W3 more closely, where the negative entries at pixels 15, 27, 39, 51, and 63 keep repeating down the same column.

Now I'm reading the raw values in row 8 of W3, where the negative entries cluster at columns 5 through 10, which is the bottom edge of the corner I was tracing.

Now I'm weighing whether this unit marks a bright block in rows 2 to 6 across columns 6 to 10 while staying dark at column 3 and row 8, which could be a region detector or an ambiguous single line.

Now I'm looking at the W3 row 55 entries, where the strong values run across roughly nine pixels in row 1, which points to a long horizontal stroke rather than a block.

Now I'm listing the remaining W3 rows where the strong positive values form single straight lines, a horizontal one in row 10, one in row 6, and so on, each with its own start and end columns.

Now I'm noting that the lines run about seven to nine pixels, with varying start points, and the border weights are strongly negative, so they rarely reach the edge.

Now I'm checking whether the border weights are really consistent with segments that sit mostly in the middle, and I'm also weighing a fixed training collection as an alternative to random placement.

Now I'm checking whether the lines can also cross, with a horizontal segment and a vertical one meeting, by looking at the W3 entries around row 25 where a vertical negative run sits beside a positive block.

Checking that no crossing pattern appears: none of the strong units paint a horizontal and vertical line together, so each one looks like a single segment. I'll sketch a concrete example, a horizontal line across row 5, columns 2 to 9, to make the picture concrete. My confidence in this reading is around 60%.

