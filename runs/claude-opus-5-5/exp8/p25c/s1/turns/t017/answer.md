The loops start at a median of ~6,000 tokens (minimum ~1,900), and the text right before the onset shows the model already circling with near-identical "Wait, ..." lines before locking into an exact cycle. The DeltaNet decay parameters (A_log, dt_bias) look like ordinary trained values with no single glaring outlier, so I have no evidence yet of a single broken parameter. My working hypothesis is that a copying mechanism (attending back to the previous instance of the current line and reproducing its continuation) becomes overconfident once a line has repeated, making the loop deterministic under top-p 0.95. Before spending forward passes, I'll inspect the activation files saved by the diagnostic run to see their format and whether finished (correct) and looping answers differ along some clear direction at particular layers. A strong, layer-localized separation would give candidate directions to test causally; a diffuse one would push me toward teacher-forced logit attribution instead.

```python
# Inspect activation dump format from gen_2
adir = '/work/gen_2_acts/'
print(os.listdir(adir)[:10], len(os.listdir(adir)))
idx = json.load(open(adir+'index.json'))
print(type(idx), (list(idx.keys()) if isinstance(idx, dict) else idx[:2]))
```