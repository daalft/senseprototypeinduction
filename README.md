# senseprototypeinduction
Word sense prototype induction

# Example usage
```
from prototypeinduction import PrototypeInduction
pi = PrototypeInduction()
# step threshold is the only parameter of the model and may need to be adapted
step_threshold = 0.4
# the expected format is a list of tuples with a context (sentence),
# indices of the target word in characters (as tuple), and unique identifier
ctx_idx_ids = ...
# result contains the input data with an additional element indicating the cluster
result = pi.full_induction(ctx_idx_ids, step_threshold) 
```

## Example input
Data taken from the 2025 Shared Task on Navigating Disagreements in NLP Annotations [CoMeDi](https://comedinlp.github.io/)
```
[('Die Projektgruppe" Bänder, Blech und auch Double", die Gruppe" Unsere Schule, unterwegs nach Europa" und der MLLV-Arbeitskreis" Offene Lernformen im Fachunterricht".',
   (27, 32),
   'Blech-2357-5-109'),
  ('Geduckt wie eine Raubkatze steht er da, schnittig gestylt, mit den ins Blech geformten Muskelsträngen.',
   (71, 76),
   'Blech-7053-14-78'),
  ('Den Teig auf eine mehlierte Arbeitsplatte geben, 5 mm dick ausrollen, auf das Blech geben und im Ofen 20–30 Min. backen lassen.',
   (78, 83),
   'Blech-__DATEI__Honig-Ingwerschnitten__-12-36'),
  ('Die Pralinenform ist ähnlich einer Muffinform, weist jedoch mehrere Einbuchtungen in unterschiedlichen Formen auf (Herzen, Kreise, Quadrate, Sterne, Halbmonde, etc.) ÿ. Pralinenformen gibt es als Blech mit zwischen 24 und 36 Mulden, in die die Pralinenmasse eingegossen wird.',
   (196, 201),
   'Blech-__DATEI__Pralinenform__-0-155')]
```
## Example output

```
 [('Die Projektgruppe" Bänder, Blech und auch Double", die Gruppe" Unsere Schule, unterwegs nach Europa" und der MLLV-Arbeitskreis" Offene Lernformen im Fachunterricht".',
   (27, 32),
   'Blech-2357-5-109',
   0),
  ('Geduckt wie eine Raubkatze steht er da, schnittig gestylt, mit den ins Blech geformten Muskelsträngen.',
   (71, 76),
   'Blech-7053-14-78',
   1),
  ('Den Teig auf eine mehlierte Arbeitsplatte geben, 5 mm dick ausrollen, auf das Blech geben und im Ofen 20–30 Min. backen lassen.',
   (78, 83),
   'Blech-__DATEI__Honig-Ingwerschnitten__-12-36',
   2),
  ('Die Pralinenform ist ähnlich einer Muffinform, weist jedoch mehrere Einbuchtungen in unterschiedlichen Formen auf (Herzen, Kreise, Quadrate, Sterne, Halbmonde, etc.) ÿ. Pralinenformen gibt es als Blech mit zwischen 24 und 36 Mulden, in die die Pralinenmasse eingegossen wird.',
   (196, 201),
   'Blech-__DATEI__Pralinenform__-0-155',
   1)]
```

# Choosing good values for step threshold
The value of `step_threshold` should be adapted. This is best done on a dataset containing sentences for both monosemous and polysemous words (with sentences for different senses of polysemous words). 
The parameter should be chosen such that monosemous words always result in one cluster, while polysemous words generate more than one cluster.

# Future plans
- Adapt for non-text use-cases
- Allow graph as input
- 
