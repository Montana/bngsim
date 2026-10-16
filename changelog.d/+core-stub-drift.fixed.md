- **The type stub for two `NetworkModel` bindings matches the bindings again.**
  `evaluate_over_times(shift=...)` is typed to accept an int as well as a float,
  as every other float argument in the stub is, and `table_function_tables` is
  typed `list` (of dicts), which is what the binding declares. The two lines had
  been written by hand, so `scripts/rebuild_editable.py` rewrote them and left
  the stub modified after every rebuild.
