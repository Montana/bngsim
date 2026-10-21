- **SBML: an event priority that is an integer too large for a C int loads.**
  The loader gave every integer-valued constant priority the engine's int
  field, so `1e10`, `-3e9` or `1e308` failed the whole load with pybind11's
  "incompatible function arguments". Such a priority takes the double path a
  fractional one does, and orders as written.
