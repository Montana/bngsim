- **A `.net` tfun whose index is written as a call, `tfun([0,10],[0,10], Atot())`,
  loads.** BioNetGen keeps a tfun's index as written and puts it in the `.net`
  as it is. The loader read the index up to its first parenthesis, as `Atot(`,
  and the load failed with "index variable 'Atot(' not found". The bare name
  loaded. Both `.net` readers share the parse and both failed, for inline,
  file-based and embedded tables, and for an index over a parameter, an
  observable, a function or the clock (`time()`). The index is now read with
  its `()` and names what the bare name does. In a file-based tfun, an index
  whose name begins with `method` (`tfun('d.tfun', method_k)`) was taken for
  the `method=>` option, and the table was read over the clock; it is the
  index now.
