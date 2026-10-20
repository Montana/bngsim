- **A source install no longer fails to import after a second configure of
  the same build tree.** With no system SuiteSparse, the first configure
  builds KLU from source and caches where it put it. A second configure, such
  as a `pip install` after a failed one, found KLU through that cache and
  skipped bundling it beside the extension, which then failed to import
  (`Library not loaded: @rpath/libklu.2.dylib`). A KLU found under the
  autobuild prefix is now bundled on every configure, decided by a path-prefix
  test (a build path holding `+` or `(` is fine), and a cached libklu that no
  longer exists is searched for again, so a deleted autobuild is rebuilt.
