# Utils

This library provides general utilities used by the other libraries:

- `collections`: `AttrDict` (a dict with attribute access) and `GlobDict` (a dict whose keys are
  glob patterns, with numeric ranges such as `limb[1-4]`).
- `string`: Parsing and matching of glob patterns with ranges, and human-readable unique IDs.
- `io`: Reading and writing files by their suffix (YAML, CSV, pickle, text, binary, ...).
- `logging`: A `Logger` of messages, saved files and CSV tables in a log directory.
- `compute`: Information about the job's process, CPUs and GPUs.
- `torch`: PyTorch modules and helpers: block-structured linear layers with shared and constrained
  weights (`block`), activations, buffers, constraints, containers and module inspection.
