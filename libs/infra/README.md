# Infra

`infra.launcher` extends [Hydra](https://hydra.cc/) with a shared base config, a local compute
backend, resolvers, logging, and a standard output directory layout.

Import the launcher from a Hydra entrypoint to register its config search path, callbacks and
resolvers:

```python
import infra.launcher
```

Extend `/infra/base` from the experiment's base config:

```yaml
# @package _global_
defaults:
  - /infra/base
  - _self_
```

The base config always uses Hydra multirun mode, including single-job runs. It launches jobs with
the Submitit local launcher (`/compute: local`), which runs a sweep's jobs one after another, and
writes each run beneath `$OUTPUT_ROOT`:

```text
$OUTPUT_ROOT/
  <experiment>/
    <timestamp>-[<tag>:]<run-name>/
      multirun.yaml
      .submitit/
      <job-id>:<overrides>/
        .hydra/
        stdout.log
        stderr.log
        <project outputs>
```

A **run** is one Hydra submission containing one or more **jobs**. The experiment name defaults to
the config name, and `tag` adds a label to the run directory name. The resources of each job are
`compute.cpu`, `compute.gpu`, `compute.mem` (GB) and `compute.time` (minutes).

Other Submitit or Hydra launchers (e.g. a Slurm cluster) can be added as further `compute` configs
next to `configs/compute/local.yaml`.
