# Neural Circuit Architectural Priors for Rat Locomotion

Nikhil X. Bhattasali\*, Jaron Cui\*, Lerrel Pinto, Grace W. Lindsay<br>
\*Equal contribution

Animals generate behavior through continuous interactions between their neural circuits, bodies, and
environment. Embodied simulations provide a means to study these interactions by testing neural
circuit models in biologically realistic settings. Recent efforts have developed embodied
simulations of diverse animals, and emerging work on biologically grounded neural architectures has
begun to integrate circuit connectivity and single-unit dynamics from experimental data into
artificial neural networks, introducing stronger inductive biases and closer mechanistic alignment
with biology. How can we develop embodied simulations that enable the study of increasingly complex
bodies and neural circuits? In this work, we address the challenge through coordinated advances in
the simulated body, neural architecture, and training infrastructure. We introduce Rat v2, a
biomechanical model that provides an efficient and flexible tool for simulating rat behavior. To
control this body, we develop Rat NCAP, a biologically grounded neural architecture for locomotion
based on mammalian spinal circuits that incorporates cell-type-specific connectivity and a novel
flexor-extensor neuromechanical interface. To optimize circuit parameters, we develop
GPU-accelerated simulation and training infrastructure. The integrated system learns to locomote at
different speeds and transition between gaits without imitation learning, producing locomotion
statistics, kinematics, and cell-type-specific neural activity consistent with experimental data.
Systematic analyses reveal that the neuromechanical interface substantially affects trainability and
gait quality, and in silico cell-type ablations reproduce known gait phenotypes and predict gait
changes under graded perturbations. Together, these findings support the view that embodied
simulations at an intermediate level of abstraction can simultaneously reproduce complex behavior
and yield mechanistic insights into the neural circuits that generate it.

Paper: [Neural Circuit Architectural Priors for Rat Locomotion](https://arxiv.org/abs/2611.23364)

Videos: [https://rat-ncap.github.io/](https://rat-ncap.github.io/)

```bibtex
@misc{Bhattasali2026RatNCAP,
  title = {Neural Circuit Architectural Priors for Rat Locomotion},
  author = {Nikhil X. Bhattasali and Jaron Cui and Lerrel Pinto and Grace W. Lindsay},
  archivePrefix={arXiv},
  primaryClass={q-bio.NC},
  eprint={2611.23364},
  year={2026},
}
```

## Structure

```text
rat-ncap/
├── checkpoints/              # Trained models: <config>/seed<N>/<epoch>.pt with configs and spaces.
├── demo.ipynb                # Demo notebook: rolling out and rendering a trained model.
├── libs/                     # First-party libraries.
│   ├── circuits/             # Neural circuit simulation in PyTorch.
│   ├── composer/             # MuJoCo/MJX environments, transforms, and rollouts.
│   ├── evolution2/           # Evolution strategies.
│   ├── infra/                # Hydra launcher, compute and output conventions.
│   └── utils/                # General utilities.
├── libs3/
│   └── tonic/                # Third-party library for reinforcement learning (modified fork).
└── projects/rodent/
    ├── envs/                 # Rat body, arenas, and the locomotion task.
    ├── models/               # Rat NCAP and MLP baselines.
    └── train/                # Entrypoint scripts and Hydra configs.
        └── configs/
            ├── 1_train/      # Training experiments.
            └── 2_test/       # Tests of the trained checkpoints.
```

## Installation

1. Create and activate the conda environment.

   ```bash
   conda env create -f conda.yaml
   conda activate rat-ncap
   ```

2. Store the environment variables in the environment and reload it. The optional argument sets
   `OUTPUT_ROOT`, where runs write their logs and checkpoints (default: `outputs/` in the
   repository).

   ```bash
   bash conda-setup.sh [OUTPUT_ROOT]
   conda deactivate
   conda activate rat-ncap
   ```

The setup script stores the required `REPO_ROOT`, `OUTPUT_ROOT`, `PYTHONPATH`, and `MUJOCO_GL`
variables in the conda environment. The provided environment file targets Linux with an NVIDIA GPU
(CUDA 12).

## Commands

Run the commands from `projects/rodent`.

Train Rat NCAP:

```bash
python train/train_evolution.py --config-name=1_train/1_4_flxext
```

Train one seed with a run tag:

```bash
python train/train_evolution.py --config-name=1_train/1_4_flxext seed=1 tag=my-run
```

Train an MLP baseline with PPO:

```bash
python train/train_tonic.py --config-name=1_train/4_3_mlp_ppo_flxext
```

Test a provided checkpoint:

```bash
python train/test.py --config-name=2_test/1_2_locomotion_statistics body/rat/controller=flxext
```

Inspect a composed Hydra config:

```bash
python train/train_evolution.py --config-name=1_train/1_4_flxext --cfg job
```

Run the Python tests:

```bash
python -m pytest ../../libs .
```

Swap `--config-name` for other configs under `train/configs/`. Runs use the local Submitit launcher
and write logs and checkpoints under `$OUTPUT_ROOT/<config>/<run>/<job>/`. A run's
`checkpoint/` directory has the same layout as the provided `checkpoints/<config>/seed<N>/`, and
both load with `mjx_objective.load_checkpoint`.

The demo notebook `demo.ipynb` provides rollout examples.

## Checkpoints

`checkpoints/<config>/seed<N>/` holds checkpoints for `train/configs/1_train/<config>`.

## License

This code is released under the [MIT License](LICENSE). The rat model is derived from
[dm_control](https://github.com/google-deepmind/dm_control) and is licensed under the Apache
License 2.0 (see [its README](projects/rodent/envs/bodies/rat/README.md)).
