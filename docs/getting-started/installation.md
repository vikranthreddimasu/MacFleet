# Installation

MacFleet ships as a Python package. Minimum: Python 3.11, macOS 14+,
Apple Silicon (M1, M2, M3, M4).

## Core install

```bash
pip install macfleet
```

Gets you everything you need for the distributed primitives: discovery,
auth, pairing, dashboard. No ML framework pulled yet.

## With PyTorch

```bash
pip install "macfleet[torch]"
```

Adds `torch>=2.1.0` and pulls the MPS backend bits. Use this if you
plan to call `pool.train(engine="torch", ...)`.

## With MLX

```bash
pip install "macfleet[mlx]"
```

Adds `mlx>=0.5.0`. Use for `engine="mlx"` training loops.

## Everything

```bash
pip install "macfleet[all]"
```

Installs torch + mlx + pyyaml. This is what most people want on a dev
machine.

## Building the docs

```bash
pip install "macfleet[docs]"
mkdocs serve
```

Opens `http://127.0.0.1:8000` with live reload.

## Verifying the install

```bash
macfleet --version
macfleet doctor    # checks thermal state, MPS, MLX, mDNS reachability
```

If `doctor` complains, read its output carefully. It lists what's
missing and what to fix.

## Platform notes

**macOS**: Apple Silicon is the target platform. Hardware, OS and engine-version
coverage must be verified on actual Macs; a passing CI job is not proof of GPU
availability or support for every macOS release.

**Linux**: NumPy networking and task primitives are testable without MLX or MPS.
Torch can use CPU. `Pool.map`/`submit` do not automatically dispatch to other Macs.

**Windows**: untested. File an issue if you try it.
