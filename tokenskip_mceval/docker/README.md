# Derived McEval Docker image

The official `multilingualnlp/mceval` image ships with several languages
missing or broken. This folder builds `mceval-full`, adding:

| Language | Source |
|---|---|
| Go, Haskell, Rust, Scala, Groovy, Node | Ubuntu 18.04 apt |
| TypeScript, CoffeeScript | npm global |
| Kotlin | JetBrains release tarball |
| Julia | julialang.org tarball |
| Dart | Google APT repo (re-keyed) |
| Swift | swift.org Linux 18.04 tarball |

It also fixes the Python branch in McEval's `excute.py` (`python` → `python3`).

## Build

```bash
cd tokenskip_mceval/docker
docker build -t mceval-full .
```

First build is ~10–20 min (downloads Swift + Julia + Kotlin tarballs,
several hundred MB).

## Use

```bash
cd tokenskip_mceval
IMAGE=mceval-full \
RESULTS_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/baseline \
SAVE_DIR=../outputs/Qwen2.5-Coder-7B-Instruct/eval-full \
bash scripts/run_eval_docker.sh
```

The build smoke-tests every binary on PATH before finishing, so any
download failure surfaces immediately rather than during eval.
