# MoM Dist

This repository contains the code accompanying the paper ![`Robust Topological Inference in the Presence of Outliers`](https://arxiv.org/abs/2206.01795).  The code uses the ![`RobustTDA.jl`](https://github.com/sidv23/RobustTDA.jl/) package for computing **MoMDist**-weighted filtrations.


# Getting Started

To get started, first clone the repository and start Julia.

```bash
$ git clone https://github.com/sidv23/momdist.git
$ cd ./momdist
$ julia
```

From the Julia REPL, you can enter the package manager by typing `]`, and activate the project environment with the required packages from the `Project.toml` file as follows.
```julia
julia> ]
pkg> activate .
pkg> instantiate .
```

Alternatively, if you use the `DrWatson.jl` package, then you can quickly activate the project environment as follows.
```julia
julia> using DrWatson
julia> @quickactivate "momdist"
```

## Contents

The [notebooks](./notebooks/) directory contains the Jupyter notebooks for the experiments and simulations. The directory contains the following files:

- [x] [`calibration.ipynb`](./notebooks/calibration.ipynb): Experiment illustrating the auto-tuning procedure using Lepski's method and the resampling based heuristic procedure (See §4.1).

- [x] [`sublevel.ipynb`](./notebooks/sublevel.ipynb): Comparison of sublevel filtration to the weighted-offset filtration for MoMDist (See §4.2). 


- [x] [`high-dim.ipynb`](./notebooks/high-dim.ipynb): High dimensional topological inference in the presence of outliers with **MoMDist** vis-à-vis DTM (See §4.3). 


- [x] [`adversarial.ipynb`](./notebooks/adversarial.ipynb): Illustration of recovering the true signal under adversarial contamination (See §4.4).


- [x] [`influence.ipynb`](./notebooks/influence.ipynb): Influence analysis for **MoMDist**, DTM and RKDE-Distance in the adversarial setting (See §4.5). 


The [scripts](./scripts/) directory contains the `.jl` source-code for the notebooks. All functions prefixed with `rtda.` are imported from the [`RobustTDA.jl` package](./src/rdpg.jl).


## Troubleshooting

The code here uses the [Ripserer.jl](https://github.com/mtsch/Ripserer.jl) backend for computing persistent homology. The exact computation of persistent homology is achieved using the **Alpha** complex which, additionally, uses the ![MiniQHull.jl](https://github.com/gridap/MiniQhull.jl) library, which has a known incompatibility with the Windows operating system (![see here](https://github.com/gridap/MiniQhull.jl/issues/5)). If you're using Windows, then you can either:
1. Use the windows subsystem for linux to run the code here, or
2. You can change the relevant parts of the code to not use the Alpha complex, e.g., `Alpha(Xn)` => `Xn`. 

For any other issues, please click [here](https://github.com/sidv23/momdist/issues/new/choose).