# Architecture

This page is a map of the code: the layers it is built in, how the modules
group into subsystems, and the path a calculation takes from a YAML file to its
output. Read it when you are trying to find where a feature lives or how the
pieces connect. The [Inputs and outputs](io.md) page covers the input schema;
this page covers the machinery behind it.

## The three layers

gradwave is organized so that automatic differentiation can pass cleanly through
the physics while the iterative solvers stay out of the tape. Every module
belongs to one of three layers, and the contract between them is what keeps the
gradients exact.

```mermaid
flowchart TB
    subgraph C["Layer C · orchestration"]
        direction LR
        cli["cli.py"]
        inputs["inputs.py"]
        api["api.py"]
        calc["calculator.py"]
    end
    subgraph B["Layer B · iterative solvers (torch.no_grad)"]
        direction LR
        scf["scf/ — SCF drivers"]
        solvers["solvers/ — Davidson, Chebyshev"]
    end
    subgraph A["Layer A · differentiable physics"]
        direction LR
        core["core/ — Hamiltonian, density, XC"]
        pseudo["pseudo/ — UPF, projectors"]
    end

    C -->|"builds a System, calls a driver"| B
    B -->|"applies H, forms rho under no_grad"| A
    A -.->|"one autograd pass on the energy = forces"| C

    classDef cLayer fill:#e8eaf6,stroke:#3949ab,color:#1a237e;
    classDef bLayer fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef aLayer fill:#e0f2f1,stroke:#00897b,color:#004d40;
    class cli,inputs,api,calc cLayer;
    class scf,solvers bLayer;
    class core,pseudo aLayer;
```

**Layer A — differentiable physics (`core/`, `pseudo/`).** Pure PyTorch, no
in-place mutation, everything traceable by autograd. The Hamiltonian apply, the
density, the exchange-correlation functionals, the structure factors, and the
per-term energy assembly live here; `core/energies/total.py` is the single
function autograd differentiates. This is the only layer the tape ever sees, so
a reverse-mode pass through the total energy yields the forces, and
differentiating the forces again yields the Hessian, one Hessian-vector product
per column.

**Layer B — iterative solvers (`scf/`, `solvers/`).** The self-consistency loop
and the eigensolvers run under `torch.no_grad`. Autograd must never trace
Davidson or the SCF iteration; instead the converged density and wavefunctions
are detached, and the gradient is recovered analytically (stationarity for
`dE/dtheta`, an implicit-differentiation solve for the density response). This
boundary is the single most important invariant in the codebase.

**Layer C — orchestration (`api.py`, `inputs.py`, `cli.py`, `calculator.py`).**
The user-facing surface. `inputs.py` parses YAML into a frozen `Input`, `api.py`
builds the `System` and dispatches the task, and `cli.py` and `calculator.py`
are the two front doors (command line and ASE calculator).

Post-SCF properties (`postscf/`) sit above this stack: each one takes a
converged result and computes a property, often by running one more autograd
pass through the Layer A energy (forces, stress, phonons, dielectric response).

## Subsystem map

The same modules, grouped by what they do rather than by layer.

```mermaid
flowchart TB
    subgraph io["Entry & I/O"]
        direction LR
        i1["cli · inputs · templates"]
        i2["api · calculator"]
        i3["output · analysis · checkpoint"]
    end

    subgraph phys["Core physics · Layer A"]
        direction LR
        p1["hamiltonian · density · energies/"]
        p2["xc/ — lda, pbe, spin, noncollinear, learnable"]
        p3["structure · occupations · grids"]
        p4["ylm · gaunt · symmetry · kpoints"]
    end

    subgraph pp["Pseudopotentials"]
        direction LR
        pp1["upf · upf_paw (parsers)"]
        pp2["kb · local · atomic (projectors)"]
    end

    subgraph solve["Solvers & SCF · Layer B"]
        direction LR
        s1["davidson · chebyshev · precond"]
        s2["loop · mixing · guess"]
        s3["uspp (USPP/PAW) · noncollinear · implicit"]
    end

    subgraph props["Post-SCF properties"]
        direction LR
        r1["bands · dos · pdos · irreps"]
        r2["forces · stress · phonons"]
        r3["dielectric · magnetism · mae · hubbard_u"]
        r4["error estimates (basis, SCF, smearing)"]
    end

    subgraph hyb["Exact exchange & hybrids"]
        direction LR
        h1["isdf · isdf_k (pair factorization)"]
        h2["exchange · exchange_multik · coulomb_kernel"]
        h3["hybrid — self-consistent hybrid SCF"]
    end

    io --> solve
    pp --> solve
    solve --> phys
    solve --> props
    phys --> props
    phys --> hyb
    solve --> hyb

    classDef ioc fill:#e8eaf6,stroke:#3949ab,color:#1a237e;
    classDef physc fill:#e0f2f1,stroke:#00897b,color:#004d40;
    classDef ppc fill:#eceff1,stroke:#546e7a,color:#263238;
    classDef solvec fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef propsc fill:#f3e5f5,stroke:#8e24aa,color:#4a148c;
    classDef hybc fill:#fce4ec,stroke:#c2185b,color:#880e4f;
    class i1,i2,i3 ioc;
    class p1,p2,p3,p4 physc;
    class pp1,pp2 ppc;
    class s1,s2,s3 solvec;
    class r1,r2,r3,r4 propsc;
    class h1,h2,h3 hybc;
```

## Anatomy of a run

What happens between `gradwave input.yaml` and the files in `out/`. The formalism
(norm-conserving versus ultrasoft/PAW) is detected from the UPF files, so one
input schema drives both paths and they rejoin at the summary.

```mermaid
flowchart TB
    yaml["input.yaml"] --> load["load_input()<br/>parse, validate, resolve pseudos"]
    load --> inp["Input (frozen dataclass)"]
    inp --> build["build_system()<br/>grids, k-points, projectors,<br/>detect NC vs USPP/PAW"]
    build --> task{"task"}

    task -->|scf| scf["run_scf"]
    task -->|relax| relax["run_relax<br/>(BFGS/FIRE, autograd forces)"]
    task -->|bands| bands["bands task<br/>(run_scf, then fixed-potential solve)"]
    task -->|magnetism| mag["run_magnetism<br/>(spin SCF + exchange)"]

    relax --> scf
    bands --> scf
    mag --> scf

    scf --> fork{"formalism"}
    fork -->|norm-conserving| ncscf["scf loop (scf/loop.py)"]
    fork -->|USPP/PAW| uscf["scf_uspp (scf/uspp.py)"]

    ncscf --> res["converged result"]
    uscf --> res
    res --> extras["optional: projections (pdos),<br/>error estimates"]
    extras --> summary["build_summary()"]
    summary --> out["out/: task.json · task.out · checkpoint.pt"]

    classDef entry fill:#e8eaf6,stroke:#3949ab,color:#1a237e;
    classDef driver fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef output fill:#f3e5f5,stroke:#8e24aa,color:#4a148c;
    class yaml,load,inp,build entry;
    class scf,relax,bands,mag,ncscf,uscf,res driver;
    class summary,out,extras output;
```

## Inside the SCF loop

The self-consistency cycle is the heart of Layer B. It runs under `torch.no_grad`;
the density and wavefunctions it returns are detached, and forces come afterward
from a single autograd pass through the three position-dependent energy terms
(the structure factors, the Ewald sum, and the nonlocal projector phases).

```mermaid
flowchart LR
    guess["initial density<br/>(guess.py or restart)"] --> ham["build Hamiltonian<br/>(core/hamiltonian.py)"]
    ham --> diag["diagonalize<br/>(solvers/davidson.py)"]
    diag --> occ["occupations & Fermi level<br/>(core/occupations.py)"]
    occ --> rho["new density<br/>(core/density.py)"]
    rho --> mix["mix<br/>(scf/mixing.py: Pulay + Kerker)"]
    mix --> conv{"converged?<br/>dE, drho"}
    conv -->|no| ham
    conv -->|yes| done["converged result"]

    done -.->|"autograd on 3 position terms"| forces["forces / stress"]

    classDef loop fill:#fff3e0,stroke:#ef6c00,color:#e65100;
    classDef grad fill:#e0f2f1,stroke:#00897b,color:#004d40;
    class guess,ham,diag,occ,rho,mix,done loop;
    class forces grad;
```

## Where each feature lives

If you are trying to find the module behind a capability, start here. The input
column is the YAML keyword (or API entry point) that turns the feature on; see
[Inputs and outputs](io.md) for the full schema.

| Feature | Turn it on with | Lives in | Tutorial |
|---|---|---|---|
| Single-point SCF | `task: scf` | `scf/loop.py`, `scf/uspp.py` | [Cookbook](cookbook.md) |
| Exchange-correlation | `xc: lda \| pbe` | `core/xc/` | [Learning XC](learning-xc.md) |
| Geometry / cell relaxation | `task: relax` | `postscf/forces.py`, `postscf/stress.py` | [Geometry optimization](geometry-optimization.md) |
| Band structure | `task: bands` | `postscf/bands.py`, `postscf/irreps.py` | [Symmetry](symmetry.md) |
| Density of states | `task: bands` / plot | `postscf/dos.py` | [Cookbook](cookbook.md) |
| Projected DOS | `projections:` | `postscf/pdos.py`, `core/hubbard.py` | [Cookbook](cookbook.md) |
| Collinear spin | `nspin: 2`, `start_mag:` | `core/xc/spin.py`, `scf/loop.py` | [Magnetism](magnetism.md) |
| Noncollinear / SOC | `noncollinear: true` | `scf/noncollinear.py`, `core/spinor_proj.py` | [Noncollinear & SOC](noncollinear-soc.md) |
| Exchange couplings (J) | `task: magnetism` | `postscf/magnetism.py`, `postscf/spin_exchange.py` | [Magnetism](magnetism.md) |
| Magnetic anisotropy | API | `postscf/mae.py` | [MAE](mae.md) |
| Hubbard U (+U and its response) | API | `core/hubbard.py`, `postscf/hubbard_u.py` | [Hubbard U](hubbard-u.md) |
| Phonons | API | `postscf/phonons.py`, `postscf/hessian.py` | [Cookbook](cookbook.md) |
| Dielectric / Born charges | API | `postscf/dielectric.py` | [Cookbook](cookbook.md) |
| Symmetry reduction | `symmetry: true` | `symmetry.py`, `scf/paw_symmetry.py` | [Symmetry](symmetry.md) |
| Smearing (metals) | `smearing:` | `core/occupations.py` | [Cookbook](cookbook.md) |
| Learnable functionals | API | `core/xc/learnable.py`, `scf/implicit.py` | [Learning XC](learning-xc.md) |
| Hybrid functionals (exact exchange) | `hybrid_scf` (Python) | `postscf/hybrid.py`, `postscf/isdf.py`, `postscf/exchange.py` | — |
| Learnable hybrid (α, ω) | Python | `postscf/exchange_multik.py`, `postscf/isdf_k.py`, `postscf/coulomb_kernel.py` | — |
| Basis / SCF error estimates | `error_estimate: true` | `postscf/convergence_error.py`, `postscf/discretization_error.py` | [Error estimation](error-estimation.md) |
| Restart / checkpoints | `restart:` | `checkpoint.py` | [Inputs and outputs](io.md) |
| ASE calculator | Python | `calculator.py` | [Cookbook](cookbook.md) |

## Pseudopotentials

The formalism is chosen by the UPF file, not the input. `pseudo/upf.py` parses
norm-conserving UPF v2; `pseudo/upf_paw.py` parses ultrasoft and PAW. The
projector modules (`kb.py`, `local.py`, `atomic.py`) build the Kleinman-Bylander
nonlocal projectors, the local potential, and the atomic-orbital projectors used
for the projected DOS and Hubbard manifolds. Everything downstream of the parser
is formalism-agnostic until the SCF driver forks.

## Exact exchange and hybrid functionals

Hybrid functionals add a fraction of exact (Fock) exchange to a semilocal
functional. The naive Fock build scales as O(N⁴), so this subsystem is built on
interpolative separable density fitting (ISDF), the plane-wave form of tensor
hypercontraction, which factorizes the orbital-pair products into a low-rank
form. `isdf.py` and `isdf_k.py` do that factorization at Γ and across a k-mesh;
`exchange.py` and `exchange_multik.py` build the Fock operator (and its
adaptively-compressed ACE form) from it; `coulomb_kernel.py` supplies the
range-separated kernel that distinguishes full (PBE0-style) from screened
(HSE-style) exchange. `hybrid.py` ties these into the SCF loop with
`hybrid_scf`, so gradwave can solve a hybrid self-consistently rather than only
evaluate exchange on a fixed density. The mixing fraction α and screening length
ω are differentiable parameters (`HybridExchangeParams`), which makes a
*learnable* hybrid trainable end to end, the same stationarity argument the
learnable-XC slot uses.

This subsystem is reached from Python today (`hybrid_scf`,
`HybridExchangeParams`); it is not yet wired into the YAML input schema. The
`xc:` key accepts `lda`, `pbe`, and `r2scan`; the hybrids stay Python-only.

## Tests and fixtures

The physics is pinned against Quantum ESPRESSO. `tests/fixtures/qe/` holds the
reference outputs and the UPF pseudopotentials the tests read; the same
pseudopotential directory backs the `examples/` inputs and the `gradwave init`
templates. Tests are tiered by cost: the default fast tier runs small
configurations in seconds, `-m standard` runs real SCFs, and `-m slow` runs the
converged QE comparisons (minutes). When a test references a QE fixture, it is
asserting that gradwave reproduces that reference to a stated tolerance.
