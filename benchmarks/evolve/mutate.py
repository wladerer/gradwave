"""The outer mutation loop: a genome-based evolutionary search over kernel
implementations, scored by the measured evaluator under its correctness oracle.

Rationale. #519 built the corpse-proof *evaluation* half (fitness = measured
wall-time, hard oracle) and proved it on hand-seeded candidates. This is the
*generation* half. A candidate is a point in a small design space -- a
**genome** of independent choices (local-term path, nonlocal contraction
association, working precision, ...) -- and a **builder** renders a genome to a
runnable kernel callable. Mutation perturbs one gene; crossover mixes two
parents; elitist selection keeps the fastest admissible genomes and breeds the
next generation from them. Because scoring goes through the oracle, an invalid
or physics-breaking genome is simply culled -- the loop explores freely without
the risk of shipping a corpse.

Two mutator backends share the #519 evaluator:
- this genome operator loop -- fully autonomous, runs in-process on asus, no
  external model; it enumerates a small space and genuinely evolves a large one.
- LLM/agent-authored candidate FILES (the file-per-candidate path in
  ``driver.py``) -- open-ended novelty, driven by spawning a mutator agent.

The builders below intentionally admit some genomes that are only
tolerance-valid (e.g. complex64) or even wrong: the oracle marks them
inadmissible, so the cost of a bad gene is a wasted evaluation, never a bad
result.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from evolve.evaluator import EvalResult, Problem, evaluate
from gradwave.core.batch import becp_b
from gradwave.solvers.davidson import _eigh_subspace

Genome = dict[str, str]

# ---------------------------------------------------------------------------
# H-apply genome + builder
# ---------------------------------------------------------------------------
HAPPLY_GENES: dict[str, list[str]] = {
    "local": ["fft", "toeplitz"],           # local V.psi path
    "nl": ["three_tensor", "fold_dij", "fold_becp"],  # KB contraction association
    "nl_impl": ["matmul", "einsum"],        # <beta|psi> via becp_b matmul or einsum
    "dtype": ["c128", "c64"],               # apply working precision
}
HAPPLY_BASELINE: Genome = {"local": "fft", "nl": "three_tensor",
                           "nl_impl": "matmul", "dtype": "c128"}


def build_apply(g: Genome) -> Callable:
    cast = torch.complex64 if g.get("dtype") == "c64" else None

    def apply(H, c):
        c0 = c
        if cast is not None:
            c = c.to(cast)
        t_r, v_eff, p, dij = H._tables(c.dtype)
        out = t_r[:, None, :] * c
        if g["local"] == "toeplitz":
            out = out + H._local_toep(c)
        else:
            H._local_fft_into(c, out, v_eff)
        if p.shape[1]:
            if g["nl_impl"] == "einsum":
                b = torch.einsum("kpg,kbg->kbp", p.conj(), c)
            else:
                b = becp_b(p, c)
            if g["nl"] == "fold_dij":
                dijp = torch.einsum("pq,kqg->kpg", dij, p)
                out = out + torch.einsum("kbp,kpg->kbg", b, dijp)
            elif g["nl"] == "fold_becp":
                bd = torch.einsum("kbp,pq->kbq", b, dij)
                out = out + torch.einsum("kbq,kqg->kbg", bd, p)
            else:
                out = out + torch.einsum("kbp,pq,kqg->kbg", b, dij, p)
        if H.hub_q is not None and H.hub_dij is not None:
            hq, hq_conj, hd = H._hub_tables(c.dtype)
            bh = torch.einsum("kpg,kbg->kbp", hq_conj, c)
            out = out + torch.einsum("kbp,pq,kqg->kbg", bh, hd, hq)
        out = out * H.bk.mask[:, None, :]
        return out.to(c0.dtype) if cast is not None else out

    return apply


# ---------------------------------------------------------------------------
# Rayleigh-Ritz genome + builder
# ---------------------------------------------------------------------------
RR_GENES: dict[str, list[str]] = {
    "s_build": ["einsum", "matmul"],
    "symmetrize": ["explicit", "uplo_l"],   # explicit 0.5(s+s^H) vs rely on eigh UPLO='L'
    "upcast": ["always", "if_needed"],
}
RR_BASELINE: Genome = {"s_build": "einsum", "symmetrize": "explicit", "upcast": "always"}


def build_rr(g: Genome) -> Callable:
    def rr(q, hq, nw):
        if g["s_build"] == "matmul":
            s = torch.matmul(q.conj(), hq.mT)
        else:
            s = torch.einsum("kig,kjg->kij", q.conj(), hq)
        if g["symmetrize"] == "explicit":
            s = 0.5 * (s + s.conj().transpose(-1, -2))
        if g["upcast"] == "always" or s.dtype != torch.complex128:
            s = s.to(torch.complex128)
        w, u = _eigh_subspace(s)
        return w[:, :nw].real.to(q.real.dtype), u[:, :, :nw].to(q.dtype)

    return rr


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------
def genome_key(g: Genome) -> tuple:
    return tuple(sorted(g.items()))


def short_name(g: Genome) -> str:
    return "|".join(f"{v}" for _, v in sorted(g.items()))


def random_genome(genes: dict[str, list[str]], rng: random.Random) -> Genome:
    return {k: rng.choice(v) for k, v in genes.items()}


def mutate(g: Genome, genes: dict[str, list[str]], rng: random.Random) -> Genome:
    child = dict(g)
    axis = rng.choice(list(genes))
    child[axis] = rng.choice(genes[axis])
    return child


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------
@dataclass
class GenerationRecord:
    gen: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    best_name: str | None = None
    best_speedup: float | None = None


def evolve(
    problem: Problem,
    genes: dict[str, list[str]],
    builder: Callable[[Genome], Callable],
    baseline_genome: Genome,
    *,
    pop: int = 8,
    gens: int = 6,
    elite: int = 3,
    reps: int = 100,
    warmup: int = 10,
    seed: int = 0,
    patience: int = 3,
) -> tuple[list[GenerationRecord], dict[str, Any], float, list[dict[str, Any]]]:
    """Run the evolutionary search; return (per-generation log, best result).

    Elitist with a genome cache: every genome is scored at most once (the space
    is small, so caching turns 'evolution' into 'measured enumeration' on small
    spaces and a guided sample on large ones -- both correct). Stops after
    ``patience`` generations with no improvement.
    """
    rng = random.Random(seed)
    cache: dict[tuple, tuple[EvalResult, Genome]] = {}

    def score(g: Genome) -> tuple[EvalResult, Genome]:
        key = genome_key(g)
        if key not in cache:
            res = evaluate(short_name(g), builder(g), problem, warmup=warmup, reps=reps)
            cache[key] = (res, g)
        return cache[key]

    base_res, _ = score(baseline_genome)
    base_ms = base_res.fitness_ms
    if base_ms is None:
        raise RuntimeError(f"baseline genome inadmissible: {base_res.error or base_res.max_err}")

    population = [baseline_genome] + [random_genome(genes, rng) for _ in range(pop - 1)]
    log: list[GenerationRecord] = []
    best: dict[str, Any] | None = None
    no_improve = 0

    for gen in range(gens):
        scored = [score(g) for g in population]
        rec = GenerationRecord(gen=gen)
        graded = []
        for res, g in scored:
            sp = (base_ms / res.fitness_ms) if (res.admissible and res.fitness_ms) else None
            rec.rows.append({"name": res.name, "genome": g, "admissible": res.admissible,
                             "max_err": res.max_err, "ms": res.fitness_ms, "speedup": sp,
                             "error": res.error})
            if sp is not None:
                graded.append((sp, res, g))
        graded.sort(key=lambda t: -t[0])
        if graded:
            top_sp, top_res, top_g = graded[0]
            rec.best_name, rec.best_speedup = short_name(top_g), top_sp
            cand = {"speedup": top_sp, "genome": top_g, "ms": top_res.fitness_ms,
                    "max_err": top_res.max_err, "name": short_name(top_g)}
            if best is None or top_sp > best["speedup"]:
                best, no_improve = cand, 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        log.append(rec)
        if no_improve >= patience:
            break

        elites = [g for _, _, g in graded[:elite]] or [baseline_genome]
        children: list[Genome] = []
        seen = {genome_key(g) for g in elites}
        tries = 0
        while len(children) < max(0, pop - len(elites)) and tries < 50:
            tries += 1
            if len(elites) >= 2 and rng.random() < 0.5:
                child = crossover(rng.choice(elites), rng.choice(elites), rng)
            else:
                child = mutate(rng.choice(elites), genes, rng)
            k = genome_key(child)
            if k not in seen:
                seen.add(k)
                children.append(child)
        population = elites + children

    all_results = []
    for res, g in cache.values():
        sp = (base_ms / res.fitness_ms) if (res.admissible and res.fitness_ms) else None
        all_results.append({"name": short_name(g), "genome": g, "admissible": res.admissible,
                            "max_err": res.max_err, "ms": res.fitness_ms, "speedup": sp,
                            "error": res.error})
    all_results.sort(key=lambda r: (r["speedup"] is None, -(r["speedup"] or 0.0)))
    best = best or {"speedup": 1.0, "genome": baseline_genome, "name": "baseline",
                    "ms": base_ms, "max_err": base_res.max_err}
    return log, best, base_ms, all_results
