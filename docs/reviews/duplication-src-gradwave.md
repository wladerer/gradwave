# Consensus review — objective: duplication, target: `src/gradwave`

Run 2026-08-11 on the post-reorg tree (main @ 64ae88d), model claude-opus-4-8, 334 agents / 9,761,098 tokens.

**Stats:** 169 files, 19 finder slices; 90 candidate issues → **65 survived** refute-to-survive verification (0 unverified passthrough); 2 cross-file issues; 3 remediation themes.

## Local issues (verified)

### [high] `src/gradwave/core/energies/esm.py:445` — Capacitor ESM energy diverged between energy/force path and stress path (net-charge term dropped)

esm_energy() capacitor mode (line 404) delegates to _capacitor_grounded_de(), which splits rho_tot = rho_n (neutral) + rho_q (smooth net charge) and adds the spectral net-charge terms `(rho_n*s_q).sum() + 0.5*(rho_q*s_q).sum()` (lines 362-376) so a charged cell is handled correctly. esm_energy_strained() capacitor mode (lines 445-448) instead recomputes the grounded correction INLINE as `esm_delta_potential + hartree_potential_capacitor - hartree_potential_esm` (identical to the _matched_cap_dv helper, lines 339-343) and returns `0.5*(rho_tot*dv_grounded).sum()*dvol` over the WHOLE rho_tot with no neutral/charge split. For a net-neutral cell the two agree; for a charged capacitor slab (constant-potential / biased electrochemistry — the advertised use case) the stress functional (esm_energy_strained) is no longer the strain-derivative of the energy functional (esm_energy). This is a fixed-in-one-not-the-other drift: esm_energy was upgraded to _capacitor_grounded_de for net charge, esm_energy_strained was left on the old plain-matched form.

**Direction:** Give esm_energy and esm_energy_strained a single shared capacitor-energy core parameterized by (g_cart/g2/volume or grid) so both consume _capacitor_grounded_de's net-charge-aware split; esm_energy_strained should call the same routine (threading the strained cell/volume) rather than its inline _matched_cap_dv copy.

### [high] `src/gradwave/postscf/hubbard_u.py:138` — _bare_response_occ double-counts spin for nspin=1, diverging from its twin _site_occupations

_site_occupations (l.82-93) and _bare_response_occ (l.96-142) are two parallel routines returning the SAME quantity: per-site total occupation N_I=Σ_σ Tr[n^{Iσ}]. They must agree in units because _fd_response_column (l.145-163) feeds the first to χ and the second to χ0, both /2α, into U=(χ0⁻¹−χ⁻¹). But they weight the occupation-matrix trace differently for nspin=1. occupation_matrices (core/hubbard.py:170) weights by occ 'in the channel's electron units' i.e. g·f, and occupations_and_entropy (core/occupations.py:157) returns degeneracy*occupation = g·f. _site_occupations passes w=0.5*res.occupations = 0.5*(g·f)=f (l.87) then multiplies the summed trace by 2 (l.93) → N_I=2f (correct: a fully-filled d-shell gives 5*2=10). _bare_response_occ passes w = occupations_and_entropy(...,degeneracy=g_spin)=g·f=2f (l.138) and ALSO multiplies by 2 (l.142) → N_I=4f (a filled d-shell would give 20, exceeding the 10-electron capacity). The in-line comment 'the old ×g×½ was a no-op' dropped the 0.5 that _site keeps, so _bare is 2× too large for nspin=1 (nspin=2 is fine: g_spin=1, ×1 tail). This corrupts χ0 relative to χ and hence the linear-response U from linear_response_u for the common nspin=1 case.

**Direction:** Collapse the two occupation-sum routines onto one shared helper (single contract: feed occupation_matrices the per-spin fractional weight f, apply the spin ×2 once for nspin=1). Concretely _bare should pass 0.5*w (or drop the trailing ×2) to match _site_occupations; add a reduction test asserting _site_occupations and _bare_response_occ agree at α=0 for an nspin=1 insulator.

### [high] `src/gradwave/solvers/davidson.py:321` — Shared _rr() Rayleigh-Ritz is missing the issue #136 fp64-upcast fix that davidson_batched's inline copy carries

davidson_batched has its OWN inline Rayleigh-Ritz at davidson.py:444-450 that force-upcasts the small subspace matrix and eigensolve to fp64: `s = (0.5*(s + s.conj().transpose(-1,-2))).to(torch.complex128); w,u = _eigh_subspace(s)` with the comment 'Rayleigh-Ritz subspace reduction ALWAYS in fp64 (issue #136): ... an fp32 eigh of a supercell insulator block ... yields inaccurate Ritz rotations in the complex64 draft. Upcast only the small matrix and the eigensolve.' The extracted `_rr()` (davidson.py:319-322), which the docstring says is 'Shared by CheFSI and LOBPCG', does NOT upcast: `s = torch.einsum('kig,kjg->kij', q.conj(), hq); s = 0.5*(s + s.conj().transpose(-1,-2)); w,u = _eigh_subspace(s)`. When q/hq are complex64, `_eigh_subspace` runs the eigh in complex64. Both chebyshev_filtered_batched_ms and lobpcg_batched_ms run a complex64 draft phase (chebyshev.py:257-262, lobpcg.py:170-172) whose inner rounds call `_rr` at chebyshev.py:190 and lobpcg.py:103,130 -- so the exact fp32-eigh regime issue #136 was fixed for in davidson_batched is reintroduced for CheFSI/LOBPCG through the shared helper. This is the highest-value clone class: two parallel implementations of one operator (inline RR in davidson_batched vs extracted _rr) where a numerical fix landed in one but not the other.

**Direction:** Make `_rr` the single canonical Rayleigh-Ritz home and fold the #136 contract into it: upcast the (nk,dim,dim) matrix and the eigh to complex128 unconditionally, return eigenvalues/eigenvectors downcast to the caller's block dtype (mirroring davidson_batched:445-448). Then delete davidson_batched's inline copy (444-450) and route it through `_rr` too, so there is exactly one RR with one always-fp64 contract.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/api/elastic.py:387` — eV/Å³→GPa factor 160.2176634 hardcoded inline; redefined 5x across repo, one copy drifted

api/elastic.py:387 `resid_gpa = float(np.abs(sigma_ref).max()) * 160.2176634` inlines the eV/Å³→GPa conversion as a bare magic number — even though this very function imports from postscf.elastic (line 156), which already defines the named `EV_A3_TO_GPA = 160.2176634` (postscf/elastic.py:43). The same constant is independently redefined in postscf/eos.py:27 (`EV_A3_TO_GPA = 160.2176634`) and, as the kbar variant, in postscf/stress.py:73 and postscf/stress_error.py:58 (`EV_A3_TO_KBAR = 1602.176634`). Critically, api/relax.py:252 uses a DIVERGED value: `gpa_to_ev_a3 = 1.0 / 160.21766208` — 160.21766208 vs the CODATA-2018-correct 160.2176634 (differ at the 7th digit; 160.2176634 = 1.602176634e-19 J/eV × 1e30 Å³/m³ / 1e9). The map declares constants.py the single source of truth for all unit conversions, but this eV/Å³↔GPa factor lives in neither constants.py nor a shared home. Collapse to one `constants.EV_A3_TO_GPA` (CODATA 2018 = 160.2176634); api/elastic.py:387 should import it, relax.py:252 should divide by it, and the kbar sites derive as ×10.

**Direction:** Add EV_A3_TO_GPA (=160.2176634) to constants.py as the single source; import it at api/elastic.py:387, api/relax.py:252 (fixing the drifted 160.21766208), postscf/elastic.py:43, postscf/eos.py:27; derive EV_A3_TO_KBAR = 10×EV_A3_TO_GPA in stress.py/stress_error.py.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/core/hubbard.py:159` — KB projector column + phase construction duplicated between hamiltonian.py and hubbard.py

hubbard.py builds the atomic-orbital projector with the identical KB formula used in hamiltonian.py: hubbard.py:159 `pref = (4.0*math.pi/math.sqrt(vol))*_MINUS_I_POW[l]` then hubbard.py:162 `q_free[...] = pref*(f_by_sp[s]*yl).to(CDTYPE)` mirrors hamiltonian.py:74-75 `pref = (4.0*math.pi/math.sqrt(volume))*_MINUS_I_POW[l]` / `cols_a.append(pref*(f*yl).to(CDTYPE))`. The phase application is likewise cloned: hubbard.py:86-90 `hubbard_projectors` (`phase_arg = einsum('kgd,ad->kga', kpg, positions); phases = exp(complex(0,-phase_arg)); q_free*phases[...]`) is the batched twin of hamiltonian.py:108-112 `projectors` (`phase_arg = kpg@positions.T; phases = exp(complex(0,-phase_arg)); f_ylm_phase_free*phases[...]`). Both docstrings explicitly state the +U path 'reuses the exact Kleinman-Bylander structure' / is 'exactly like the KB ProjectorData/projectors() split' yet reimplement it. lm ordering `l*l+m` is also re-spelled (hamiltonian.py:59 vs hubbard.py:162). Drift hazard: a fix to the (4pi/sqrt(Omega))(-i)^l prefactor or the e^{-i(k+G).tau} sign would have to be made in two force-critical places.

**Direction:** Collapse into one shared builder in core/hamiltonian.py: a `projector_column(pref_l, F, Ylm)` helper returning the phase-free (4pi/sqrt(Omega))(-i)^l Y_lm F factor, and one `apply_phase(kpg, positions, atom_of_col)` used by both ProjectorData and HubbardData (batched by broadcasting). Contract: phase-free factor is position-independent and frozen per geometry; positions enter only via e^{-i(k+G).tau}.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/core/spinor_proj.py:176` — build_so_projectors and strained_so_projector_cols are twin implementations of FR spinor projector assembly

The column-building loop in strained_so_projector_cols (lines 203-216) is a line-for-line clone of the loop in build_so_projectors (lines 158-169): identical prefactor `(4.0*math.pi) * _MINUS_I_POW[l]` (lines 160 vs 209), identical `_cg(l,j,mj)` CG weighting, identical `qu/qd = base * (c_* * ylm_c[..., l*l+l+m_*])` fill with the same `if m_up/m_dn is not None` guards, identical `torch.cat([qu,qd])` doubled-axis stack, and the same e^{-i(k+G)·tau} phase and 1/sqrt(Omega) vol_norm. The only real differences are batched-over-k vs single-k and where vol_norm is multiplied (line 172 end vs line 209 inline). The module docstring even says strained_so_projector_cols 'Mirrors build_so_projectors'. Any fix to the phase sign, the (-i)^l/4pi prefactor, the doubled-axis ordering, or the CG convention must be applied in BOTH or the SCF projectors and the stress-strain projectors silently diverge (wrong forces/stress with correct energy).

**Direction:** Extract one `_assemble_so_columns(col_meta, lmax, f_of_channel, ylm_c, phases, vol_norm)` helper that owns the (4pi)(-i)^l prefactor, CG weights, doubled [up-npw, dn-npw] axis, and column order; have both the batched SCF builder and the single-k strain builder call it, differing only in how they supply the radial form factor (precomputed table vs beta_of_g) and the k-batch axis.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/io/analysis.py:138` — dos_frame reimplements the canonical Gaussian-broadened DOS (window padding + kernel) instead of reusing postscf.pdos

analysis.dos_frame open-codes the whole DOS broadening: the default window `(eig.min()-10*width, eig.max()+10*width)` (line 139), the `np.linspace` grid (line 140), the spin degeneracy `g_spin = 2.0 if nspin==1 else 1.0` (line 141), and the Gaussian sum `np.exp(-0.5*((grid[:,None]-e[None,:])/width)**2)/(width*np.sqrt(2*np.pi))*w*g_spin` (lines 146-147). This is a structural clone of postscf/pdos.py: `_broaden` (pdos.py:126-134) is byte-for-byte the same kernel `exp(-0.5*((grid-e)/width)**2)*inv` with `inv=1/(width*math.sqrt(2*math.pi))`, and `spectral_grid` (pdos.py:138-148) owns the exact same `10*width` padding rule whose docstring says it exists 'so the padding rule lives in one place ... shared by the DOS functions here and cohp._finalize'. analysis.dos_frame is a second, un-shared copy that already breaks that contract and will silently drift if the padding/normalization convention changes (e.g. window width, or the g_spin=2/nspin convention used at pdos g_spin). Canonical home: postscf.pdos.spectral_grid + a shared per-state broadener; dos_frame should build `per_state = kw-broadcast * g_spin` and call it.

**Direction:** Have dos_frame import and call postscf.pdos.spectral_grid for (window, grid) and reuse the _broaden kernel (promote _broaden to a public helper), passing per_state = w*g_spin, so the padding rule and Gaussian normalization exist in exactly one place.

### [medium] `src/gradwave/postscf/dielectric.py:446` — Three parallel E-field DFPT drivers (nspin=1 / nspin=2 / SOC) reimplement the same Sternheimer+screening+Born loop

dielectric_born (149, nspin=1), _dielectric_born_spin (446, nspin=2) and _dielectric_born_soc (639, SOC) each independently build: the ξ=P_c r ψ Sternheimer solve, the Anderson fixed-point screening loop on u=K_Hxc[Δρ], the drho accumulation `w*(kw*(psi.conj()*dpsi).real).sum()/vol`, the eps column `eye - (Nπ E2/vol)*col`, and the Born-charge autograd backward (t_loc=local_energy(...)+t_nl, grad, born[s,a,a]+=charges[s]). They differ only in channel count and the prefactor pair (16π/4 for f=2 vs 8π/2 for f=1) and which projector/kernel primitive is called. ~350 lines of parallel code. A real divergence already exists: IBZ symmetry (_field_response_symmetrized/VectorFieldSymmetrizer) is implemented only in the nspin=1 path; spin and SOC raise NotImplementedError — so a symmetry fix has to be ported three ways.

**Direction:** Extract one driver parameterized by (n_channels, eps_prefactor, density_prefactor, projector-builder, k_hxc kernel, g_to_r/box_to_sphere pair); the nspin=1/2/SOC entry points supply those. Collapses the Born-charge backward and the screening loop to a single implementation.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/postscf/uspp_implicit.py:717` — Composite (grid, becsum, +U) pack/unpack reimplemented three times

The flat<->(per-spin grid, per-spin per-atom becsum, per-channel Re/Im hub) layout has three independent implementations: newton._pack/_unpack (postscf/newton.py:47-97, the canonical one used by uspp_position.py:471-476 via _pack_all/_unpack_all), the inline split/join in uspp_implicit.py:717-758, and the byte-identical inline split/join in uspp_softmode.py:132-170. All three encode the same block order and the same 'complex Hermitian hub matrix -> (Re,Im) pair' tail convention; the docstrings even cross-reference each other ('Layout mirrors the mixer', 'Same layout as ... uspp_implicit''s adjoint', 'mirroring the USPP adjoint''s join/split'). Because this vector crosses the χ̃/K adjoint boundary, any edit to the hub tail ordering or block order in one copy (e.g. adding a spin-off-diagonal block) silently desynchronizes the others, corrupting gradients rather than crashing.

**Direction:** Make newton._pack/_unpack (or a small CompositeLayout dataclass built from cs: nsp, nbec, hub_dims, nsh) the single home; have uspp_implicit.uspp_density_loss_param_grads and uspp_softmode.build_uspp_screening import it instead of re-deriving split/join.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/postscf/uspp_softmode.py:181` — symmetrize() duplicated verbatim between softmode and implicit adjoints

The per-spin composite symmetrizer — rho block via g_to_r_box(rho_symmetrizer.apply(r_to_g(w)), real=True) and becsum block via [m.real for m in becsum_sym.apply([m.to(CDTYPE)...])] — is written twice with identical logic: uspp_softmode.py:181-193 and uspp_implicit.py:770-785. Both guard on system.rho_symmetrizer/becsum_sym being non-None and both rely on these being self-adjoint projections. A change to how becsum symmetrization is applied (e.g. handling apply_m for a magnetization channel, as scf paths already special-case) would need editing in both.

**Direction:** Hoist to a single _ConvergedUSPP.symmetrize(w_sp, d_bare_sp) method (cs already owns system); both callers invoke it.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/pseudo/radial_torch.py:40` — Spherical Bessel j_l implemented twice (numpy sph_jl vs torch jl_t) with the closed-form trig expressions copy-pasted and the l-range already drifted

radial_torch.py::jl_t (l 40-75) is a hand-maintained mirror of radial.py::sph_jl (l 71-113): the ascending-series recurrence `term = term * (-0.5*x2) / (k*(2*l+2*k+1))` is byte-identical, and the closed trig forms for l=0..4 are the same polynomials in 1/x·(sin,cos) written twice (e.g. l=3: `(15/xb**4 - 6/xb**2)*s - (15/xb**3 - 1/xb)*c` in numpy vs `(15.0*u2*u2 - 6.0*u2)*s - (15.0*u2*u - u)*c` in torch, u=1/xb). They have ALREADY diverged: sph_jl raises for l>4 (`0 <= l <= 4`) while jl_t supports l<=5 (adds an l==5 branch for the _djl_t derivative). Any fix to the SERIES_X cancellation boundary or a coefficient typo must be applied in both by hand, and the l-range mismatch is exactly the drift class the rubric targets.

**Direction:** Collapse to one source of the per-l closed-form coefficients (a table of the sin/cos polynomial coefficients keyed by l, alongside DOUBLE_FACTORIAL/SERIES_X/SERIES_TERMS already in _bessel_data), consumed by both the numpy setup path and the torch differentiable path, so the trig forms and l-range live in exactly one place.

*(1 of 3 skeptics dissented)*

### [medium] `/home/wladerer/github/gradwave/src/gradwave/scf/learned_precond.py:238` — _diis_unroll_logres re-implements PulayMixer's bordered DIIS solve (parallel copy, drift risk)

_diis_unroll_logres (learned_precond.py 238-251) hand-copies the exact bordered, diagonal-normalized, Tikhonov-regularized DIIS coefficient solve that mixing.PulayMixer.step runs (mixing.py 395-423): both build B_ij from a Kerker-weighted inner product, form d=sqrt(diag).clamp_min(1e-300), bn=B/d/d, add the SAME 1e-10*eye ridge, border with rows/cols 1/d and rhs[-1]=1, solve, then divide the coefficients back by d. The docstring even asserts it is 'the same bordered ... solve mixing.PulayMixer runs'. It is a literal structural clone kept in sync by hand: the whole purpose of fit_multipole is to train a preconditioner against the DIIS the real scf() will run, so if PulayMixer's regularizer (1e-10), metric weight 1/(g2+q0^2), or normalization ever change, this copy silently trains against a stale model and the abstention-gate margins (calibrated to ~0.75 log-unit noise) become meaningless with no test catching it. Canonical home: extract a single `diis_coefficients(B_metric_matrix)` (or a differentiable `pulay_extrapolation_step`) into mixing.py and call it from both PulayMixer.step and _diis_unroll_logres.

*(1 of 3 skeptics dissented)*

### [medium] `/home/wladerer/github/gradwave/src/gradwave/scf/layout.py:60` — MixLayout pack/unpack re-implemented as spinor_common.pack/unpack_grid_channels, bypassing the declared single source of truth (plus a name collision)

MixLayout's module docstring calls itself 'the composite mixing vector's single source of truth' and says the packing was previously 're-derived independently ... every copy was one normalization bug waiting to happen'. Yet scf_noncollinear does not use MixLayout: it packs/unpacks its (rho, m-vec) mixing vector via spinor_common.pack_grid_channels (used at noncollinear.py 704) and unpack_grid_channels (noncollinear.py 527). Those functions are the exact grid core of MixLayout.pack/unpack: pack_grid_channels = `cat([r_to_g(f).reshape(-1)[mask] for f in fields])`, byte-identical to MixLayout.pack's per-channel line (layout.py 65); unpack_grid_channels = `box[mask]=slice; g_to_r_box(box.reshape(shape), real=True)`, byte-identical to MixLayout.unpack (layout.py 87-89). Worse, there is a hard name collision with OPPOSITE meaning: MixLayout.unpack_grid_channels (layout.py 99-102) returns RAW G-space slices with NO FFT, while spinor_common.unpack_grid_channels returns real-space fields WITH an inverse FFT. Canonical home: have MixLayout support the noncollinear 4-channel (rho,mx,my,mz) layout (nbec=0 already works) and drop the spinor_common copies, or at minimum rename to remove the collision.

*(1 of 3 skeptics dissented)*

### [medium] `src/gradwave/scf/paw_symmetry.py:26` — Gauss-Legendre x uniform-phi sphere quadrature copy-pasted in 3 modules

The same 'Gauss-Legendre in cos(theta) x uniform phi' unit-sphere quadrature is built three separate times with the identical `dirs = np.stack([st*cos, st*sin, z])` expression and `w = wz * (2*pi/nphi)` weight: scf/paw_symmetry.py:26-37 (_sphere_quad), scf/paw_onsite.py:152-162 (OneCenter.__init__), and core/gaunt.py:70-76. Only the node-count formulas differ (paw_symmetry n=lmax+2/nphi=2*lmax+3; paw_onsite nth=(rad_lmax+2)//2; gaunt nct/nphi=2*(3*lmax_beta)+4) -- the geometric construction and weight normalization are byte-identical stanzas. A fix to the quadrature (e.g. phi endpoint handling, weight normalization) has to be applied in three places.

**Direction:** Extract one `sphere_quadrature(n_theta, n_phi) -> (dirs, weights)` (natural home: core/gaunt.py, already imported by both scf modules for ylm_np/real_gaunt_table); each caller supplies only its node counts.

### [medium] `src/gradwave/scf/uspp_loop.py:195` — Per-k davidson_gen lacks the indefinite-S guard its batched twin has; batched comment falsely claims parity

davidson_gen (per-k, uspp_loop.py:155-226) and davidson_gen_batched (uspp_batch.py:99-233) are the same generalized-Davidson reduction (Cholesky of S -> two triangular solves for L^-1 H L^-dagger -> eigh -> back-solve). The batched copy was hardened against an INDEFINITE overlap S at low ecut PAW: uspp_batch.py:166-178 uses torch.linalg.cholesky_ex, drops the OLDEST subspace entries while info>0, and falls back to s_sub + 1e-10*eye. The per-k copy at uspp_loop.py:195 is a bare `ell = torch.linalg.cholesky(s_sub)` with no cholesky_ex, no drop-oldest loop, no jitter fallback -- it raises _LinAlgError the moment vSv-dagger loses positive-definiteness. Both paths are reachable: the `batched` flag selects davidson_gen_batched at uspp_loop.py:883 vs the per-k davidson_gen at uspp_loop.py:928. The batched comment (uspp_batch.py:152-153) even asserts 'The factorization catch is the whole guard, as in the per-k path' -- but the per-k path has no catch, so the claim is stale/false. The batched docstring cites the exact regime (low-ecut Si PAW, 8-12 Ry) where S tips non-PD.

**Direction:** Collapse the generalized reduction into one shared helper (subspace build -> guarded Cholesky w/ drop-oldest + jitter fallback -> L^-1 H L^-dagger eigh -> back-solve) consumed by both davidson_gen and davidson_gen_batched, so the indefinite-S guard cannot exist in only one copy. At minimum, backport the cholesky_ex/drop-oldest guard to davidson_gen and correct the false 'as in the per-k path' comment.

### [medium] `src/gradwave/scf/uspp_hubbard.py:99` — Parallel Hubbard atomic-orbital projector builder duplicated between core/hubbard and scf/uspp_hubbard

The phase-free atomic-orbital projector-factor construction is implemented twice with the same structure: core/hubbard.py:144-167 (sbt radial FT -> ylm_all -> pref=(4pi/sqrt(vol))*MINUS_I_POW[l] -> loop mm: pref*(f*y[:, l*l+mm])) and uspp_hubbard.py:99-116 phi_free_at_sphere (qmag=sqrt(kpg2) -> ylm_all -> pref=(4pi/sqrt(vol))*_MINUS_I_POW[ll] -> loop mm: pref*(f*y[:, ll*ll+mm])). The site-descriptor loop is also duplicated: uspp_hubbard.py:42-55 hubbard_sites vs core/hubbard.py:135-142; and the atom_of_col tensor build: uspp_hubbard.py:129-131 vs core/hubbard.py:164-166. The only real deltas are the USPP/PAW radial source (msh truncation at uspp_hubbard.py:104-106, raw-unnormalized orbitals) and S-dressing done later. A Ylm-ordering or prefactor fix in one will not reach the other.

**Direction:** Extract a shared phi-factor builder taking (radial table, l, sph, vol) and returning pref*(f*Ylm) columns, parameterized by the mesh-truncation/normalization policy; both the UPF (core/hubbard) and PAW-S-dressed (uspp_hubbard) paths call it. Share hubbard_sites/atom_of_col too.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/api/elastic.py:205` — time-reversal predicate `not (inp.noncollinear and not inp.nonmagnetic)` copied 3x

The k≡−k eligibility test `not (inp.noncollinear and not inp.nonmagnetic)` appears verbatim at api/elastic.py:42 (_elastic_time_reversal), api/elastic.py:205 (run_elastic inline), and api/system.py:155 (build_system), with comments at 203-204 and 41 both noting they 'mirror build_system'. This encodes the k-point-weight/time-reversal convention (a documented recurring bug class); three hand-kept copies mean a change to when TR folding is valid (e.g. SOC nuance) can be fixed in one and missed in the others. Collapse into one helper, e.g. `_common.time_reversal_ok(inp)`.

**Direction:** Add a single `time_reversal_ok(inp)` helper in api/_common.py and call it from system.py:155 and both elastic.py sites.

### [low] `src/gradwave/api/elastic.py:186` — XC-from-Input 3-branch (noncollinear/nspin==2/else) reconstructed across api/* instead of one resolver

The dispatch `if inp.noncollinear: NoncollinearXC(SPIN_XC_REGISTRY[inp.xc]()) elif inp.nspin==2: SPIN_XC_REGISTRY[inp.xc]() else: XC_REGISTRY[inp.xc]()` is written twice inside elastic.py alone (30-37 in _elastic_rebuild, 186-192 in run_elastic) and re-derived in fragments elsewhere: api/scf.py:86/172, api/dispatch.py:47, api/system.py:160, api/summary.py:217/300-301, api/phonons.py:102/197. summary.py:300-301 already wraps a `_spin_setup`-style helper, showing a canonical home was started but not adopted. These are the sole readers of XC_REGISTRY/SPIN_XC_REGISTRY, so a single `build_xc(inp)` in _common.py would own the spin/noncollinear-selection contract.

**Direction:** Add `build_xc(inp)` to api/_common.py encapsulating the noncollinear/nspin==2/else selection and have elastic/scf/dispatch/phonons/summary call it.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/api/relax.py:631` — _relax_newton is a near-verbatim clone of _relax_joint

_relax_joint (relax.py:535-628) and _relax_newton (relax.py:631-720) share ~90% of their body; _relax_newton's own docstring says 'Same contract, guard, and nested fallback as `_relax_joint`'. Identical blocks: the `_joint_supported` guard + `_species_upfs`, the cell0/pos0/omega/smax setup, the try/except (ValueError, RuntimeError) fallback, the `if not res.converged` fallback, and the entire final ASE-consistent recompute (atoms.copy/set_cell/set_positions/_build_relax_calc, energy=get_potential_energy, forces, fmax_final, sp_kw+stress, SinglePointCalculator frame, the ~20-key `relax` dict, the `if inp.relax.cell` stress block, and the verbose print). Only the inner optimizer call (joint_relax vs newton_cg_relax) and the provenance fields (n_closures/n_cycles vs n_newton/n_grad/n_hvp) differ. A fix to the recompute contract (e.g. the fmax convention or a new reported field) applied to one will silently skip the other. Collapse into one `_finalize_forward_engine(inp, res, verbose, extra_fields)` helper the two engines call after their optimizer returns.

**Direction:** Extract the shared guard+fallback+ASE-consistent-recompute+relax-dict assembly into one helper; each engine supplies only its optimizer call and provenance dict.

### [low] `src/gradwave/api/relax.py:252` — eV/Å³→GPa conversion redefined inline and DRIFTED to the pre-2018 CODATA value

relax.py:252 hardcodes `gpa_to_ev_a3 = 1.0 / 160.21766208` to convert the user's GPa target pressure into eV/Å³ for FrechetCellFilter.scalar_pressure. The SAME physical conversion (1 eV/Å³ = pressure) is defined independently in FOUR other places, all with a DIFFERENT literal: postscf/eos.py:27 `EV_A3_TO_GPA = 160.2176634`, postscf/elastic.py:43 `EV_A3_TO_GPA = 160.2176634`, postscf/stress.py:73 `EV_A3_TO_KBAR = 1602.176634`, postscf/stress_error.py:58 `EV_A3_TO_KBAR = 1602.176634`. relax.py's 160.21766208 derives from the CODATA-2014 elementary charge (1.6021766208e-19); the other four use 160.2176634 from the CODATA-2018 charge (1.602176634e-19). This is a once-identical constant that has drifted, and it directly contradicts constants.py's header claim to be 'the single source of truth ... CODATA 2018' — yet constants.py defines NO pressure conversion at all, forcing every consumer to reinvent it. Numeric impact is tiny (~8e-9 relative) so no physics breaks today, but it is a genuine divergence in a codebase whose stated invariant is one unit-conversion home. Canonical fix: add `EV_A3_TO_GPA` (and derive `EV_A3_TO_KBAR = 10*EV_A3_TO_GPA`) to constants.py and have eos.py/elastic.py/stress.py/stress_error.py/relax.py import it; delete all five local literals.

**Direction:** Define the pressure conversion once in constants.py (CODATA-2018, 160.2176634) and import everywhere; remove the five copies including the drifted relax.py literal.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/calculator.py:55` — XC name→class registries duplicated between calculator and api/_common

calculator.py:55/58 define `_XC = {"lda": LDA_PW92, "pbe": PBE, "r2scan": R2SCAN}` and `_SPIN_XC = {"lda": LSDA_PW92, "pbe": SpinPBE, "r2scan": SpinR2SCAN}`. api/_common.py:29/33 define the structurally identical `XC_REGISTRY` and `SPIN_XC_REGISTRY` with the same keys and classes. The calculator's own comment (line 56-58) admits 'same registry the api's _spin_setup uses'. Two independent maps of the supported-functional string set: adding a new functional (or renaming one) requires editing both, and forgetting one makes the ASE-calculator path (task via GradWave) and the YAML/api path disagree on which xc names are valid — a silent divergence exactly of the kind this objective targets. Canonical home: api/_common's XC_REGISTRY/SPIN_XC_REGISTRY; calculator imports them.

**Direction:** Have calculator._make_xc index api._common.XC_REGISTRY/SPIN_XC_REGISTRY; delete the local _XC/_SPIN_XC dicts.

### [low] `src/gradwave/calculator.py:95` — _fft_grid helper copied verbatim between calculator.py and api/system.py

calculator.py:95-98 defines `def _fft_grid(system): return system.grid` with a docstring ('Both `System.grid` and `USPPSystem.grid` are `FFTGrid`; this just names the shared field...'). api/system.py:110-113 defines the byte-identical function with the identical docstring. calculator.py already imports other shared helpers from api (`_load_upf`, `_is_uspp` via lazy `from gradwave.api import ...` at lines 539/545), so this is gratuitous duplication of a trivial-but-contractual accessor. Collapse to one home (api.system._fft_grid) and import it in calculator.py.

**Direction:** Import _fft_grid from api.system in calculator.py; delete the local copy.

### [low] `src/gradwave/calculator.py:1038` — Symmetric-tensor→ASE Voigt packing repeated three times in calculator.py

The 3x3→Voigt-6 stress packing `np.array([sig[0,0], sig[1,1], sig[2,2], sig[1,2], sig[0,2], sig[0,1]])` is written out three times: _calculate_nc (calculator.py:1038-1040), _calculate_uspp (1214-1216), and the dispersion add in _apply_dispersion (593-594, `s[...]`). Same index pattern, same ASE (xx,yy,zz,yz,xz,xy) convention. A single `_to_voigt(sig)` helper removes the copy and the risk of one site permuting an index differently.

**Direction:** Add a `_to_voigt(sig3x3) -> np.ndarray` helper and use it at all three sites.

### [low] `src/gradwave/api/eos.py:26` — Scaled-system builder duplicated: _eos_build vs run_eos._build_at

_eos_build (eos.py:26-48, used by the SeedPool worker path) and the nested closure _build_at (eos.py:115-133, used by the serial/reference path) are parallel implementations of the same operation: isotropically scale the cell (`cell0 * scale**(1/3)`), map fractional positions to Cartesian, then branch on uspp to call setup_uspp(...) / setup_system(...) with the same argument set. _eos_build's docstring even says 'mirrors run_eos' `_build_at`'. The only substantive difference is parameter plumbing (worker rebuilds upfs/soa vs the closure captures them). Two copies of the scaled-cell + pseudo-branch logic can drift (e.g. one gets a new setup_* kwarg like kshift/ecutrho and the other doesn't — note _eos_build passes kshift only in the NC branch, matching _build_at, so they must be kept in lockstep by hand). Collapse to a single module-level builder both call sites use.

**Direction:** Keep one module-level _eos_build(inp, upfs, uspp, soa, scale, fft_shape) and have run_eos._build_at delegate to it (resolving upfs/soa once).

### [low] `src/gradwave/core/batch.py:297` — GPU band-chunk sizing formula duplicated between BatchedHamiltonian._band_chunk and density_b

The dense-box band-chunk budget `int(_GPU_DENSE_BUDGET_BYTES/(elem_bytes*n*max(nk,1)))` with a CPU no-limit escape appears twice: BatchedHamiltonian._band_chunk (lines 226-235, CPU sentinel 1_000_000) and density_b (lines 296-299, CPU sentinel nb). Same budget policy, two copies with two different 'no limit' sentinels; a change to _GPU_DENSE_BUDGET_BYTES semantics or the ~4-temporaries assumption must be mirrored, and the divergent CPU branches make it easy to update one and not the other.

**Direction:** Extract a single module-level band_chunk(nk, n, elem_bytes, device, n_bands) helper and call it from both the apply chunk loop and density_b.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/core/energies/esm.py:83` — reciprocal_cell (2pi*inv(cell).T) reimplemented inline in torch twice instead of shared helper

grids.reciprocal_cell(cell)=2pi*inv(cell).T is named the single source of truth for the 2pi-inside-reciprocal-vectors convention (watch-list #1), and ewald.py correctly imports it (lines 62,71). esm.py instead open-codes `b = 2.0*math.pi*torch.linalg.inv(cell).transpose(-2,-1)` in _esm_geom (line 83) AND again identically in esm_energy_strained (line 424). Two inline copies of the reciprocal-cell contract that must stay in lockstep with grids.reciprocal_cell; any change to the 2pi convention now has three edit sites.

**Direction:** Add a torch/differentiable reciprocal_cell variant beside grids.reciprocal_cell (accepting a grad-carrying strained cell) and call it from both esm sites so the 2pi convention lives in exactly one place.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/core/batch.py:268` — becp <p|psi> contraction reimplemented inline for Hubbard branch instead of calling becp_b

becp_b (lines 316-321) is the canonical <p|psi> overlap `einsum('kpg,kbg->kbp', p_conj, c)`, and the KB nonlocal branch uses it via becp_b(p,c,p_conj) (line 264). The DFT+U branch of apply() re-inlines the identical contraction `torch.einsum('kpg,kbg->kbp', hq_conj, c)` (line 268) rather than calling becp_b(hq, c, hq_conj). Same operator, parallel implementation.

**Direction:** Route the Hubbard branch through becp_b(hq, c, hq_conj) so both nonlocal terms share the single becp contraction.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/core/metagga.py:42` — _GPU_DENSE_BUDGET_BYTES and _band_chunk copied from core/batch.py (now three copies of the formula)

metagga.py:42 `_GPU_DENSE_BUDGET_BYTES = 4e8` duplicates core/batch.py:31 (the comment even says 'matches core.batch'), and metagga.py:45-48 `_band_chunk` is a near-verbatim copy of the BatchedK._band_chunk method at batch.py:226-235 (same non-cuda `1_000_000` return, same `max(1, int(_GPU_DENSE_BUDGET_BYTES/(elem_bytes*n*max(nk,1))))`). The same formula is inlined a THIRD time in batch.py:297 (density_b). metagga.py already imports from core.batch (line 37: BatchedK, g_to_r_b, box_to_sphere_b), so the copy is gratuitous. Drift hazard: retuning the ~380MB GPU budget in one place leaves the meta-GGA tau paths on the old value, changing peak memory / chunking silently.

**Direction:** Export a single `band_chunk(nk, n, elem_bytes, device)` free function (and the budget constant) from core/batch.py; have BatchedK._band_chunk, density_b, and metagga all call it.

### [low] `src/gradwave/core/xc/r2scan.py:75` — f''(0) spin-stiffness constant duplicated with DIFFERENT values (_FZ20 vs spin._F_DD0)

r2scan.py:75 `_FZ20 = 1.709920934161365617563962776245` and spin.py:31 `_F_DD0 = 1.709920934161365` are two copies of the same physical constant f''(0) (the PW92 second derivative of f(ζ) at ζ=0). They disagree: spin.py's copy is truncated ~3 ULP short of the exact value (float64 ULP at 1.7 ≈ 2.2e-16; the two differ by ~6e-16). Both feed the SAME PW92 spin-interpolation term (r2scan uses _FZ20 in _f_pw line 145/146; spin uses _F_DD0 in eps_c_pw92_spin line 49). A textbook drifted copy: one home holds the full-precision constant, the other a rounded one. Canonical home: a single named constant (e.g. constants.py or lda_pw92) F_ZETA_DD0 imported by both.

**Direction:** Define f''(0) once (full precision 1.709920934161365617...) and import it into both r2scan._f_pw and spin.eps_c_pw92_spin; delete the truncated spin._F_DD0.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/core/xc/r2scan.py:126` — PW92 correlation G-function + parameter tables triplicated across lda_pw92 / spin / r2scan, with drifted A parameters

The PW92 G(rs) closed form lives three times: lda_pw92._g_pw92 (line 32, canonical, imported by spin.py), and r2scan.py:126 `_pw92_g` reimplements the identical q0/q1/log1p form (plus an analytic derivative). The parameter sets are also split and DRIFTED: lda_pw92._EC0 A=0.031091 vs r2scan._PW_A[0]=0.0310907 (same unpolarized A, different rounding); spin._EC1 A=0.015545 (line 36) vs r2scan._PW_A[1]=0.01554535 (line 69); spin._MAC A=0.016887 (line 37) vs r2scan._PW_A[2]=0.0168869 — all three rows share identical β1..β4/α1 but the A coefficient rounds differently between the spin.py copy and the r2scan copy. Two independent PW92 parameterizations of the same fit that can silently diverge further. Canonical home: one PW92 module owning _g_pw92, its analytic derivative, and the full (A,α1,β1..β4) table for all three rows (unpolarized/polarized/stiffness).

**Direction:** Collapse _pw92_g into lda_pw92._g_pw92 (add the derivative there), move spin's _EC1/_MAC next to _EC0 as the single PW92 table, and have r2scan import them; if r2SCAN legitimately needs libxc's higher-precision A (modified-PW92), make that an explicit, documented single alternate constant, not an ad-hoc rerounding.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/core/xc/spin.py:47` — Spin-interpolation f(ζ) and its denominator (2^{4/3}-2) duplicated between spin.py and r2scan.py

The Perdew-Zunger/PW92 spin function f(ζ)=((1+ζ)^{4/3}+(1-ζ)^{4/3}-2)/(2^{4/3}-2) exists twice: spin.py:47 computes it inline with `_FZ_DEN` (defined spin.py:32), and r2scan.py:79-80 `_f_zeta` with an identically-defined `_FZ_DEN` (r2scan.py:76). Same constant literal `2.0 ** (4.0 / 3.0) - 2.0` is declared in both files. The two versions differ only in a caller-supplied zeta clamp, not the formula. Should collapse to one exported `f_zeta` helper.

**Direction:** Export a single f_zeta(z) (and _FZ_DEN) from a shared spin helper module; import into both spin.eps_c_pw92_spin and r2scan.

### [low] `src/gradwave/core/xc/r2scan.py:115` — LDA (Slater) exchange energy density reimplemented inline in r2scan instead of reusing eps_x_lda

r2scan.py:115 computes `eps_x_unif = -3.0/(4.0*math.pi)*kf` with kf=(3π²n)^{1/3} (line 114), which is algebraically identical to lda_pw92.eps_x_lda(n) = _CX·n^{1/3} with _CX=-0.75·(3/π)^{1/3} (the LDA exchange pbe.py and spin.py both import). r2scan re-derives the same uniform-exchange constant by hand rather than importing the canonical eps_x_lda, so a future correction to the Slater coefficient would miss this copy.

**Direction:** Reuse lda_pw92.eps_x_lda for the uniform exchange piece inside _ex_unpol.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/distributed.py:226` — shard_system / shard_uspp_system duplicate the shard-range + zero-share guard + DistKContext boilerplate

shard_system (line 167) and shard_uspp_system (line 226) are near-identical: both call shard_range, both raise the SAME `rank ... would get zero k-points` ValueError (lines 197-201 vs 259-263, byte-identical message), both slice spheres/kweights/proj_data via dataclasses.replace, and both build an identical DistKContext (lines 214-222 vs 288-296). Only the batched-field handling differs (NC rebuilds BatchedK; USPP re-truncates smooth_flat_idx). The shared prologue/epilogue could collapse into a common helper, keeping only the per-formalism replace() body distinct — reducing the risk of the two guards/contexts drifting.

**Direction:** Extract a _shard_common(system, rank, world_size, group) helper returning (start, end, ctx) and the raised guard; let each shard_* only compute its formalism-specific replaced fields.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/kpoints.py:28` — The (-1/2,1/2] fold-to-BZ helper is copy-pasted between kpoints and symmetry, and a second differently-half-open convention coexists

kpoints.py defines `fold(x) = -((-x+0.5)%1.0-0.5)` (lines 28-30) to map k_frac to (-1/2,1/2]. symmetry.py:175 inlines the identical expression `kfrac = -((-kfrac+0.5)%1.0-0.5)  # fold to (-1/2,1/2]`. Meanwhile symmetry.py:98 and irreps.py:304 use the OTHER form `(frac+0.5)%1.0-0.5`, which folds the zone boundary the opposite way (k=0.5 -> +0.5 for the first form, -0.5 for the second). The two conventions living side by side as ad-hoc inline expressions is exactly the drift hazard the map flags for k_frac ('folded to (-1/2,1/2]'): a future edit to one boundary rule won't propagate. Canonical home: a single `fold_bz(frac)` util (near kpoints) with the (-1/2,1/2] contract, used by both kpoints.monkhorst_pack and symmetry.reduce_mesh; equality-check call sites (irreps.py) can share it too.

**Direction:** Extract one shared fold_bz helper with the documented (-1/2,1/2] convention and replace the three inline copies, so the boundary convention is defined once.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/io/checkpoint.py:29` — Energy-term breakdown is defined in three places; energies_eV_dict (the documented single source) omits fock/e0, which are bolted on downstream

energies_eV_dict (checkpoint.py:29-40) is documented as 'the ... energy breakdown (eV) shared by the checkpoint payload and the api summary' — the intended single source. But api/summary.py:80-84 does `{**energies_eV_dict(e), 'fock': float(getattr(e,'fock',0.0)), 'e0': ...}`, i.e. `fock` and `e0` are appended outside the canonical dict, and io/output.py:_energy_lines (output.py:194-197) hard-codes yet a third independent list of the terms to render — including `('fock','Fock exchange')` which the canonical dict never emits. Net effect: (a) checkpoint payloads written via energies_eV_dict (checkpoint.py:104) silently carry no fock term, so a hybrid-SCF checkpoint loses the Fock energy component; (b) adding/renaming an energy term requires editing three disjoint lists (energies_eV_dict, summary.py's manual keys, output.py's `shown` tuple) that can drift out of sync. The docstring also says '11-term' but the dict has 12 keys.

**Direction:** Fold `fock` (and any future components) into energies_eV_dict as the single contract (default 0.0 via getattr), let summary.py add only the derived e0, and have output._energy_lines iterate the dict keys rather than a hand-maintained parallel list.

### [low] `src/gradwave/opt/newton.py:316` — newton_cg_relax duplicates the entire outer basis-rebuild scaffolding of joint_relax

newton_cg_relax (newton.py:316-393) and joint_relax (joint.py:410-519) run byte-for-byte-equivalent per-cycle setup and rebuild logic; only the inner solver differs (Newton-CG _newton_inner vs the L-BFGS closure). Identical blocks: setup_system(...use_symmetry=False) (newton 317-320 / joint 411-414); the n_occ derivation incl. the same odd-electron ValueError text (newton 322-326 / joint 416-420); occ=torch.full((nk,n_occ),2.0) (newton 327 / joint 421); tabs=[RadialTables(u)...] (newton 328 / joint 422); the orbital-seed block — cycle-0 count_h_applies+scf(max_iter=seed_scf_iters,etol=0.0,rhotol=0.0,diago_tol=1e-4) else Miller-transfer+lowdin (newton 331-340 / joint 425-434); ekin_ref = np.mean of kinetic_band(...).mean() (newton 342-344 / joint 437-439); teter_precond list (newton 345-346 / joint 440-441); leaf construction of eps_p/frac_p/z_params via view_as_real (newton 348-359 / joint 444-455); and the apply-strain/decide-rebuild tail eps_np symmetrize→cell→positions→coeffs detach→strain_step→break (newton 381-392 / joint 504-517). newton.py already imports 7 helpers from joint.py, confirming the intended one-way dependency; the scaffolding was copy-pasted rather than shared. Drift risk is real: a fix to the seed-SCF schedule or the rebuild-convergence gate must be duplicated in both, and the highest-value bug is one fixed in one driver only.

**Direction:** Extract two shared helpers into joint.py (already the canonical home newton imports from): a _setup_cycle(cell,positions,...) -> (system, occ, tabs, precond, leaves, eps_p, frac_p, z_params, npws, coeffs_init, h_seed_delta) and a _finish_cycle(eps_p, frac_p, z_params, precond, npws, a0_np) -> (cell, positions, coeffs_init, strain_step). Both drivers then only own their inner optimizer loop.

### [low] `src/gradwave/opt/newton.py:158` — Volume-collapse strain guard copy-pasted between newton.energy and joint closure with identical magic constants

The det(1+eps_s) guard is textually identical in newton.py:158-161 and joint.py:473-477: threshold `float(detj)<0.2 or >5.0` and penalty `1e3*((detj-1.0)**2 + (eps_s**2).sum() + 1.0)`. This is a numerics contract (when to reject a strain trial and what finite penalty to substitute so the line search / trust region backtracks off a NaN-inducing 1/Omega blowup). If the 0.2/5.0 bracket or the penalty scale is retuned in one engine, the two joint relaxers silently reject different steps for the same physics.

**Direction:** Hoist to a single helper in joint.py, e.g. strain_penalty(eps_s) -> Tensor|None (returns the penalty when out of the [0.2,5.0] det bracket, else None), and call it from both joint_energy's closure and newton's energy().

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/opt/newton.py:170` — Convergence-metric (fmax/smax) computation duplicated as newton.metrics and joint._grad_metrics

newton.py:170-181 metrics() and joint.py:348-362 _grad_metrics() compute the same convergence quantities from the joint gradients: fmax via de_dpos = g_frac @ inv(a_e).T then mean-subtracted per-atom, max|.|; smax via 0.5*(g_eps+g_eps.T)/omega0, max|.|; cmax via max|g_z|. Same formulas and same fix_cell branch (smax=0). They differ only in how grads are sourced (newton takes an autograd.grad list arg; joint reads leaf .grad attributes). Any change to the convergence definition (e.g. adding a rigid-translation/rotation projection to forces) must be edited in both, and the mean-subtraction / a_e^{-T} transform is exactly the kind of detail that drifts.

**Direction:** Factor a single joint_grad_metrics(g_eps_or_None, g_frac, g_z_list, a_e_np, omega0) -> (fmax, smax, cmax) in joint.py and have both call sites pass their grads into it.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/_response.py:190` — K_Hxc screening operator has 5+ drifting parallel implementations; a fork's divergence rationale is now stale

_response.py:_k_hxc_apply (line 190) is one of at least five near-identical implementations of the Hxc response kernel K_Hxc w = hartree_kernel(total drho) + fxc HVP at rho(+NLCC core), all built from the same _response.py primitives (hartree_kernel/fxc_hvp/fxc_hvp_spin) but re-orchestrated: (1) scf/implicit.py:375 apply_k_hxc, (2) this _k_hxc_apply, (3) scf/soft_mode.py:72 _k_hxc_fxc_scaled ('Mirrors scf/implicit.py::apply_k_hxc exactly'), (4) postscf/dielectric.py:412 _k_hxc + :429 _k_hxc_spin, plus referenced hubbard_u._k_hxc_spin and the USPP k_hxc_grid twin. They agree on the physics-critical NLCC-core split (nspin=1: rho+rho_core; nspin=2: 0.5*rho_core per channel), but the copies have DRIFTED in their documentation of each other: dielectric.py:419 justifies its private fork with 'The shared scf.implicit.apply_k_hxc omits the core; this local copy folds it in' — yet scf/implicit.py:400 demonstrably folds rho+core in (its own docstring at :381-388 says it evaluates f_xc at rho+rho_core). The stated reason for forking no longer holds; the copies are now behaviorally identical, so the fork is redundant and its rationale is a stale divergence marker — exactly the pattern where a future fix to one copy silently won't reach the others.

**Direction:** Collapse to a single core-aware K_Hxc apply whose density source is a parameter: keep the raw-density _k_hxc_apply(grid, xc, w_s, rho_s, rho_core, nspin) as the canonical kernel in _response.py, and have scf/implicit.apply_k_hxc, soft_mode._k_hxc_fxc_scaled (with an fxc_scale kwarg), dielectric._k_hxc/_k_hxc_spin, and the hubbard_u/USPP twins all delegate to it (extracting rho_s/rho_core from their SCFResult once). Delete dielectric.py's now-redundant _k_hxc/_k_hxc_spin and fix/remove the stale 'omits the core' comment.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/cohp.py:313` — Minimum-image fractional-distance computation reimplemented ad hoc in cohp.py and bader.py

The minimum-image convention (frac = d @ inv(cell); frac -= np.round(frac); dist = norm(frac @ cell)) is open-coded in cohp.py:_min_image_dist (307-314) and cohp.py:_nearest_image_R (317-324), and again in bader.py:216-226 for attractor->nucleus binding (dfrac -= np.round(dfrac); dcart = dfrac @ cell; d2 = einsum). core/energies/ewald.py also carries its own integer-image machinery. These are structural clones of one geometric primitive with no shared home; each re-fetches positions.detach().cpu().numpy() and inverts the cell independently.

**Direction:** Add a single min_image_delta(cell, frac_or_cart) / min_image_distance helper (in core/structure.py or grids.py) returning the wrapped Cartesian delta, and route cohp._min_image_dist, cohp._nearest_image_R, and bader's nucleus binding through it. Low priority but removes three copies of a convention that must stay consistent with the cell handling used elsewhere.

### [low] `src/gradwave/postscf/eos.py:27` — eV·Å³→GPa/kbar factor copied across ~5 sites, one drifted to the CODATA-2014 value

The map states constants.py is the single source of truth and everything is CODATA 2018. But eV/Å³→GPa is redefined locally: eos.py:27 `EV_A3_TO_GPA=160.2176634`, elastic.py:43 an identical second copy (not imported from eos), stress.py:73 & stress_error.py:58 `EV_A3_TO_KBAR=1602.176634`, and api/elastic.py:387 hardcodes `160.2176634`. api/relax.py:252 uses `gpa_to_ev_a3 = 1.0/160.21766208` — a DRIFTED value derived from the CODATA-2014 elementary charge 1.6021766208e-19 rather than the CODATA-2018 1.602176634e-19 used everywhere else (correct GPa factor = 160.2176634). constants.py (checked) has HARTREE_EV/E2/etc but no GPa/kbar factor, so every consumer rolls its own.

**Direction:** Add EV_A3_TO_GPA (and EV_A3_TO_KBAR = 10×) to constants.py derived from the same CODATA-2018 elementary charge, import it in eos.py/elastic.py/stress*.py/api, and delete api/relax.py:252's 160.21766208 literal so the drifted copy cannot re-enter.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/exchange_multik.py:96` — coulomb_potential_q is a superset of exchange.coulomb_potential (its own docstring calls exchange's the q=0 special case), yet both are hand-coded

exchange.coulomb_potential (exchange.py:53) computes v_g = 4π·E2·σ_g·_inv_g2_masked(g2) (bare kernel, G=0→0). exchange_multik.coulomb_potential_q (96) computes v_g = σ_g·coulomb_kernel(qg2, mode, omega); coulomb_kernel.py confirms mode='full' returns exactly 4π·E2/|q+G|² with the G=0 cell zeroed. coulomb_potential_q's docstring even states it is 'exchange.coulomb_potential (its q=0, full-kernel special case)'. The FFT normalization (r_to_g/g_to_r_box) is identical. Same for physical_orbitals (exchange.py:42) vs physical_periodic_orbitals (exchange_multik.py:167) both re-deriving the /√Ω normalization.

**Direction:** Make exchange.coulomb_potential call coulomb_potential_q with q=0/mode='full', or have both share a single (kernel, sigma)→v_r primitive. One Coulomb-potential contract keyed on the range-separation kernel.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/exchange_multik.py:230` — _inv_softplus (and the sigmoid reparam pattern) copy-pasted from core/xc/learnable.py

exchange_multik.py defines _inv_softplus (230) and _inv_sigmoid (225) for HybridExchangeParams; core/xc/learnable.py:76 defines an identical _inv_softplus and uses the same softplus/sigmoid reparameterization for its learnable-XC parameters. exchange_multik's HybridExchangeParams docstring says 'Mirrors core/xc/learnable.py' — the mirroring is literal copy of the inverse-reparam helpers.

**Direction:** Move _inv_sigmoid/_inv_softplus into a shared reparameterization util (e.g. core/xc/_reparam.py or a small nn helper) and import from both learnable.py and exchange_multik.py.

### [low] `src/gradwave/postscf/irreps.py:91` — _cartesian_rotation duplicates the fractional→Cartesian rotation a_t·W·a_t⁻¹ open-coded across symmetry.py and elastic.py

_cartesian_rotation (irreps.py:91-93) computes a_t @ w_mat @ np.linalg.inv(a_t) — the identical similarity transform mapping a fractional rotation W to Cartesian is repeated inline four times in symmetry.py (lines 261, 429, 510, 660: s = a_t @ w_mat @ a_t_inv) and again in postscf/elastic.py:127 _cart_rotations. Six copies of one operator with no single owner; a sign/transpose fix in one would silently not propagate.

**Direction:** Promote a single cartesian_rotation(W, cell) helper in symmetry.py (contract: rows-of-cell convention a_t=cell.T) and have irreps._cartesian_rotation, elastic._cart_rotations, and the four symmetry.py sites call it.

### [low] `src/gradwave/postscf/magnetism.py:69` — _atomic_moment_vectors duplicates moment_config._atomic_moments (same einsum, same physical quantity)

magnetism._atomic_moment_vectors (lines 69-71): `cf = system.grid.volume / system.grid.n_points; return torch.einsum("axyz,ixyz->ai", weights, m) * cf`. moment_config._atomic_moments (lines 83-88): `return torch.einsum("axyz,ixyz->ai", weights, m) * cell_factor`. Byte-identical einsum computing M_I = ∫ w_I m dr, and magnetism.py already imports atomic_weights from moment_config (line 28), so the canonical helper is right next door. The volume/n_points cell factor is also recomputed inline in both files (magnetism L70; moment_config L114, L156). Beyond this helper, magnetism.characterize_magnetism (L116-121: seed noncollinear SCF along +z → M → norm) reimplements the exact body of moment_config.reference_moment_magnitudes (L108-116: seed noncollinear SCF along dirs → M → norm); only the seed direction differs.

**Direction:** Import and reuse moment_config._atomic_moments (promote it to a non-underscore shared helper) from magnetism.py; fold the volume/n_points factor into it. Have characterize_magnetism reuse reference_moment_magnitudes-style extraction so the seeded-SCF→moment-magnitude path lives in one place.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/pdos.py:420` — Spilling formula 1 − captured/Σkweight triplicated across the three PDOS entry points

Identical spilling computation repeated in projected_dos (L420-421: captured=(weights.sum(axis=2)*kweight_state).sum(); spilling=1-captured/kweight_state.sum()), projected_dos_noncollinear (L497-498), and projected_dos_soc (L658-659). The group-broadening tail (labels=sorted({_group_key...}); build boolean masks; broaden per group; total) is likewise near-duplicated three times (L430-438, L505-513, L666-669) differing only in _group_key vs _group_key_so and the channel set.

**Direction:** Add a _spilling(weights_sum_over_proj, kweight_state) helper and a _grouped_broaden(all_e, per_state, kweight_state, cols, group_key_fn, width, grid) helper; the three PDOS functions reduce to computing per-state weights and calling them.

### [low] `src/gradwave/postscf/thermo.py:24` — cm⁻¹↔eV conversion defined twice as independent reciprocals with mismatched precision

thermo.py:24 `CM1_TO_EV = 1.239841984e-4` is an independent hardcode of the reciprocal of phonons.py:43 `_EV_TO_CM1 = 8065.54393734921` (1/8065.54393734921 = 1.2398419843320e-4, so thermo's value is the same constant truncated to 10 sig figs vs phonons' 15). phonons_supercell.py and hessian.py correctly reuse phonons.py's constant via import, but thermo.py — which consumes the very cm⁻¹ frequencies those phonon modules produce — carries its own rounded copy instead. Two hand-maintained reciprocals of the same e/(hc) factor are a divergence hazard: updating one leaves the round-trip freq→eV→freq no longer exact.

**Direction:** Put a single cm⁻¹ conversion (or its eV reciprocal) in constants.py and have both phonons.py and thermo.py derive from it, so the eV→cm⁻¹ and cm⁻¹→eV factors are guaranteed reciprocal to full precision.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/volumetric.py:254` — Thomas-Fermi kinetic prefactor recomputed inline across modules (two spellings of one value)

The uniform-gas kinetic prefactor 0.3*(3π²)^(2/3) is defined ad hoc in volumetric.py:254 (_C_F), r2scan.py:118 and :192 (inline tau_unif), and uspp_loop.py:409 (cf). Its spin-polarized sibling 0.3*(6π²)^(2/3) appears in r2scan.py:36 (K_FACTOR_C) and uspp_loop.py:412, and volumetric.py:324 computes the SAME number a third way as _C_F * 2**(2/3) (since (3π²)^(2/3)·2^(2/3) = (6π²)^(2/3)). Per the repo map, constants.py is the single source of truth for physical constants; these scattered inline literals are exactly the class of drift-prone repeated constant it exists to prevent.

**Direction:** Define TF_KINETIC_C = 0.3*(3π²)^(2/3) (and the 2^(2/3) spin scaling) once in constants.py or core/metagga.py; import everywhere. Contract: c_F for D_h = c_F ρ^{5/3}, with the per-spin gas carrying the 2^(2/3) factor.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/postscf/uspp_position.py:435` — Sternheimer warm-start init boilerplate repeated ~7x

The nested comprehension [[torch.zeros_like(c[:n_sv]) for c, n_sv in zip(cs.c_win[isp], cs.n_solve[isp], strict=True)] for isp in range(nsp)] is copy-pasted at uspp_implicit.py:766, uspp_softmode.py:199, and uspp_position.py:435/479/536/584/661 (the :584 variant drops the outer spin list). It always depends only on cs (c_win, n_solve, nspin), so every call site re-encodes the same shape contract for the occupied-band response buffers.

**Direction:** Add a _ConvergedUSPP.zero_warm_starts(per_spin=True) helper returning the correctly-shaped zero buffers; replace the seven inline copies.

### [low] `src/gradwave/postscf/uspp_implicit.py:497` — Smooth-density-from-orbital-response reduction repeated across three response paths

The identical 'ψ*δψ + δψ*ψ' smooth-density accumulator drho_sm[isp] += 2.0*wk*einsum('b,bxyz->xyz', fw, (psi_r.conj()*dpsi_r).real) — with the same load-bearing comment that the 2.0 is the conjugate pair and NOT the spin degeneracy (which lives in f_win) — appears in apply_chi0 (uspp_implicit.py:497-498), bare_map_derivative (uspp_position.py:367-368), and _total_orbital_response (uspp_position.py:600-601). The subtle 2.0-vs-f_win-g_spin split is exactly the kind of factor that could be fixed in one path and missed in another.

**Direction:** Extract a shared helper smooth_drho_from_dpsi(fw, psi_r, dpsi_r, wk) so the conjugate-pair/degeneracy convention lives in one place.

### [low] `src/gradwave/pseudo/radial_torch.py:239` — Local-potential erf-split assembly (short-range SBT + analytic Gaussian tail) duplicated between numpy local.vloc_of_g and torch RadialTables.vloc_of_g

RadialTables.vloc_of_g (radial_torch.py:239-245) re-derives the same split as local.py::vloc_of_g (local.py:51-64): `short = 4π * sbt(0, vsr*r², ...)` and `tail = -4π·z·E2·exp(-0.25*(g*rc)²)/g²`. The 4π prefactor, the -0.25 Gaussian width, and the /g² tail are written independently in each. RadialTables already reuses local.py's _v_short_range/alpha_z/_msh/RC_DEFAULT, so only the final `short + tail` assembly is cloned — but that clone carries the exact G→0 tail constant that, if changed in one, silently diverges the differentiable stress path from the setup path.

**Direction:** Factor the split-assembly (given a short-range SBT value and (z,rc,g)) into one shared helper parameterized over the transform backend, or at minimum share the tail-factor expression, so the numpy vloc form factor and the strain-differentiable form factor cannot drift in their G=0 tail.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/alchemical.py:121` — Linear alchemical blend formula re-inlined inside _alchemical_ionic_terms instead of reusing alchemical_charges / blend_local_table

alchemical.py defines named blend helpers `alchemical_charges` (`(1-lam)*z_a + lam*z_b`, l 45-50) and `blend_local_table` (`(1-lam)*tab_a + lam*tab_b`, l 64-70), and `per_atom_local_tables` (l 303) correctly calls blend_local_table. But `_alchemical_ionic_terms` re-inlines both formulas verbatim in its scalar branch (`(1.0 - lam) * z_a + lam * z_b` and `(1.0 - lam) * tab_a + lam * tab_b`, l 121-122) and again in the per-atom branch (l 124-126). Three copies of the same charge/table blend; if the blend ever became nonlinear or gained clamping, the inlined copies would silently drift from the named helpers.

**Direction:** Have _alchemical_ionic_terms call alchemical_charges and blend_local_table (broadcasting lam as needed) so the composition-blend contract lives in one place.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/common.py:326` — Per-spin occupations+entropy accumulation loop duplicated between shared_fermi_occupations and constant_mu_occupations

The loop `for isp in range(nspin): o, s_ent = occupations_and_entropy(eigs_s[isp], mu_t, scheme, width, degeneracy=g_spin); occ_s.append(o); ent = ent - width * (g_spin * kweights[:,None] * s_ent).sum()` appears verbatim in shared_fermi_occupations (common.py:305-309) and constant_mu_occupations (common.py:326-331). The entropy-term sign and prefactor (`- width * (g_spin * kweights * s_ent).sum()`) are the load-bearing piece and are maintained in two places, so a fix to the entropy convention in one grand-canonical/shared-mu path would not reach the other.

**Direction:** Extract a `_occ_and_entropy_per_spin(eigs_s, kweights, scheme, width, mu_t, g_spin, device)` helper returning (occ_s, ent[, n_e]) and call it from both fixed-mu and solved-mu paths.

### [low] `/home/wladerer/github/gradwave/src/gradwave/scf/noncollinear.py:936` — band_structure_nc re-derives the converged v_r/b_xc and v_tau operator fields already built by _nc_effective_potential/_nc_metagga_step

band_structure_nc (noncollinear.py 936-957) rebuilds the effective potential with the exact sequence of _nc_effective_potential (427-444): rho_g_box=r_to_g(rho); v_h=g_to_r_box(hartree_potential_g(rho_g_box, grid.g2), real=True); v_xc,b_xc,_=vxc_and_bxc(...); if nonmagnetic: b_xc=zeros; v_r=v_h+v_xc+vloc_r. It then rebuilds the meta-GGA v_tau fields with the exact sequence of _nc_metagga_step (491-497): vtu,vtd=vtau_up_dn(...); if nonmagnetic: v0=0.5*(vtu+vtd), vvec=zeros(3,...); else v0,vvec=tau_operator_fields(vtu,vtd,...). Two copies of the same potential-assembly contract that can drift (e.g. a sign or a rho_core split fixed in one place only). Canonical home: a shared `build_nc_potentials(xc, rho, m, grid, system, vloc_r, nonmagnetic, tau_up, tau_dn) -> (v_r, b_xc, v0, vvec)` called by the SCF loop and by band_structure_nc.

*(1 of 3 skeptics dissented)*

### [low] `/home/wladerer/github/gradwave/src/gradwave/scf/loop.py:940` — fp32-draft eigenvector renormalization copied verbatim in _solve_bands and _solve_spinor_bands

The 'fp32 draft -> renormalize in fp64 so the G=0 electron count is conserved through mixing' fix is duplicated verbatim: _solve_bands (loop.py 940-944) `c = c / torch.linalg.norm(c, dim=-1, keepdim=True).clamp_min(1e-30)` and _solve_spinor_bands (noncollinear.py 346-350) `coeffs = coeffs / torch.linalg.norm(coeffs, dim=-1, keepdim=True).clamp_min(1e-30)`, each guarded by the identical `use_low = mixed_precision and tol_eff > mp_crossover` predicate and the same comment. A change to the renorm (e.g. masking padded slots, or a different clamp) would need to land twice. Minor, but it is the same numerical contract in two drivers; a shared `renorm_fp64(c)` helper would collapse it.

*(1 of 3 skeptics dissented)*

### [low] `/home/wladerer/github/gradwave/src/gradwave/scf/guess.py:53` — SAD guess re-derives the structure factor S_a(G)=exp(-iG.tau) inline instead of core/structure.py

guess.py 52-54 computes the structure factor by hand: `gvec = grid.g_cart.reshape(-1,3); phase = gvec @ pos[atoms].T; sfac_a = exp(complex(0, -phase))`. That is exactly S_a(G)=exp(-iG.tau), which the repo map designates core/structure.py as owning (differentiable structure factors). The sign and the no-double-2pi convention happen to be correct here (g_cart already carries 2pi), but it is a second, detached copy of a phase-convention-sensitive primitive that watch-list item #1 (double/missing 2pi on structure-factor phases) flags as a recurring bug class. Collapsing to a `structure_factor(g_cart, positions)` call (used detached here) removes one place the convention can silently diverge.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/spin_precond.py:186` — build_stoner_precond and build_stoner_precond_nc are ~80% structural clones

build_stoner_precond (69-157) and build_stoner_precond_nc (186-276) share almost their entire body: f'-via-autograd through scheme.occupation, picks list with k_start offset + fp_cut threshold, gather_list_cat under dist_ctx, the identical canonical sort `picks.sort(key=lambda t:(-abs(c),*idx))` + truncate to max_bands, per-pick codensity build (g_to_r -> |psi|^2/vol -> r_to_g[mask] into u_full/w_full), the `if not (0<=ik<nk_local): continue` owner guard, all_reduce_ of u_full/w_full, and `return StonerSpinPrecond(u_full,w_full,cvals,float(vol))`. Genuine differences are tiny: spin loop `for isp in (0,1)` vs single spinor list, pick-tuple arity ((isp,ik,n,c) vs (ik,n,c)), codensity (single |psi|^2 vs |psi_up|^2+|psi_dn|^2), and kernel source (m_r vs |m_vec|). The nc twin even abbreviates the shared distributed-contract comment to 'see build_stoner_precond'. Latent divergence already present: nk_local is `len(system.spheres)` at line 137 but `eigs.shape[0]` at line 253 -- two different expressions for the same local-shard k-count, so a distributed-layout change fixed in one path silently skips the other.

**Direction:** Collapse into one builder parameterized by a (a) picks-generator and (b) codensity-provider callback; the distributed gather/sort/truncate/owner-guard/all_reduce assembly and StonerSpinPrecond construction live exactly once. Standardize nk_local on a single expression.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/paw_noncollinear.py:41` — onsite_nc_exc reimplements OneCenter._exc_t's LDA radial-angular XC quadrature

onsite_nc_exc (paw_noncollinear.py:41-62) rebuilds the same one-center LDA XC quadrature that OneCenter._exc_t (paw_onsite.py:447-497) already owns: radial values `(rl @ ylm.T)*rm2 + core`, form spin channels, `xc.eval_energy_density(up,dn)`, then `(e.reshape(mesh,nx) * wq * ww).sum()`. The non-collinear case differs only in how up/dn are formed (from n +/- |m| rather than two independent channels) and the GGA guard, but the r^2-fold, core folding, and wq*ww weighting are duplicated and can drift from _exc_t (e.g. if the core-fraction or weight convention changes).

**Direction:** Route onsite_nc_exc through a shared _exc_t-style core that accepts an (n_up, n_dn) density pair, so the r^2/core/wq*ww quadrature weighting is defined once; the nc wrapper only computes up/dn = (n +/- |m|)/2.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/paw_onsite.py:601` — 'autograd grads of scalar over becsum leaves' idiom repeated ~6x

The exact block `leaves=[x.detach().clone().requires_grad_(True) for x in ...]; with torch.enable_grad(): e=f(leaves); grads=torch.autograd.grad(e,leaves); return float(e.detach()), [g.detach() ...]` recurs in paw_onsite.py energy_and_ddd (601-607), energy_and_ddd_batch (623-628), _xc_exact (439-445), hvp_factory (558-561), and paw_noncollinear.py onsite_nc_energy_and_field (71-75) and onsite_nc_energy_and_ddd (121-125). Six copies of the same detach/clone/enable_grad/grad discipline.

**Direction:** Provide one helper `energy_and_grads(fn, leaves)` (and a create_graph variant for hvp_factory) so the detach/enable_grad/grad contract is stated once.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/uspp_hubbard.py:184` — hubbard_e_channel re-inlines the Dudarev E_U trace and the occupation-matrix contraction that core/hubbard owns

core/hubbard.py:183-189 hubbard_energy is the canonical Dudarev channel energy: 0.5*uj*(trace(n) - trace(n@n)); core/hubbard.py:170-180 occupation_matrices is the canonical n^I_{mm'} = einsum('kb,kbp,kbq->pq', w, becp, becp.conj()) build. hubbard_e_channel (the differentiable-through-positions force path) re-derives BOTH inline: uspp_hubbard.py:183 n_full += einsum('b,bm,bn->mn', w, sproj, sproj.conj()) duplicates occupation_matrices, and uspp_hubbard.py:189-191 e += 0.5*uj*(diagonal(blk).sum() - diagonal(blk@blk).sum()).real duplicates hubbard_energy. The module header (uspp_hubbard.py:13-14) even claims 'hubbard_energy ... are reused verbatim' -- true for the SCF path but NOT for this force path. Divergence risk: any change to the Dudarev convention (sign, factor, +J term) in core/hubbard.hubbard_energy silently fails to propagate to hubbard_e_channel, yielding forces inconsistent with the energy. A subtle textual drift already exists: hubbard_e_channel Hermitizes blk (uspp_hubbard.py:188) before trace(blk@blk) whereas hubbard_energy does not.

**Direction:** Have hubbard_e_channel build its per-site n blocks and then call core/hubbard.hubbard_energy on them (the S-dressed sproj build is the only genuinely force-specific part). One canonical Dudarev-energy contract.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/uspp_setup.py:41` — (-i)^l table hand-rolled in uspp_setup instead of importing constants.MINUS_I_POW

constants.py:28 is the single source of truth: MINUS_I_POW = (1.0+0j, -1.0j, -1.0+0j, 1.0j, 1.0+0j), tabulated to l=4. uspp_setup.py:41 redefines the same table locally: _MINUS_I_POW_L = [1.0, -1.0j, -1.0, 1.0j, 1.0] (used at uspp_setup.py:169) and re-exports it publicly through the scf.uspp facade (uspp.py:16). By contrast the sibling module uspp_hubbard.py:112 correctly consumes the constants-sourced value (via core/hubbard, which does `from gradwave.constants import MINUS_I_POW` at hubbard.py:32). This is a constants-ownership leak (watch-list #4): the local copy can silently drift from constants.py if the phase convention or table length is ever corrected.

**Direction:** Replace the literal in uspp_setup.py:41 with `from gradwave.constants import MINUS_I_POW as _MINUS_I_POW_L` (or drop the facade re-export). No numeric change today; removes a drift surface.

*(1 of 3 skeptics dissented)*

### [low] `src/gradwave/scf/uspp_loop.py:888` — S-metric norm and S-apply (1 + sum q|beta><beta|) re-implemented across per-k, batched, and spinor paths

The overlap S = 1 + sum_ij q_ij |beta_i><beta_j| and the S-norm ||psi||^2_S = sum|c|^2 + <beta|psi>^dagger q <beta|psi> are open-coded in at least three places each. S-apply: _HkS.s (uspp_loop.py:150-152, per-k matmul `c + (b @ q) @ p`), BatchedHS.s (uspp_batch.py:69-73, einsum 'kbp,pq,kqg->kbg' with mask), SpinorBatchedHS.s (uspp_noncollinear.py:164-174, per-spin-block). S-norm: uspp_loop.py:888-891 (batched), uspp_loop.py:930-932 (per-k), uspp_noncollinear.py:423-425 (spinor bu/bd). All encode the identical contract; they differ only in tensor layout (matmul vs batched einsum vs doubled-axis). A change to the S convention (e.g. q dtype/conjugation) must be edited in every copy.

**Direction:** Provide a single apply_S(coeffs, becp, q, p) and s_norm(coeffs, becp, q) in a shared module (e.g. next to becp_b in core/batch), and route _HkS.s, BatchedHS.s, the spinor S, and all three S-norm sites through them.

### [low] `src/gradwave/symmetry.py:660` — Cartesian rotation S = Aᵀ W A⁻ᵀ open-coded in ~7 sites; symmetrize_forces recomputes inv(Aᵀ) inside its loop

The Cartesian rotation build `a_t = cell.T; a_t_inv = inv(a_t); s = a_t @ w @ a_t_inv` is repeated in magnetic_spacegroup (261), MagneticSymmetrizer.__init__ (429), CollinearMagneticSymmetrizer.__init__ (510), VectorFieldSymmetrizer.__init__ (590-591), symmetrize_tensor (624), symmetrize_atom_tensor (643), and symmetrize_forces (660). All must agree on the S = Aᵀ W A⁻ᵀ convention documented once at the module top. symmetrize_forces (line 660) is the odd one out: it recomputes `np.linalg.inv(a_t)` once per op inside the loop instead of hoisting `a_t_inv`, unlike every other site -- a divergence that is currently only a perf wart but is exactly where a future edit to the convention could be missed.

**Direction:** Add one `cartesian_rotations(sg, cell) -> (nops,3,3)` helper (and an axial variant multiplying by det) as the single source of the S=AᵀWA⁻ᵀ convention, and have all seven callers consume it.

### [low] `src/gradwave/symmetry.py:448` — MagneticSymmetrizer.apply_m and VectorFieldSymmetrizer.apply are the same G-space vector fold, differing only in axial vs polar matrix

MagneticSymmetrizer.apply_m (452-458) and VectorFieldSymmetrizer.apply (602-609) are line-for-line identical except for the per-op 3x3 matrix set: both do `flat = field.reshape(3,-1) * rs.mask; gathered = flat[:, rs.idx]; mixed = torch.einsum('oab,bon->aon', M.to(flat.dtype), gathered); acc = (rs.phase * mixed).mean(dim=1) * rs.mask; return acc.reshape(3, *self.shape)` -- the only difference is `self.axial` (det(S)·S, pseudovector) vs `self.rot` (S, proper vector). A fix to the mask/phase handling in one would silently not reach the other.

**Direction:** Extract `_fold_vector_field(rho_sym, field3, rot3x3)` as the shared kernel and have both symmetrizers pass their axial/polar matrices into it; the axial-vs-polar distinction stays in the constructors where it belongs.

### [low] `src/gradwave/solvers/lobpcg.py:120` — Unit-normalize-before-ortho idiom and t_band einsum duplicated between davidson_batched and lobpcg_batched

The 'normalize residual directions before orthogonalization' guard is copy-pasted: lobpcg.py:120-121 `wn = torch.linalg.norm(w, dim=-1, keepdim=True).real; w = torch.where(wn > 1e-300, w / wn.clamp_min(1e-300), w)` is identical to davidson.py:503-504 (`dn`/`d`). The Teter band-kinetic contraction is also duplicated verbatim: lobpcg.py:118 `t_band = torch.einsum('kbg,kg,kbg->kb', x.conj(), tc, x).real` vs davidson.py:493 `torch.einsum('kbg,kg,kbg->kb', x.conj(), t.to(x.dtype), x).real`. Both idioms encode the same contract (protect residual direction below the ortho null-threshold; compute <psi|T|psi> per band) and could drift independently.

**Direction:** Move both into precond.py next to teter_b: a `normalize_rows(d)` helper and a `t_band(x, t)` helper, consumed by davidson_batched, lobpcg_batched (and CheFSI where applicable).

## Cross-file (global) issues

### [low] `src/gradwave/api/relax.py:252` — eV/Å³↔GPa/kbar pressure factor defined independently in 6+ sites with no constants.py home; api/relax.py has drifted to the pre-2018 CODATA value

CONFIRMED. constants.py (the declared single source of truth for unit conversions) has NO pressure constant. The factor is redefined independently in: postscf/eos.py:27 (EV_A3_TO_GPA=160.2176634), postscf/elastic.py:43 (EV_A3_TO_GPA=160.2176634), postscf/stress.py:73 (EV_A3_TO_KBAR=1602.176634), postscf/stress_error.py:58 (EV_A3_TO_KBAR=1602.176634), io/output.py:479 (inline 1602.176634), api/elastic.py:387 (inline 160.2176634). api/relax.py:252 uses `gpa_to_ev_a3 = 1.0/160.21766208` — the DRIFTED value: 160.21766208 derives from e=1.6021766208e-19 (CODATA-2014) whereas the canonical 160.2176634 uses the exact SI 2018 e=1.602176634e-19 mandated by the repo's 'CODATA 2018' convention. So the user's target-pressure GPa→eV/Å³ conversion in relax uses a factor inconsistent with the stress/eos/elastic reporting paths.

### [low] `src/gradwave/postscf/dielectric.py:419` — dielectric._k_hxc forks scf.implicit.apply_k_hxc on a FALSE rationale — its comment claims the shared kernel omits the NLCC core, but it includes it

postscf/dielectric.py:412-426 keeps a private `_k_hxc` whose docstring (line 419) justifies the duplicate: "The shared `scf.implicit.apply_k_hxc` OMITS the core; this local copy folds it in." That premise is false. scf/implicit.py:375-407 apply_k_hxc DOES fold in the NLCC core: line 400 `rho_xc = res.rho if core is None else res.rho + core`, and for nspin=2 line 403 `c2 = 0.5*core`; its own docstring (implicit.py:381-388) says f_xc is evaluated at ρ+ρ_core "INCLUDING the NLCC partial core." For nspin=1 the two functions are byte-equivalent (`hartree_kernel + fxc_hvp at rho+core`). This is exactly a cross-slice blind spot: the slice-9 local reviewer, reading only dielectric.py, reported 'K_Hxc means different things: dielectric folds NLCC into fxc, apply_k_hxc does not' — trusting dielectric's stale comment about the sibling module. The whole K_Hxc family (postscf/_response.py:190 `_k_hxc_apply`, uspp_softmode _k_hxc_grid_scaled, dielectric._k_hxc) is maintained as parallel forks partly on divergence rationales that no longer hold.

## Remediation themes (judge-panel synthesis)

### Extract small shared low-level math/util helpers

A tail of small verbatim/near-verbatim helpers with no shared home: reciprocal_cell (2pi*inv(cell).T) reimplemented inline in torch twice (esm.py:83); _fft_grid copied between calculator.py and api/system.py (calculator.py:95); Gauss-Legendre x uniform-phi sphere quadrature copy-pasted in 3 modules (paw_symmetry.py:26); symmetric-tensor->ASE Voigt packing repeated three times (calculator.py:1038); _inv_softplus + sigmoid reparam copy-pasted from core/xc/learnable (exchange_multik.py:230); and coulomb_potential_q hand-coded as a superset of exchange.coulomb_potential (exchange_multik.py:96). Low severity each, but a shared util module clears the whole long tail. Grouped last as cleanup breadth.

Proposal scores: 36/40 — Strong accept: a rigorously verified minimal-diff plan that routes each dup to an existing home with identity-preserving numerics; only the numbered item-4 sphere_quad arithmetic (lmax+2 vs 2*lmax+2) is misstated, though the sketch/risks correct it.; 32.5/40 — Well-researched, physically sound consolidation whose invariant-carrying typed contracts are strong; adopt items 1/2/4 nearly as-is, but scope down item 3 (coulomb) to share the kernel rather than force a g2→g_cart signature change on the hot Fock path, and fold in the third _inv_softplus copy in learned_precond.; 32/40 — Accept — claims verified against the tree; physically sound consolidation to single sources of truth, docked for the broken voigt_from_tensor placeholder, a couple of missed inline sites, and an understated (low-to-moderate, ~10-file) footprint.

**Recommendation:**

## Final recommended remediation — "Extract small shared low-level math/util helpers"

**Design stance:** Keep the winner's minimal-diff spine (route each duplicated helper to the nearest *existing* low-level home, delete copies, no new util package). Make exactly **one** amendment: for the reciprocal-cell helper, adopt the runner-up's **backend-generic single definition** instead of the winner's numpy-plus-`_t`-sibling pair. This is the one place where "one canonical function" beats "one more sibling," because the failure mode being remediated (the differentiable-stress path drifting from the energy path) is *caused by having two implementations of the same geometry* — a torch sibling still leaves two bodies that can drift; a single dispatching function leaves zero.

---

### 1. `reciprocal_cell` — backend-generic, one definition (GRAFTED from runner-up)

**Home:** `grids.py` (replaces the existing numpy `reciprocal_cell` at grids.py:27 in place — same name, same numpy contract, extended to tensors).

**Contract / signature:**
```python
def reciprocal_cell(cell):
    """Reciprocal cell: rows b_i [Å⁻¹] with a_i·b_j = 2π δ_ij, i.e. B = 2π (A⁻¹)ᵀ.
    Backend-generic: torch.Tensor in → torch.Tensor out (grad/dtype/device preserving);
    array-like in → np.ndarray out. Batched: acts on the trailing (...,3,3)."""
    if torch.is_tensor(cell):
        return 2.0 * math.pi * torch.linalg.inv(cell).transpose(-2, -1)
    cell = np.asarray(cell)
    return 2.0 * math.pi * np.linalg.inv(cell).swapaxes(-2, -1)
```

Notes that make this behavior-preserving:
- The numpy branch is the pre-existing body; `.swapaxes(-2,-1)` on a 2-D array is identical to `.T`, so all current numpy callers are bit-for-bit unchanged.
- The torch branch is *exactly* the expression currently inlined at every stress/geometry site, so numerics are bit-for-bit unchanged there too. `transpose(-2,-1)` on a 2-D tensor `== .T`; batched cells are handled for free.
- `grids.py` already imports numpy; add `import math, torch` at module top (torch is a hard dep of the package, so no optionality needed).

**Migration path — replace the 6 inline torch reimplementations with `reciprocal_cell(cell)`:**
- `esm.py:83`, `esm.py:424`
- `dispersion_d4.py:360`
- `paw_stress.py:173`
- `_strain.py:79`
- `opt/joint.py:239`

Each site currently holds `2.0*math.pi*torch.linalg.inv(cell).transpose(-2,-1)` (or `.inv(cell).T`); swap for the import `from gradwave.grids import reciprocal_cell` and the call. This is the highest-value item: it collapses the 6 drift-prone differentiable copies **and** the original numpy definition behind a single source of truth, so the strain graph and the energy path can no longer diverge.

---

### 2. Symmetric-tensor → ASE Voigt packing — reuse existing (from winner, unchanged)

**Home:** `postscf/elastic.py` — reuse the existing `stress_to_voigt(sigma)` (elastic.py:73), which already emits `[s00,s11,s22,s12,s02,s01]` via its `_VOIGT` table.

**Migration:** `from gradwave.postscf.elastic import stress_to_voigt` into `calculator.py`; replace the three inline list-builds at `calculator.py:594, 1039, 1215`. `elastic` does not import `calculator`, so no import cycle. Zero new code.

---

### 3. `inv_softplus` / `inv_sigmoid` — canonicalize into `dtypes.py` (from winner, unchanged)

**Home:** `dtypes.py` (already the torch-dtype home, already imports torch).

**Contract:**
```python
def inv_softplus(y) -> torch.Tensor:
    yt = torch.tensor(float(y), dtype=RDTYPE); return yt + torch.log(-torch.expm1(-yt))
def inv_sigmoid(y) -> torch.Tensor:
    yt = torch.tensor(float(y), dtype=RDTYPE); return torch.log(yt) - torch.log1p(-yt)
```

**Migration:** collapse the 3 copies at `exchange_multik.py:225/230`, `core/xc/learnable.py:76`, `scf/learned_precond.py:59`. Canonical form returns a tensor. `learned_precond`'s current variant returns a Python float, so wrap its 3 use sites with `float(...)` to preserve behavior — those values seed `torch.tensor/full`, so an fp64 tensor coerces identically.

---

### 4. `sphere_quad(nz, nphi)` — Gauss-Legendre-z × uniform-φ quadrature (from winner)

**Home:** `core/gaunt.py` (already `from scipy.special import roots_legendre`; the angular-math module).

**Contract / signature:**
```python
def sphere_quad(nz: int, nphi: int) -> tuple[np.ndarray, np.ndarray]:
    z, wz = roots_legendre(nz)
    phi = np.arange(nphi) * (2.0*np.pi/nphi)
    zz, pp = np.meshgrid(z, phi, indexing="ij"); st = np.sqrt(1.0 - zz**2)
    dirs = np.stack([st*np.cos(pp), st*np.sin(pp), zz], -1).reshape(-1, 3)
    w = (wz[:, None] * np.full(nphi, 2.0*np.pi/nphi)).reshape(-1)
    return dirs, w
```

**Migration (medium item, flagged separately from the low tail):** rewrite `paw_symmetry.py`'s `_sphere_quad(lmax)` as a call with the size mapping it already implies — `dirs, w = sphere_quad(2*lmax+2, 4*lmax+3)` (i.e. `(lmax'+2, 2·lmax'+3)` with `lmax'=2·lmax`); repoint `paw_onsite.py:152` and the in-module caller at `gaunt.py:70`. Verify the two callers' existing `nz/nphi` choices reduce to the same integers before deleting their local builders.

---

### 5. `_fft_grid` — keep one, import it (from winner, unchanged)

**Home:** `api/system.py:110` (the `System`/`USPPSystem` home). Body is the trivial `return system.grid`.

**Migration:** `from gradwave.api.system import _fft_grid` in `calculator.py`, delete calculator's copy at `calculator.py:95`.

---

### 6. `coulomb_potential_q` vs `exchange.coulomb_potential` — DEFER (from winner, unchanged)

Do **not** unify under this theme. `exchange.coulomb_potential(sigma_r, shape, g2)` uses a hard-coded `4πe²/G²` masked kernel; `exchange_multik.coulomb_potential_q(sigma_r, q, g_cart, mode, omega)` routes through `coulomb_kernel`. Unifying forces `g_cart`+`mode` plumbing through every plain-exchange call site — larger and riskier than the entire rest of this batch combined. Leave both; add a one-line cross-reference comment. Documented future direction if unified: `exchange.coulomb_potential` → `coulomb_potential_q(q=0, mode="full")`.

---

### Out of scope (do not fold in)

`esm.py:445` (high severity — net-charge term dropped in the capacitor-ESM stress path) is a **correctness bug**, not a duplication. No extraction here touches it; it needs its own targeted patch to the ESM stress term. Flag it as a separate work item so it is not silently considered "handled" by this refactor.

---

### Net effect

Seven tail items land as: **1 canonical backend-generic function** (item 1, killing 6+1 copies and the drift root-cause), **3 reuse-existing zero-new-code moves** (items 2, 5, and the elastic packing), **2 small canonicalizations** into their natural homes (items 3, 4), and **2 explicit deferrals with recorded direction** (item 6 and the esm.py:445 bug). No new util package, no new abstraction layer, and the single behavioral risk points (learned_precond float-wrap, paw quadrature size mapping) are called out with their guards.

### One scaled/alchemical system-builder

The scaled-system builder is duplicated as _eos_build vs run_eos._build_at (eos.py:26), and the linear alchemical blend formula is re-inlined inside _alchemical_ionic_terms instead of reusing alchemical_charges / blend_local_table (alchemical.py:121). Both construct a parameter-scaled system; one builder covers volume-scaling and alchemical blending. Low severity, small breadth.

Proposal scores: 37/40 — Accept — two clean, verified-bit-identical DRY consolidations that collapse duplicated builders into their canonical single source; low risk, autograd/behavior preserved, squarely on-theme.; 33/40 — Accept — a well-reasoned, behavior-preserving dedup; the EOS VolumeScan is high-value and faithful, and BlendWeight is numerically exact, though it should also absorb the identical scalar/per-atom fork in _alchemical_core_density to fully deliver its "one normalization step" claim.; 32/40 — Solid, honestly-scoped unification — accept Part A's _lerp reroute and the EOS builder collapse; defer or drop the speculative Tensor-scale autograd seam to keep it a pure, verifiable refactor. Two unmentioned nits: torch.lerp is exact at w=0 but only ~1e-16-exact at w=1 (a+ (b-a) != b), grazing the docstrings' endpoint-exactness claim (negligible vs SCF tol), and the `_eos_build`->`_build_scaled` rename must also update the `_eos_spoke_worker` call site (L67), which the sketch only implies.

**Recommendation:**

Grounded both files (`api/eos.py` and `scf/alchemical.py` in `/home/wladerer/github/gradwave`). Note: the winner's "eos.py" line numbers (108-109, 111-112, 137, 165, 191; `_build_at` at 115-133; `_eos_build(inp, upfs, uspp, soa, scale, fixed)`) resolve to **`src/gradwave/api/eos.py`**, not `postscf/eos.py`. The synthesis below is the reconciled recommendation.

---

# Recommended remediation — "One scaled/alchemical system-builder"

Theme invariant to establish: **exactly one function scales a system, and exactly one pair of helpers blends an alchemical atom.** The winner satisfied this by convention (forwarding calls); the grafted runner-up upgrades edit (1) so the invariant is *structural* — the duplication cannot silently reappear, and serial-vs-SeedPool E(V) cannot drift because both paths are the same code object.

Two independent edits, both float-path bit-identical.

## Edit (1) — EOS: collapse the triplication into one value object

**Problem (verified triplication).** Three verbatim copies of the isotropic-scale body exist in `src/gradwave/api/eos.py`:
- `_eos_rebuild` (L21-23) — worker-side derivation of `(upfs, uspp, soa)`
- `_eos_build` (L26-48) — module-level builder used by the SeedPool worker
- `run_eos._build_at` (L115-133) — the identical body re-inlined as an in-process closure

The winner deduped (2→1) by pointing `_build_at` at `_eos_build`. The graft goes to 1: fold all three into a single frozen value object that is the sole scaler.

**The object / contract.** Add one module-level frozen dataclass in `src/gradwave/api/eos.py`:

```python
@dataclass(frozen=True)
class _VolumeBuilder:
    """The single isotropic-scale system builder for the EOS volume scan.
    Exactly one place maps a -> a * scale**(1/3); the grid-pinning pre-pass,
    the serial chain, the SeedPool reference build, and the spoke worker all
    route through build_at, so serial and SeedPool E(V) cannot structurally drift."""
    inp: Input
    upfs: Any
    uspp: bool
    soa: Any

    @classmethod
    def from_input(cls, inp: Input) -> "_VolumeBuilder":          # replaces _eos_rebuild
        _species, upfs, soa = _species_upfs(inp)
        return cls(inp, upfs, _is_uspp(upfs), soa)

    def build_at(                                                  # replaces _eos_build + _build_at
        self, scale: float, fft_shape: tuple[int, ...] | None
    ) -> tuple[System | USPPSystem, Any]:
        import numpy as np
        cell = np.asarray(self.inp.atoms.cell.array, dtype=float) * scale ** (1.0 / 3.0)
        pos = self.inp.atoms.get_scaled_positions() @ cell
        if self.uspp:
            from gradwave.scf.uspp import setup_uspp
            return setup_uspp(
                cell, pos, self.soa, _as_paws(self.upfs), ecut=self.inp.ecut,
                kmesh=self.inp.kpoints.mesh, ecutrho=self.inp.ecutrho,
                nbands=self.inp.nbands, use_symmetry=self.inp.symmetry,
                fft_shape=fft_shape), cell
        from gradwave.scf.loop import setup_system
        return setup_system(
            cell=cell, positions=pos, species_of_atom=self.soa,
            upfs=_as_upfs(self.upfs), ecut=self.inp.ecut, kmesh=self.inp.kpoints.mesh,
            kshift=self.inp.kpoints.shift, nbands=self.inp.nbands,
            use_symmetry=self.inp.symmetry, fft_shape=fft_shape), cell
```

**Signatures.** `_VolumeBuilder.from_input(inp: Input) -> _VolumeBuilder`; `builder.build_at(scale: float, fft_shape: tuple[int,...] | None) -> (System | USPPSystem, cell)`. The `build_at` contract is exactly the old `_eos_build`/`_build_at` contract — same body, same return tuple.

**Where it lives.** Module level in `src/gradwave/api/eos.py`, so it is importable and picklable — but see the picklability note below.

**Migration path (mechanical, behavior-preserving):**
1. Delete `_eos_rebuild` (L21-23) and `_eos_build` (L26-48); add `_VolumeBuilder` in their place.
2. In `_eos_spoke_worker` (L59-72): replace L66-67 with
   `builder = _VolumeBuilder.from_input(spoke.inp)` / `system, cell = builder.build_at(spoke.scale, spoke.fixed)`.
   **`_EosSpoke` is unchanged** — it still carries only `(inp, scale, fixed, ckpt_path, idx)`. Crucially, `upfs`/`soa` are still *re-derived on the worker* via `from_input`, never shipped across the pickle boundary. This preserves the current design's deliberate "derive-on-worker" behavior; the frozen object holding `upfs` is only ever instantiated inside a process, never pickled.
3. In `run_eos`: after the existing L108-109 derivation, add `builder = _VolumeBuilder(inp, upfs, uspp, species_of_atom)` (reuse the already-derived values — no second `_species_upfs` call). Delete the hoisted locals `cell0` (L111) and `frac` (L112) and the whole `_build_at` closure (L115-133).
4. Rewrite the three call sites: L137 `_build_at(s, None)` → `builder.build_at(s, None)`; L165 → `builder.build_at(scales[ref_idx], fixed)`; L191 → `builder.build_at(s, fixed)`.

Deadnix is satisfied (cell0/frac removed; `upfs`/`uspp`/`species_of_atom` are still read to construct the builder). The float path is bit-identical because `build_at`'s body is the verbatim old body.

*Fallback if `_VolumeBuilder` is judged too much surface for this changeset:* degrade to the winner's exact edit (forward `_build_at` → `_eos_build`, delete `cell0`/`frac`). Same 3 call sites, no new type. The value object is the recommended target; the forward is the minimal floor.

## Edit (2) — Alchemical: route through the existing blend helpers (winner, unchanged)

In `src/gradwave/scf/alchemical.py`, `_alchemical_ionic_terms` (L114-128) re-inlines the two linear blends that `alchemical_charges` (L45-50) and `blend_local_table` (L64-70) already own. Replace L121-126 to call the helpers:

```python
def _alchemical_ionic_terms(lam, na, z_a, z_b, tab_a, tab_b, shape, pd_a, pd_b):
    lam = torch.as_tensor(lam, dtype=RDTYPE)
    if lam.dim() == 0:
        charges = alchemical_charges(z_a, z_b, lam).reshape(()).repeat(na)
        vloc_atom = blend_local_table(tab_a, tab_b, lam).unsqueeze(0).expand(na, *shape)
    else:
        charges = alchemical_charges(z_a, z_b, lam)
        w = lam.reshape(na, *([1] * tab_a.dim()))
        vloc_atom = blend_local_table(tab_a, tab_b, w)
    proj = [blend_projector_data(pd_a[k], pd_b[k], lam) for k in range(len(pd_a))]
    return charges, vloc_atom, proj
```

The runner-up's only qualification is honored: the scalar branch keeps `.unsqueeze(0).expand(na, *shape)`, so the `na` dense-box tables are a zero-copy broadcast view — no `na` materialized copies. `blend_local_table` broadcasts identically whether given a scalar `lam` or the reshaped per-atom `w`, and both helpers call `torch.as_tensor(..., RDTYPE)` internally, so behavior is bit-identical. No new functions/types here.

## Why this is the right synthesis

- Winner's structure (two self-contained edits, float path bit-identical, no change to public signatures or call-site arity) is preserved.
- The grafted "VolumeScan collapse" / single-`build_at` value object is folded into edit (1), turning "one function scales a system" from a convention into a structural invariant — the strongest, most theme-aligned runner-up element, and the one that closes the serial-vs-SeedPool drift risk at its root.
- The grafted zero-copy `.expand` caveat is folded into edit (2), so the dedup adds no memory regression.
- Picklability is respected by keeping `_EosSpoke` as-is and reconstructing the builder on the worker via `from_input`, so no heavy `upfs` objects cross the process boundary.

Relevant files: `/home/wladerer/github/gradwave/src/gradwave/api/eos.py`, `/home/wladerer/github/gradwave/src/gradwave/scf/alchemical.py`.

### Deduplicate physical-quantity reductions with a documented owner

Several physical quantities are computed by parallel hand-coded reductions: the energy-term breakdown is defined in three places and the documented single source energies_eV_dict omits fock/e0 which are bolted on downstream (checkpoint.py:29 - a reporting-completeness bug); _atomic_moment_vectors duplicates moment_config._atomic_moments (magnetism.py:69); the per-spin occupations+entropy loop is duplicated between shared_fermi_occupations and constant_mu_occupations (common.py:326); and band_structure_nc re-derives v_r/b_xc/v_tau already built by _nc_effective_potential (noncollinear.py:936). Give each quantity one authoritative producer. Low severity but includes an incomplete documented-source-of-truth.

Proposal scores: 37/40 — Strong accept: four accurate, self-contained one-owner refactors that preserve numerics exactly and even fix a latent Fock round-trip bug; only gap is under-specifying the one test (test_review_api.py:74 exact-keyset assert) that the key additions will break.; 33/40 — Accurate, physics-aware dedup grounded in the real code; land A/C then B (with snapshot updates), gate D behind the NC meta-GGA + constrained-moment regression.; 32/40 — Physically safe, high-value dedup anchored by an excellent single-source-of-truth energy registry; land item (1) but fix two gaps — the output.py refactor drops the smearing row and dynamic disp label, and item (2) misses the 4th M_I einsum at checkpoint.py:203.

**Recommendation:**

# Recommended remediation — "Deduplicate physical-quantity reductions with a documented owner"

## Unifying contract (the principle all four edits obey)

Every physical quantity that is currently computed in two or more places gets exactly **one producer that is also its documented owner** — a named function or method whose docstring declares "single source for X." No new module and no new class is introduced. Where the winning proposal hand-copies the owner's *contents* into a serializer (and that copy has already drifted), we graft the runner-up's structural move: **the owner emits itself**, so the set of things it owns and the set of things it exports are equal *by construction*, not by discipline. This is the difference between "deduplicated today" and "cannot re-duplicate tomorrow," and it is exactly what the theme title's "documented owner" asks for.

Backbone = the winner's four self-contained edits. Graft = B's `dataclasses.fields`-driven self-serialization for edit (1), which is the only one of the four where the "producer" was itself a drifting hand-written copy rather than a genuine second computation.

---

## Edit (1) — Energy-term breakdown: make `EnergyBreakdown` serialize itself

**Owner:** the `EnergyBreakdown` dataclass at `core/energies/total.py:48`. It already owns every term as a field (`kinetic … esm`, lines 50–61) and every derived scalar as a property (`total` 66, `free_energy` 73, `e0` 78). It should therefore also own the eV dict — nothing else has the standing to.

**Grounded bug this fixes (verified, not hypothetical):** `energies_eV_dict` at `io/checkpoint.py:29` emits 11 keys and *omits `fock` and `esm`*. `api/summary.py:82-84` bolts `fock` and `e0` back on — but **not `esm`**. Net result today: a hybrid run's Fock term is missing from the *checkpoint* payload, and the open-boundary `esm` correction is missing from **both** checkpoint and summary. The hand-copied key list has already drifted away from the dataclass fields. Fixing the copy is not enough; the copy must be deleted.

**New contract / signature — a method on the dataclass:**

```python
# core/energies/total.py, inside class EnergyBreakdown
import dataclasses

def as_eV_dict(self) -> dict[str, float]:
    """THE serialization of this breakdown (eV). Single source for the
    checkpoint payload (io/checkpoint.py) and the api summary (api/summary.py).
    Term keys are derived from the dataclass fields, so a new energy term is
    added in ONE place (a field) and propagates to every consumer; derived
    scalars (total/free_energy/e0) are appended explicitly."""
    d = {f.name.rstrip("_"): float(getattr(self, f.name))   # nonlocal_ -> "nonlocal"
         for f in dataclasses.fields(self)}
    d["total"] = float(self.total)
    d["free_energy"] = float(self.free_energy)
    d["e0"] = float(self.e0)
    return d
```

The `f.name.rstrip("_")` handles the sole field/key mismatch (`nonlocal_` → `"nonlocal"`). Because the term keys come from `dataclasses.fields`, `{terms in the breakdown} == {keys emitted}` is now true by construction — the drift class that produced the `fock`/`esm` bug becomes unrepresentable.

**Where it's called (migration path):**

```python
# io/checkpoint.py:29 — energies_eV_dict collapses to a one-line shim (or delete
# it and call e.as_eV_dict() at the payload site, checkpoint.py:104)
def energies_eV_dict(e) -> dict:            # kept only for import stability
    return e.as_eV_dict()

# api/summary.py:81-85 — drop the bolt-on entirely
    "energies_eV": e.as_eV_dict(),          # was {**energies_eV_dict(e), "fock":.., "e0":..}
```

**Backward compatibility (verified):** `e.e0` ≡ `total + 0.5*smearing` ≡ the old summary's `0.5*(total+free_energy)` (since `free_energy = total + smearing`) — byte-identical value, so no summary regression. The one text consumer, `io/output.py:189 _energy_lines`, reads every key via `e.get(key, 0.0)` and already lists a `"fock"` row, so old JSONs still render and the newly-present `fock`/`esm` keys just populate. New keys that lack a display row (e.g. `esm`) are simply not printed until a row is added — data is captured regardless.

---

## Edit (2) — Atomic moment vectors: promote the one true einsum

**Owner:** `_atomic_moments(m, weights, cell_factor)` at `postscf/moment_config.py:83` — `torch.einsum("axyz,ixyz->ai", weights, m) * cell_factor`, the ∫w_I·m⃗ reduction. Its two in-module callers (`:115`, `:157`) already use it, and the sibling `atomic_weights` (`:59`) is already public, so promotion is idiomatic.

**Duplicates to retire (there are three copies, not one):**
1. `postscf/magnetism.py:70` — `_atomic_moment_vectors` re-writes the same einsum verbatim.
2. `scf/noncollinear.py:~439` inside `_nc_effective_potential` — `m_at = torch.einsum("axyz,ixyz->ai", atom_weights, m) * cf` (the constraining-field branch). Same reduction, third copy.

**Signature (rename `_atomic_moments` → public `atomic_moments`, body unchanged):**

```python
def atomic_moments(m, weights, cell_factor) -> torch.Tensor:  # (na,3) [μB]
    return torch.einsum("axyz,ixyz->ai", weights, m) * cell_factor
```

**Migration:**

```python
# moment_config.py:115,157 — update the two internal call sites to the new name.
# magnetism.py:29 import; :69-71 body becomes a delegation:
from gradwave.postscf.moment_config import atomic_moments, atomic_weights
def _atomic_moment_vectors(system, m, weights):
    cf = system.grid.volume / system.grid.n_points
    return atomic_moments(m, weights, cf)          # (na,3) [μB]; call site :119 unchanged
# noncollinear.py:~439 — replace the inline einsum with atomic_moments(m, atom_weights, cf).
```

Fold the noncollinear copy in only if `noncollinear.py` is in scope for this pass; the magnetism delegation is fully self-contained on its own.

---

## Edit (3) — Per-spin occupations + entropy: one in-file helper, two callers

**Owner:** a new module-private helper in `scf/common.py`. This is the one case with no pre-existing owner — two genuine byte-identical loops (`shared_fermi_occupations:304-310` and `constant_mu_occupations:325-332`) differ only in that `constant_mu` also accumulates `n_e`. The helper folds that in and `shared_fermi` discards it. Reuse of an in-file loop, not a new abstraction.

```python
# scf/common.py
def _accumulate_occ_entropy(eigs_s, mu_t, scheme, width, kweights, nspin, g_spin, device):
    """Per-spin occupations + the −σS entropy term at a fixed μ. Single source
    for shared_fermi_occupations and constant_mu_occupations.
    Returns (occ_s, entropy_term, n_electrons)."""
    occ_s, ent, n_e = [], torch.zeros((), dtype=RDTYPE, device=device), 0.0
    for isp in range(nspin):
        o, s_ent = occupations_and_entropy(eigs_s[isp], mu_t, scheme, width, degeneracy=g_spin)
        occ_s.append(o)
        ent = ent - width * (g_spin * kweights[:, None] * s_ent).sum()
        n_e += float((kweights[:, None] * o).sum())
    return occ_s, ent, n_e
```

**Migration (both return signatures preserved exactly):**

```python
# shared_fermi_occupations (compute g_spin/mu_t as it does now):
occ_s, ent, _ = _accumulate_occ_entropy(eigs_s, mu_t, scheme, width, kweights, nspin, g_spin, device)
return occ_s, mu, ent
# constant_mu_occupations:
occ_s, ent, n_e = _accumulate_occ_entropy(eigs_s, mu_t, scheme, width, kweights, nspin, g_spin, device)
return occ_s, float(mu), ent, n_e
```

---

## Edit (4) — NC effective potential (v_r, b_xc): route the band path through the SCF producer

**Owner:** `_nc_effective_potential(...)` at `scf/noncollinear.py:415`, returning `(v_r, b_xc)`. `band_structure_nc` (`:936-945`) re-derives ρ_g→v_h→vxc_and_bxc→nonmagnetic-zeroing→v_r by hand, with no constraint. Have it build the still-local `vloc_r`, then call the owner with the constraint knobs turned off:

```python
# scf/noncollinear.py:936-945
vloc_r = g_to_r_box(local_potential_g(system.positions, system.species_index,
                                      system.vloc_tables, grid.g_cart, grid.volume), real=True)
v_r, b_xc = _nc_effective_potential(
    xc, res.rho, res.m, grid, system, vloc_r, nonmagnetic,
    None, 0.0, "perp", None, None, grid.volume, tau_up, tau_dn)
#   constrain_dirs=None, constrain_lambda=0.0, constrain_mode="perp",
#   constrain_target_mag=None, atom_weights=None  -> the unconstrained branch
```

With `constrain_dirs=None` and `nonmagnetic` passed through, the owner's constraint branch (the third atomic-moment einsum from edit 2) is never entered, so the band path gets exactly the SCF's v_r/b_xc. In scope only if `noncollinear.py` is being touched.

**Explicitly deferred:** the parallel v_τ operator-field re-derivation at `:947-957` could route through `_nc_metagga_step` (`:465`), but that helper's return shape is not `(v0, vvec)`-only, so unifying it needs a small return-shape refactor. Leave as a follow-up rather than widening this change.

---

## Recommended follow-up (extends edit 1's owner to full "by construction")

The grafted `as_eV_dict` makes `{fields} == {emitted keys}` structural. The stronger runner-up idea closes the remaining two gaps so that **{terms summed into `total`} == {keys emitted} == {rows printed}** all hold by construction:

- Derive the `total` property (`total.py:66`) from the same field iteration it now serializes (sum over `dataclasses.fields` minus the non-additive `smearing`, which is added only in `free_energy`), so a new term can never be added to the dict but forgotten in the sum.
- Drive `output.py:_energy_lines`' `shown` label list (`:189`) from a class-level `field → (label)` map on `EnergyBreakdown`, so a new term also auto-gets a printed row.

Do the `as_eV_dict` graft now (it fixes the live `esm`/`fock` bug and is low-risk); schedule the `total`/printer unification as the immediate next step — it is the piece that makes the checkpoint-completeness drift permanently unrepresentable.

---

## Migration & verification order

1. **(1)** first — smallest blast radius, fixes two real reporting bugs. Verify: round-trip a hybrid (`fock ≠ 0`) and an `open_z` (`esm ≠ 0`) run; assert `esm`/`fock` now appear in both `save_checkpoint` payload and `build_summary`, and that `energies_eV["e0"]` is unchanged vs. the old `0.5*(total+free_energy)` on a regression fixture.
2. **(3)** next — pure internal refactor, both public return signatures byte-identical; assert `n_e` from `constant_mu_occupations` and occupations/entropy from both callers are bit-unchanged.
3. **(2)** — swap the einsum for the promoted `atomic_moments`; assert moment vectors identical on a collinear + a noncollinear fixture.
4. **(4)** — only if `noncollinear.py` is in scope; assert `band_structure_nc`'s `v_r`/`b_xc` match the previous inline derivation on an NC band fixture.

Files touched: `core/energies/total.py`, `io/checkpoint.py`, `api/summary.py`, `postscf/moment_config.py`, `postscf/magnetism.py`, `scf/common.py`, and (optional) `scf/noncollinear.py`. No new module, no new class — every owner already existed; (3) is the only new symbol and it is module-private.
