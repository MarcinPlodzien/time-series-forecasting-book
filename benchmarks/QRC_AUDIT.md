# QRC implementation audit (`tsforecast/models/quantum_reservoir.py`)

Date: 2026-06-18. Scope: a careful, line-by-line correctness review of the quantum
reservoir against the physics and the Fujii-Nakajima scheme. This is a review by
reading, not a formal proof or an independent third-party reproduction. Verdict up
front: **no correctness bugs found.** One cosmetic dead import, and several
modelling choices (not errors) that should be, and are, disclosed in the text.

How to read the verdicts: [OK] correct, [NOTE] a modelling choice worth disclosing,
[NIT] cosmetic.

---

## 1. Pauli matrices and embedding (lines 51-63)

- `_I2,_X,_Y,_Z` are the standard 2x2 Paulis; `_Y=[[0,-i],[i,0]]`. [OK]
- `_kron_op(op,site,n)` builds `I⊗...⊗op⊗...⊗I` with `op` on `site`, using
  `mats[0]` as the leftmost (most significant) tensor factor, so qubit 0 is the
  most significant index. Internally consistent with `np.kron` ordering and with
  `_inject` below. [OK]

## 2. Constructor (lines 69-123)

- Defaults: `n_qubits=6` (D=64), `t=4`, `n_virtual=5`, full Pauli readout,
  `J_scale=1`, `h_x=1`, `h_z=0.5`, `ridge=None` (closed-loop selection),
  `washout=100`, `warmup_cap=120`, `seed=0`. Parameters are only stored here. [OK]

## 3. Reservoir construction `_build` (lines 126-157)

- `rng = default_rng(seed)`: couplings are drawn fresh on every `_build` but with a
  fixed seed, so the reservoir is reproducible, and different benchmark seeds give
  genuinely different reservoir instances (the intended source of QRC seed
  variance). [OK]
- Hamiltonian `H = Σ_i (h_x X_i + h_z Z_i) + Σ_{i<j} J_ij Z_iZ_j`, with
  `J_ij ~ U(-J_scale, J_scale)` drawn in the `i<j` loop. Each qubit gets its field
  terms exactly once; there are C(n,2)=15 coupling draws at n=6. This is the
  transverse-field Ising model of Eq.(qrc_tfim) in Chapter 8 with uniform fields
  (`h_x` is the transverse `g`, `h_z` the longitudinal `h`). [OK]
- `_Usub = expm(-i H · t/n_virtual)`: the sub-step propagator. With `t=4, nv=5` the
  sub-step time is 0.8, and `nv` sub-steps compose to the full per-input evolution
  `t`. So the total dynamics per input is the same for any `nv`; `nv` only adds
  intermediate readouts. `_Usubdag` is its conjugate transpose, used as `U ρ U†`.
  [OK]
- Observables: `one_body = Xs+Ys+Zs` (3n=18). `two_body`: for each pair `i<j`,
  "full" adds `A_i B_j` for all `A,B ∈ {X,Y,Z}` (9 per pair, 9·15=135). Total
  18+135 = **153** at n=6, stacked to `(153,64,64)`. Each `A_i B_j` with `i≠j` is a
  product of commuting Hermitian single-site operators, hence Hermitian, hence has
  a real expectation. Pairs are counted once (`i<j`). [OK]

## 4. Expectation values `_expect` (lines 159-164)

- `einsum("kij,ji->k", Obs, rho).real` computes `Σ_ij O[k,i,j] ρ[j,i] = Tr(O_k ρ)`
  for every observable in one pass. `.real` drops the numerical imaginary dust of a
  Hermitian expectation. [OK]

## 5. Input reset `_inject` (lines 166-175)

This is the only non-unitary step and the most error-prone, so checked carefully.
- `rest = D//2 = 32`. `rho.reshape(2,rest,2,rest)` splits the row index into
  `(q0_row, rest_row)` and the column index into `(q0_col, rest_col)`, valid
  because qubit 0 is the most significant index (matches `_kron_op`). [OK]
- `rho_rest = r[0,:,0,:] + r[1,:,1,:]` is exactly the partial trace over qubit 0,
  `Tr_0(ρ) = Σ_a ⟨a|_0 ρ |a⟩_0`. [OK]
- `psi = [√(1-u), √u]`, `rho_q0 = |psi⟩⟨psi|`, return `kron(rho_q0, rho_rest)`.
  This realises the reset channel `E_u(ρ) = |psi(u)⟩⟨psi(u)| ⊗ Tr_0(ρ)` of
  Chapter 8, Eq.(qrc input). It correctly discards qubit 0's prior correlations (a
  reset must) and preserves the rest as memory. Trace is preserved
  (`Tr(rho_rest)=Tr(ρ)`), so a normalised state stays normalised. [OK]

## 6. One input step `_step` (lines 177-188)

- Inject, then `for _ in range(nv): ρ ← U_sub ρ U_sub†; record _expect(ρ)`.
  Readouts land at times `t/nv, 2t/nv, …, t` after injection, concatenated into a
  feature vector of length `nv·153` (765 at nv=5). This is exactly the
  temporal-multiplexing / virtual-node construction. The persistent `ρ` carries
  memory across steps. [OK]

## 7. Input scaling `_to_unit/_from_unit` (lines 191-195)

- Min-max scaling to/from `[0,1]` using the training min/max. [NOTE] Test values
  outside the training range map outside `[0,1]`, but `_inject` clamps `u` to
  `[0,1]` and `forecast` clamps predictions before feeding back, so the encoding
  stays physical. Reasonable; just a modelling choice.

## 8. Training features `_features_targets` (lines 198-208)

- Start from the maximally mixed state `I/D` (input-independent), drive the
  reservoir along the inputs, and after `washout=100` steps collect
  `(bias=1, observables)` against the next sample `u[t+1]`. One-step-ahead
  supervised pairs with a persistent state and a bias term. The washout discards
  the transient that still remembers the arbitrary initial state. [OK]

## 9. Closed-loop ridge selection `_select_ridge_closedloop` (lines 210-242)

- Splits the training inputs at `cut = 0.8·n`. Builds features once on the fit part
  (the features do not depend on the ridge), then for each ridge in the coarse grid
  solves the cheap readout, sets `self._W`, runs a short rolling-origin closed-loop
  forecast over the held-out tail, and keeps the ridge with the highest validation
  `C_tot`. Falls back to a small ridge when there is too little data to split.
  This is the corrected, objective-aligned selection (closed-loop, not one-step),
  and it selects on validation only, never the test continuation. [OK]
- The method mutates `self._W` during the sweep; `fit` re-solves on the full
  training set afterward, so nothing leaks. `self._u_min/_max` are set in `fit`
  before this is called. [OK]

## 10. `fit` (lines 244-259)

- `_build`; set scaling from train; choose `eff` ridge (closed-loop if `ridge is
  None`, else the fixed float); build features on the full training set; solve the
  readout; store `_ridge_used`. Correct order, and `_build` runs once. [OK]

## 11. Closed-loop rollout `forecast` (lines 262-278)

- Re-drive the reservoir along the recent history (capped to `warmup_cap=120`) to
  set `ρ`, take the first prediction from the last feature vector, then iterate:
  clamp to `[0,1]`, record, feed the prediction back as the next input, predict
  again. Returns the unscaled trajectory. Standard free-running rollout. [OK]
- [NOTE] Clamping to `[0,1]` bounds the rollout and can present a divergence as
  saturation rather than blow-up; this is a deliberate, common safeguard and the
  capacity metric handles it.

---

## Modelling caveats (disclosed, not bugs)

1. **Expectation-value (ensemble) regime.** Observables are exact `Tr(Oρ)`: no shot
   noise and no measurement back-action. This is the standard Fujii-Nakajima
   idealisation and is why the intermediate virtual-node readouts are free here; on
   hardware they would require repeated runs or weak measurement. Stated in the
   docstring and in the new Chapter 8 paragraph.
2. **Reset encoding is coherent amplitude encoding**, non-analytic at the endpoints;
   Chapter 8 already notes the Volterra expansion is then an approximation away from
   the endpoints.
3. **Min-max input scaling and `[0,1]` clamping** are pipeline choices, not physics.
4. **No quantum-advantage claim**; only a linear readout is trained, on a fixed
   random substrate.

## Cosmetic / cleanup

- [NIT] Line 47 `from ..readout import solve_readout` is now unused (`fit` imports
  `ridge_solve` locally). Harmless; remove for tidiness.

## What I did NOT do, and how you can independently verify

- I did not formally prove the channel is CPTP or run an external simulator
  cross-check. To verify independently:
  - Reproducibility: rerun and confirm the QRC rows reproduce.
  - Sanity invariants: after `_inject`, `Tr(ρ)=1` and `ρ` is Hermitian PSD; after
    each `_step`, eigenvalues of `ρ` stay in `[0,1]`. A 5-line assertion script can
    check these on a few steps.
  - Consistency: the QRC nv=5 row in experiment 03 must match the nv=5 point of the
    nv-scan (experiment 07) within seed noise.
  - Ground-truth-adjacent: on Mackey-Glass the QRC sits among the strong methods, as
    expected for a long-Lyapunov, highly predictable signal.

## Empirical verification performed (2026-06-18)

Ran the density-matrix invariants over 40 reset-and-evolve steps at n=6, nv=5:
`Tr(ρ)=1` to 1e-9, `ρ` Hermitian to 1e-9, and all eigenvalues in `[0,1]` at every
step. So the reset channel followed by the unitary sub-steps keeps `ρ` a valid
(CPTP-image) quantum state in practice, confirming Sections 5-6 at runtime.

Confidence: high that the code faithfully implements the intended QRC mechanism and
contains no correctness bug. The numbers it produces are therefore trustworthy as
"what this model does", subject to the modelling choices above, which are the honest
scope of the toy.
