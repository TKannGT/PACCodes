# ============================================================
# RunGenericPACSetThenSparseTV_RhoZhat.jl
#
# Two-stage experiment driver:
#
#   STAGE 1 — Information-set design under a generic PAC transform.
#             Kicker-escalation search (1-swap, 2-swap, 3-swap) on a
#             tail-event FER objective.  The freeze and add halves of every
#             swap can now use the empirical metrics described below.
#
#   STAGE 2 — From-scratch PAC precoder construction.
#             Starting from identity/generic/custom, iteratively add taps
#             using a non-greedy batch search.  In each step, 1-edge and
#             multi-edge kicker candidates are screened, the best candidates
#             are fully confirmed, and the best confirmed move is committed.
#
# New empirical scoring (Stage 1):
#
#   rho_i (freeze efficacy).  For each info bit i with source_sigma_i > 0,
#     rho_i = both_count_i / source_count_i
#           ~= P(u_i = 1 | v_i = 1, dangerous path).
#     Bits with rho_i ~ 1 propagate v_i = 1 into u_i = 1 reliably, so
#     freezing v_i actually disrupts a real BM event at step i.  Bits with
#     rho_i ~ 0 carry v_i = 1 as cargo that the precoder cancels into
#     u_i = 0; freezing them eliminates dangerous paths but the underlying
#     PM win is elsewhere and similar paths can re-emerge.
#
#     When freeze_use_rho = true the removal priority is
#       priority_rho_i = source_sigma_i * max(rho_i, freeze_rho_floor),
#     which equals both_sigma_i with floor = 0.  source_sigma (not post_sigma)
#     is the multiplier base because freezing acts on v: it eliminates exactly
#     the dangerous paths with v_i = 1, and rho is the rate that those kills
#     actually correspond to a u_i = 1 event at step i.  post_sigma * rho
#     would over-credit positions whose u_i = 1 comes from T-mixing of earlier
#     v bits, which a v_i freeze cannot touch.
#
#   Zhat_j^dang (add-side reliability).  For each currently-frozen bit j,
#     Zhat_j^dang = E[ exp(-|LLR_j|/2) | trial is dangerous ]
#     is a Bhattacharyya-like reliability conditioned on the trials where
#     the current code actually fails.  Lower Zhat_j^dang means the
#     bit-channel at j stays confident on the failure tail, which is the
#     population where the new info bit would have to perform.  Compared
#     to DEGA (which averages over the entire channel distribution
#     including easy trials), this ranking should push truly useful add
#     candidates toward the top of the list.  Companion statistics
#     (mean and 10th-percentile |LLR|, sign-error rate against the frozen
#     all-zero u_j) are also computed for diagnostic purposes.
#
# Rewrite notes vs the previous version of stage 2:
#   * Edge gain switched from (lambda + sigma)*M[i,j] to per-path
#     saved-path count.  Removes the first-error conditioning artifact
#     that under-counted exactly the (high-sigma, low-lambda) edges
#     the precoder needs.
#   * 2-edge kicker now competes with 1-edge moves in the same batch,
#     instead of only running after all single-edge moves fail.
#   * Multi-edge scoring uses combined saved paths, not positive excess only.
#   * Optional exploratory near-neutral moves are allowed while the best Tmat
#     seen over the walk is tracked and returned.
#   * L0-aware edge scoring separates trials where the correct all-zero path
#     survives the final SCL list from trials where it has already been pruned.
#     For lost-L0 trials, candidate taps only get saved-path credit if their
#     frozen target occurs before a configurable urgency deadline.
#
# Files required next to this script:
#   DangerousPathSpectrum.jl
#   TailAwarePM.jl                 # optional, only for :dega reliability
#   ClassicalPolarCodeFuncSCD.jl   # via DangerousPathSpectrum's include
# ============================================================

using Dates
using DelimitedFiles
using LinearAlgebra
using Printf
using Statistics

const SCRIPT_DIR = @__DIR__

include(joinpath(SCRIPT_DIR, "DangerousPathSpectrum.jl"))

if !isdefined(Main, :TailAwarePM)
    include(joinpath(SCRIPT_DIR, "TailAwarePM.jl"))
end

include("../Misc.jl")

const DPS = DangerousPathSpectrumClean

# ------------------------------------------------------------
# SLURM-friendly stdout: line-buffered when not a TTY.
# ------------------------------------------------------------
function setup_unbuffered_stdout()
    Sys.iswindows() && return
    isa(stdout, Base.TTY) && return
    try
        fp = unsafe_load(cglobal(:stdout, Ptr{Cvoid}))
        ccall(:setlinebuf, Cvoid, (Ptr{Cvoid},), fp)
    catch
    end
    return
end
setup_unbuffered_stdout()


# ------------------------------------------------------------
# Run logging helpers: tee all script stdout/stderr to a file.
# This does not change any algorithmic logic; it only mirrors output.
# ------------------------------------------------------------

mutable struct TeeIO <: IO
    primary::IO
    secondary::IO
end

Base.isopen(io::TeeIO) = isopen(io.primary) && isopen(io.secondary)

function Base.write(io::TeeIO, x::UInt8)
    write(io.primary, x)
    write(io.secondary, x)
    return 1
end

function Base.write(io::TeeIO, s::AbstractString)
    n1 = write(io.primary, s)
    write(io.secondary, s)
    return n1
end

function Base.write(io::TeeIO, xs::Vector{UInt8})
    n1 = write(io.primary, xs)
    write(io.secondary, xs)
    return n1
end

function Base.write(io::TeeIO, p::Ptr{UInt8}, n::UInt)
    n1 = write(io.primary, p, n)
    write(io.secondary, p, n)
    return n1
end

function Base.flush(io::TeeIO)
    flush(io.primary)
    flush(io.secondary)
    return nothing
end

function _active_log_path(cfg)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    logdir = ensure_dir(joinpath(cfg.outdir, "_active_logs"))
    return joinpath(logdir, "stage123_stdout_stderr_$(timestamp)_pid$(getpid()).log")
end

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

Base.@kwdef mutable struct PipelineConfig
    # Code/channel parameters.
    N::Int = 64
    K::Int = 32
    EbN0_dB::Float64 = 3.0
    L::Int = 32

    generic_taps::Vector{Int} = [1, 0, 1, 1, 0, 1, 1]

    # ── Stage-1 custom Tmat ──
    # By default Stage 1 builds Tgeneric from generic_taps (the standard
    # Arikan PAC convolutional precoder).  If you want set design to run
    # against a different precoder — e.g., the sparse T from a previous
    # Stage 2 run, or a hand-crafted matrix — provide it via one of:
    #   set_custom_Tmat       — an N×N AbstractMatrix{<:Integer}, used directly
    #   set_custom_Tmat_path  — path to a CSV with the matrix (Int)
    # Priority: matrix > path > taps fallback.  The matrix must be
    # upper-triangular with 1s on the diagonal (the same shape pac_tmat_from_taps
    # produces); we validate this on load.  This affects Stage 1 only — Stage 2
    # still has its own pac_initial/pac_custom_Tmat knobs.  The custom matrix
    # is also what `evaluate_constructed_precoder` uses as the "generic" baseline
    # in the final comparison, so the head-to-head FER plot will show
    # (identity, your_custom_T, constructed_T) instead of (identity, generic_PAC,
    # constructed).
    set_custom_Tmat::Any = nothing
    set_custom_Tmat_path::Union{Nothing,String} = nothing

    initial_A::Union{Nothing,Vector{Int}} = nothing
    reliability_method::Symbol = :dega
    reliability_Q::Int = 24
    # Reliability method used only for the initial information-set selection
    # (when initial_A is not provided).  Defaults to :rm (Hamming-weight / RM order)
    # so the search starts from the canonical RM baseline regardless of the DEGA
    # profile used for iterative add-pool ordering.  Set to :dega to match
    # reliability_method, or to any other supported symbol.
    initial_reliability_method::Symbol = :rm

    # ── Stage 1: information-set design ──
    set_iters::Int = 5
    n_trials_set::Int = 200_000
    n_trials_candidate::Int = 50_000
    nremove::Int = 4
    nadd::Int = 12
    max_candidates_per_iter::Int = 24

    # Non-greedy Stage-1 confirmation.  Each set-optimization iteration now
    # pools candidates from all configured kicker_sizes, small-screens them,
    # fully evaluates the top set_full_eval_top_k candidates, and accepts the
    # best full-budget result.  This avoids committing to a greedy replacement
    # before comparing other high-ranked swaps at the full budget.
    set_full_eval_top_k::Int = 8
    set_full_eval_only_small_improvers::Bool = false

    set_risk_score::Symbol = :sigma
    set_objective::Symbol = :fer
    min_set_improvement::Float64 = 0.0

    risk_mix::NamedTuple = (lambda = 0.0, sigma = 1.0)

    # Which support statistic should play the role of sigma in Stage-1
    # information-set search.
    #   :source  -> pre-convolution/source-domain path bit u_i
    #   :post    -> post-convolution/channel-facing bit v_i charged by LLR_i
    #   :hybrid  -> set_sigma_pre_weight * source_sigma_i
    #              + set_sigma_post_weight * post_sigma_i
    # The default keeps the original behavior.  Use :post for the experiment
    # discussed in the chat.
    set_sigma_domain::Symbol = :post
    set_sigma_pre_weight::Float64 = 1.0
    set_sigma_post_weight::Float64 = 1.0

    # When set_sigma_domain != :source, the driver constructs Stage-1 swap
    # candidates locally so that the chosen sigma is actually used for ranking.
    # :auto uses DPS.suggest_tail_swaps_kicker for :source and the local
    # generator for :post/:hybrid.
    set_candidate_generator::Symbol = :local  # :auto, :dps, :local

    # ── Empirical rho_i / Zhat_j^dang knobs (off by default for back-compat) ──
    # rho_i = both_count_i / source_count_i ~ P(u_i = 1 | v_i = 1, dangerous).
    # When freeze_use_rho = true, the removal priority is replaced by
    #   priority_rho_i = source_sigma_i * max(rho_i, freeze_rho_floor)
    #                  ~ P(v_i = 1, dangerous) * P(u_i = 1 | v_i = 1, dangerous)
    #                  ~ P(v_i = 1 AND u_i = 1, dangerous) (= both_sigma when floor=0).
    # source_sigma is used as the base because freezing acts on v: it eliminates
    # exactly the dangerous paths with v_i = 1, so the kill-able mass is the
    # source-domain support, and rho is the rate at which those kills actually
    # disrupt the BM at step i.  Using post_sigma here would inflate the score
    # with incidental u = 1 events generated by T-mixing from earlier v bits,
    # which freezing v_i cannot touch.  set_sigma_domain still controls the
    # "sigma" column displayed in the report; only the removal score changes.
    freeze_use_rho::Bool = false
    freeze_rho_floor::Float64 = 0.0
    # If > 0, candidates with rho_i strictly below this threshold are dropped
    # from the remove pool entirely (hard filter rather than soft weight).
    freeze_rho_min::Float64 = 0.0

    # Zhat_j^dang for add-side ranking.  When add_use_zhat_dang = true, the
    # local candidate generator ranks the add pool by Zhat ascending (lower
    # = more reliable on dangerous trials).  add_zhat_weight controls how
    # strongly Zhat enters the composite (remove_priority - beta * Zhat)
    # ordering of (i, j) swap pairs; setting it to 0 keeps the existing
    # remove-side / add-reliability tie-break.
    add_use_zhat_dang::Bool = false
    add_zhat_weight::Float64 = 1.0
    # Numerical floor below which Zhat is treated as zero; protects against
    # exp(-|LLR|/2) underflowing to literally 0 for very reliable bits.
    add_zhat_floor::Float64 = 0.0

    # ── post_sigma filter for the add pool (within-run quantile) ──
    # add_post_sig_keep_quantile is the q-quantile of post_sigma over the
    # current frozen bits.  Only frozen bits with post_sigma_j ≤ q-quantile
    # remain in the add pool, so positions where the current precoder T is
    # actively routing wrong-path mass (post_sig > 0) are deprioritized
    # against structurally-clean positions (post_sig ≈ 0).  The threshold
    # adapts automatically to the run's post_sig scale — no hard-coded
    # absolute cutoff required.
    #
    #   add_post_sig_keep_quantile = 1.0  → no filter (legacy behavior).
    #   add_post_sig_keep_quantile = 0.5  → keep bottom 50% by post_sig.
    #   add_post_sig_keep_quantile = 0.0  → keep only post_sig = 0 (or the
    #                                        minimum among frozen bits).
    # If the filter would empty the pool, it falls back to the unfiltered F.
    add_post_sig_keep_quantile::Float64 = 0.5

    add_strategy::Symbol = :top_reliability

    kicker_sizes::Vector{Int} = [1, 2, 3]
    max_kicker_candidates::Int = 128

    # When set_kicker_escalate = true, Stage 1 tries kicker_sizes one at a time
    # in order and commits the first size that produces an improvement.  It only
    # escalates to a larger kicker size when the smaller size finds no improvement.
    # When false (default), all sizes are pooled and screened together so the
    # best swap across all sizes is accepted (original non-greedy batch behavior).
    set_kicker_escalate::Bool = false

    # ── Stage 2: iterative PAC precoder construction ──
    pac_max_edges::Int = 96             # raised: matches generic-PAC entry count
    pac_max_span::Int = 24
    pac_top_edges_per_step::Int = 16
    pac_min_improvement::Float64 = 0.0
    pac_n_trials_small::Int = 50_000
    pac_n_trials_full::Int = 200_000
    pac_min_events_for_edges::Int = 20
    pac_target::Symbol = :frozen        # :frozen or :all
    # 1-edge and multi-edge candidate sets are evaluated in one batch.
    pac_kicker_sizes::Vector{Int} = [1, 2]
    pac_kicker_max_candidates::Int = 96

    # Non-greedy additive PAC search controls.  The old code tried candidates
    # in saved-path order and accepted the first confirmed improvement.  This
    # version evaluates a batch, confirms the best small-run candidates, then
    # commits the best confirmed move.
    pac_candidate_sets_per_step::Int = 128
    pac_full_confirm_top_candidates::Int = 12
    pac_kicker_single_pool::Int = 32
    pac_small_screen_slack_events::Float64 = 2.0
    pac_confirm_best_even_if_screen_fails::Bool = true

    # If a greedy step stalls, optionally take the best confirmed near-neutral
    # move and continue.  The best Tmat seen over the whole walk is tracked and
    # returned, so exploratory moves cannot damage the final output.
    pac_explore_when_stalled::Bool = true
    pac_max_explore_steps::Int = 4
    pac_explore_max_rel_worse::Float64 = 0.03
    pac_explore_max_abs_worse::Float64 = 0.0
    pac_explore_max_rel_above_best::Float64 = 0.10
    pac_explore_max_abs_above_best::Float64 = 0.0
    pac_return_best_seen::Bool = true

    # If true, score edges with only the actually-dangerous (D <= 0) samples.
    # Setting to false widens the support to all SCL list paths above all-zero,
    # which can help when the dangerous set is very small.
    pac_only_dangerous::Bool = true

    # ── pac_edge_ranking ──
    # How single-edge candidates are ranked before being fed into the kicker.
    #   :saved_paths  — legacy.  Rank every (i, j) pair globally by saved-paths
    #                   count (largest first).  Joint (source, target) ranking.
    #                   Tends to fill the candidate pool with edges from a
    #                   handful of high-σ sources that each have multiple
    #                   decent targets.
    #   :source_first — pick the best target *per source*.  For each info bit
    #                   i with σ_source_i > 0, find argmax_j saved[i, j] and
    #                   record that pair.  Rank pairs by σ_source descending
    #                   (so each iteration's top candidate is "the most-firing
    #                   info bit's best amplifier"), tie-break by saved-paths.
    #                   Naturally diversifies sources so the pool contains one
    #                   edge per high-σ info bit rather than many edges from
    #                   the same bit.  Cleaner story per edge, and avoids
    #                   over-committing capacity to a single source.
    #
    # pac_source_first_min_saved is the safety floor: a per-source pick is
    # only emitted if its saved-paths count meets this minimum.  Prevents the
    # algorithm from spending an edge on a high-σ source that happens to have
    # no good amplifier in span.
    pac_edge_ranking::Symbol = :saved_paths
    pac_source_first_min_saved::Int = 1

    # ── pac_edge_metric ──
    #   :trials — credit an edge (or edge set) only for TRIALS in which every
    #             dangerous path clears the gap after the counterfactual
    #             Delta.  Aligns the screening score with FER: when several
    #             wrong paths beat l0 in the same trial, lifting one of them
    #             does not save the trial.
    #   :paths  — legacy per-path crediting (overstates impact exactly in the
    #             multi-path trials, which dominate late in the search).
    pac_edge_metric::Symbol = :trials

    # L0-aware scoring for PAC tap additions.  A dangerous path with L0 still in
    # the final SCL list is a normal tail-ordering failure: a helpful frozen bit
    # can occur anywhere in the allowed span.  A dangerous path with L0 absent
    # is an early-dropout failure: a tap is credited only if its target frozen
    # bit occurs before first_error + pac_l0_lost_window.
    pac_l0_aware_scoring::Bool = true
    pac_l0_lost_window::Int = 32
    pac_l0_kept_weight::Float64 = 1.0
    pac_l0_lost_weight::Float64 = 1.0
    pac_l0_late_penalty::Float64 = 0.0
    # Initial T for stage 2.  :identity is the principled "from scratch"
    # choice; :generic starts from the generic PAC; :custom loads/uses a
    # caller-provided Tmat.
    pac_initial::Symbol = :identity
    pac_custom_Tmat::Any = nothing
    pac_custom_Tmat_path::Union{Nothing,String} = nothing

    # ── Stage 2 design mode ──
    #   :edges  — legacy per-edge additive search (run_iterative_pac_design).
    #   :parity — per-target COLUMN design (run_iterative_pac_design_parity).
    #             For a frozen target j, the precoder column S_j forces
    #             u_j = XOR_{i in S_j} v_i on every path, so whether a wrong
    #             path pays |LLR_j| depends on the PARITY of its error
    #             pattern restricted to S_j.  Adding one source at a time is
    #             blind to this structure: each new source XOR-flips coverage
    #             for every path that contains it (the "turning off" /
    #             even-parity evasion seen when many edges pile onto one
    #             target).  The parity search instead optimizes the whole
    #             source set of one column by local flips against a
    #             trial-level saved count, then commits one column per
    #             confirmed step.  Works in both directions from the same
    #             machinery: pac_initial = :identity grows columns from
    #             empty; pac_initial = :generic seeds each column search from
    #             the generic taps and lets flips remove as well as add.
    pac_design::Symbol = :edges

    # ── Replay screening (used by :parity mode; harmless otherwise) ──
    # Keep the channel LLR vectors of every trial whose final-list margin is
    # <= pac_replay_delta (all current errors + near-misses) and re-run the
    # actual SCL decoder on just those trials for each candidate column.
    # Exact under the modified T (wrong-path re-optimization included),
    # paired across candidates (same noise), and the only cheap screen that
    # can see a candidate BREAKING a currently-safe near-miss trial — which
    # the dangerous-sample counterfactual cannot, since it never observes
    # safe trials.  Pruning decisions in particular are replay-driven: the
    # dangerous-trial surrogate can never favor removing a tap (removals only
    # delete penalties wrong paths are paying), so "is this tap dead weight?"
    # is answerable only on the near-miss population.
    pac_replay_screen::Bool = true
    pac_replay_delta::Float64 = 10.0

    # ── :parity mode knobs ──
    pac_parity_max_steps::Int = 32          # committed column redesigns
    pac_parity_source_pool::Int = 48        # candidate sources per target (capped at 64)
    pac_parity_max_flip_rounds::Int = 64    # local-search rounds per column
    pac_parity_restarts::Int = 8            # random restarts of the flip search
    pac_parity_targets_screened::Int = 24   # columns local-searched per step (by upper bound)
    pac_parity_candidates_per_step::Int = 8 # best columns replay-screened per step
    pac_parity_full_confirm::Int = 3        # columns fresh-MC confirmed per step
    pac_parity_min_trial_gain::Float64 = 2.0 # surrogate floor to consider a column at all
    pac_parity_threshold::Float64 = 0.0     # D + Delta must exceed this to count as saved
    # Sparsity controls for the column search.  The unconstrained search will
    # happily buy density (every extra tap is free), which defeats a
    # sparse-precoder story.  max_col_weight caps the number of designed
    # taps per column (0 = unlimited); tap_penalty subtracts a per-tap cost
    # from the search objective so a tap must save at least that many
    # (weighted) trials to be kept.  Independent of these, taps whose removal
    # costs nothing are always stripped after the search (the flip search
    # only takes strict improvements, so plateau taps survive it otherwise).
    pac_parity_max_col_weight::Int = 0
    pac_parity_tap_penalty::Float64 = 0.0

    # Bulk prune (the :generic entry point).  Before the redesign loop, walk
    # the nontrivial columns once, clear each, and keep every clear the
    # replay buffer says is free (error count does not increase).  One full
    # MC confirmation at the end of the pass, reverted if it regressed by
    # more than pac_parity_bulk_prune_rel_slack.  This is what makes pruning
    # cost one pass rather than one full-budget evaluation per removed tap.
    pac_parity_bulk_prune::Bool = false
    pac_parity_bulk_prune_slack::Int = 0        # allowed replay-error increase per clear
    pac_parity_bulk_prune_rel_slack::Float64 = 0.02

    n_trials_eval::Int = 200_000

    # ─────────────────── Stage 3 (optional joint refinement) ───────────────────
    # After Stage 1 (info-set design with generic T) and Stage 2 (precoder
    # construction with fixed A), an optional Stage 3 takes the post-Stage 2
    # (A, T) and proposes joint (swap + edge) moves: remove info bit j, add
    # info bit i, and simultaneously add one or more PAC edges sourced from
    # the new info bit i.  This catches synergies that neither stage can see
    # alone — "bit i would make a great info bit BUT only if it also gets
    # edge (i, k)."  Sequential Stage 1→Stage 2 cannot find these because
    # Stage 1 evaluates the promotion of i against Tgeneric (which doesn't
    # have the edge yet) and Stage 2 evaluates edges against a fixed A
    # (which doesn't yet include i).  The joint move proposes both at once.
    #
    # Stage 3 is OFF by default to keep the legacy two-stage pipeline
    # bit-identical when the knob isn't set.  Enable with enable_joint_stage3.
    enable_joint_stage3::Bool = false
    joint_iterations::Int = 5
    joint_max_remove_candidates::Int  = 6
    joint_max_promote_candidates::Int = 8
    joint_max_amplifiers_per_source::Int = 4
    # Multi-edge variants: 1 = swap + one edge; 2 = swap + up to two edges.
    joint_max_edges_per_move::Int = 1
    # Whether to also propose bare (swap, no edge) candidates.  Setting to
    # false forces every Stage 3 move to add at least one edge — useful when
    # you want to investigate purely-coupled moves.
    joint_allow_bare_swap::Bool = true
    # Whether to propose pure edge additions/removals (no swap).  Closes
    # the loop on existing info bits whose best amplifier the Stage 2
    # search missed.
    joint_allow_edge_only::Bool = true
    joint_max_total_candidates::Int = 96
    joint_full_eval_top_k::Int = 8
    joint_n_trials_small::Int = 50_000
    joint_n_trials_full::Int  = 200_000
    joint_min_improvement::Float64 = 0.0

    # Marginal-composite ranking weights for Stage 3.  The composite is
    #   score = w_R · norm(remove_score) + w_A · norm(add_score) + w_E · norm(edge_amp_sum)
    # where each component is the same metric Stage 1 / Stage 2 already use
    # (so `add_strategy = :q10_llr_dang` propagates into Stage 3 unchanged,
    # `freeze_use_rho` likewise, etc.).  Components are normalized to [0, 1]
    # within the candidate pool so the weights are scale-free.  Setting any
    # weight to 0 disables that component's influence.
    joint_w_remove::Float64 = 1.0
    joint_w_add::Float64    = 1.0
    joint_w_edge::Float64   = 1.0

    modes::Tuple = (:explicit,)
    skip_all_positive::Bool = true
    eta_grid = range(0.0, 5.0, length = 51)
    alphas::Tuple = (0.01, 0.05, 0.10)
    seed::Int = 7
    eval_seed::Int = 11

    outdir::String = joinpath(SCRIPT_DIR, "pac_sparse_pipeline_out")
    save_matrices::Bool = true
    verbose::Bool = true
end

# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

function noise_sigma_from_ebn0db(N::Integer, K::Integer, EbN0_dB::Real)
    0 < K <= N || throw(ArgumentError("K must satisfy 0 < K <= N."))
    R = Float64(K) / Float64(N)
    ebn0 = 10.0^(Float64(EbN0_dB) / 10.0)
    return sqrt(1.0 / (2.0 * R * ebn0))
end

function llr_mean_from_ebn0db(N::Integer, K::Integer, EbN0_dB::Real)
    sigma = noise_sigma_from_ebn0db(N, K, EbN0_dB)
    return 2.0 / (sigma * sigma)
end

function identity_tmat(N::Int)
    T = zeros(UInt8, N, N)
    @inbounds for j in 1:N
        T[j, j] = 0x01
    end
    return T
end

function pac_tmat_from_taps(N::Int, taps::AbstractVector{<:Integer})
    isempty(taps) && throw(ArgumentError("taps must include the feed-through tap."))
    isodd(Int(taps[1])) || throw(ArgumentError("taps[1] must be 1 for an invertible causal transform."))
    T = identity_tmat(N)
    m = length(taps) - 1
    @inbounds for d in 1:m
        if isodd(Int(taps[d + 1]))
            for i in 1:(N - d)
                T[i, i + d] = 0x01
            end
        end
    end
    return T
end

# Validate that an arbitrary integer-valued matrix is shaped like a PAC
# precoder Tmat (upper-triangular over GF(2) with 1s on the diagonal) and
# convert it to UInt8.  Reduces all entries mod 2 along the way so the
# caller can pass Int or Bool matrices without worrying about element type.
function _validate_and_normalize_tmat(T_in::AbstractMatrix, N::Int; label::String = "Tmat")
    size(T_in) == (N, N) ||
        throw(ArgumentError("$label must be cfg.N × cfg.N = $N × $N (got $(size(T_in)))."))
    T = Matrix{UInt8}(undef, N, N)
    @inbounds for i in 1:N, j in 1:N
        T[i, j] = UInt8(mod(Int(T_in[i, j]), 2))
    end
    @inbounds for j in 1:N
        T[j, j] == 0x01 ||
            throw(ArgumentError("$label must have 1 on the diagonal; position $j is 0."))
        for i in (j + 1):N
            T[i, j] == 0x00 ||
                throw(ArgumentError("$label must be upper-triangular; entry [$i, $j] is nonzero."))
        end
    end
    return T
end

# Build the Stage-1 precoder Tmat from the configuration.  Priority:
#   1. cfg.set_custom_Tmat  (direct matrix)
#   2. cfg.set_custom_Tmat_path  (CSV path)
#   3. cfg.generic_taps  (convolutional taps — legacy default)
function build_stage1_Tmat(cfg::PipelineConfig)
    if cfg.set_custom_Tmat !== nothing
        T = _validate_and_normalize_tmat(cfg.set_custom_Tmat, cfg.N;
                                          label = "cfg.set_custom_Tmat")
        cfg.verbose &&
            println("Stage-1 Tmat: cfg.set_custom_Tmat (",
                    count(==(0x01), T) - cfg.N, " off-diagonal entries)")
        return T
    elseif cfg.set_custom_Tmat_path !== nothing
        path = cfg.set_custom_Tmat_path
        isfile(path) ||
            throw(ArgumentError("cfg.set_custom_Tmat_path file not found: $path"))
        T_raw = readdlm(path, ',', Int)
        T = _validate_and_normalize_tmat(T_raw, cfg.N;
                                          label = "cfg.set_custom_Tmat_path ($path)")
        cfg.verbose &&
            println("Stage-1 Tmat: loaded from CSV at ", path, "  (",
                    count(==(0x01), T) - cfg.N, " off-diagonal entries)")
        return T
    else
        cfg.verbose && println("Stage-1 Tmat: generic PAC from taps = ", cfg.generic_taps)
        return pac_tmat_from_taps(cfg.N, cfg.generic_taps)
    end
end

function rm_reliability(N::Int)
    return Float64[count_ones(UInt(i - 1)) + 1e-6 * i for i in 1:N]
end

function dega_reliability(N::Int, K::Int, EbN0_dB::Real; Q::Int = 24)
    m = llr_mean_from_ebn0db(N, K, EbN0_dB)
    rule = Main.TailAwarePM.GHRule(Q)
    dega0 = Main.TailAwarePM.conditional_dega_path(zeros(Int, N), m; rule = rule)

    rel = zeros(Float64, N)
    @inbounds for i in 1:N
        sig = Float64(dega0.sigma[i])
        mu = Float64(dega0.mu[i])
        if sig > 0.0
            rel[i] = mu / sig
        elseif mu > 0.0
            rel[i] = Inf
        elseif mu < 0.0
            rel[i] = -Inf
        else
            rel[i] = 0.0
        end
    end
    return rel
end

function reliability_vector(cfg::PipelineConfig)
    if cfg.reliability_method == :dega
        return dega_reliability(cfg.N, cfg.K, cfg.EbN0_dB; Q = cfg.reliability_Q)
    elseif cfg.reliability_method == :rm
        return rm_reliability(cfg.N)
    else
        throw(ArgumentError("Unknown reliability_method $(cfg.reliability_method)."))
    end
end

function initial_information_set(cfg::PipelineConfig, reliability::AbstractVector{<:Real})
    if cfg.initial_A !== nothing
        A = sort(unique(Int.(cfg.initial_A)))
        length(A) == cfg.K || throw(ArgumentError("initial_A must have length K."))
        all(i -> 1 <= i <= cfg.N, A) || throw(ArgumentError("initial_A out of range."))
        return A
    end
    init_rel = if cfg.initial_reliability_method == cfg.reliability_method
        reliability
    elseif cfg.initial_reliability_method == :rm
        rm_reliability(cfg.N)
    elseif cfg.initial_reliability_method == :dega
        dega_reliability(cfg.N, cfg.K, cfg.EbN0_dB; Q = cfg.reliability_Q)
    else
        throw(ArgumentError("Unknown initial_reliability_method $(cfg.initial_reliability_method)."))
    end
    idx = sortperm(Float64.(init_rel), rev = true)
    return sort(idx[1:cfg.K])
end

function spectrum_for_A(cfg::PipelineConfig,
                        A::AbstractVector{<:Integer},
                        Tmat::AbstractMatrix{<:Integer};
                        n_trials::Int,
                        seed::Int,
                        label::String = "",
                        collect_replay::Bool = false)
    sigma = noise_sigma_from_ebn0db(cfg.N, cfg.K, cfg.EbN0_dB)
    depths = DPS.infer_depths(Tmat)
    cfg.verbose && !isempty(label) && println("\n--- Spectrum: ", label, " ---")
    cfg.verbose && @printf("A length = %d, trials = %d, sigma = %.6g\n",
                            length(A), n_trials, sigma)

    return DPS.dangerous_path_spectrum(
        cfg.N, cfg.K, sort(Int.(A)), sigma, n_trials, cfg.L, Tmat;
        depths = depths,
        modes = cfg.modes,
        skip_all_positive = cfg.skip_all_positive,
        eta_grid = cfg.eta_grid,
        alphas = cfg.alphas,
        seed = seed,
        collect_replay = collect_replay,
        replay_delta = cfg.pac_replay_delta,
    )
end

function tail_score(res, cfg::PipelineConfig)
    return DPS.tail_risk_objective(res; mode = :explicit, combine = cfg.set_objective)
end


function _safe_property(x, name::Symbol, default)
    try
        return hasproperty(x, name) ? getproperty(x, name) : default
    catch
        return default
    end
end

function _safe_vector_lookup(x, names::Tuple, i::Int, default::Float64 = 0.0)
    for name in names
        v = _safe_property(x, name, nothing)
        v === nothing && continue
        try
            if i <= length(v)
                return Float64(v[i])
            end
        catch
        end
    end
    return default
end


function _first_existing_vector(s, names::Tuple)
    for nm in names
        if hasproperty(s, nm)
            v = getproperty(s, nm)
            v === nothing && continue
            return v
        end
    end
    return nothing
end

@inline function _bit01(v, j::Int)
    return Int(v[j]) != 0 ? 1 : 0
end

function _stage1_n_trials(res, fallback::Int)
    return Int(_safe_property(res, :n_trials,
               _safe_property(res, :trials,
               _safe_property(res, :num_trials, max(1, fallback)))))
end

function _stage1_events(res)
    ev = _safe_property(res, :events, nothing)
    if ev !== nothing && !isempty(ev)
        return ev
    end
    try
        return res.samples_by_mode[:explicit]
    catch
        return []
    end
end

# For source/post support we must use the explicit path samples, not the
# compressed first-error event summaries.  The summaries carry fields such as
# first_error and D, but they generally do not carry the full pre/post path
# vectors.  Using them makes source_sigma and post_sigma appear as zero even
# when the pre/post diagnostic report shows large support.
function _stage1_explicit_samples(res)
    try
        return res.samples_by_mode[:explicit]
    catch
        return _stage1_events(res)
    end
end

function _stage1_support_vectors(res, N::Int, n_trials::Int)
    source_count = zeros(Int, N)
    post_count = zeros(Int, N)
    mismatch_count = zeros(Int, N)
    # both_count[j] = #(v_j = 1 AND u_j = 1) over considered dangerous paths.
    # This is the numerator of rho_j = P(u_j = 1 | v_j = 1, dangerous).
    both_count = zeros(Int, N)
    # sigma_llr accumulators.  These build the per-bit "expected freeze PM
    # penalty" on dangerous trials:
    #   source_abs_llr_sum[j] = Σ_t  (source bit at j on trial t) · |LLR_j^t|
    #   post_abs_llr_sum[j]   = Σ_t  (post   bit at j on trial t) · |LLR_j^t|
    # Normalizing by n_trials gives source_sigma_llr / post_sigma_llr below.
    # These weight high-σ-low-LLR positions DOWN (lots of v=1 but with weak
    # LLR magnitude → freezing them only changes the BM by a small amount)
    # and high-σ-high-LLR positions UP (lots of v=1 with strong LLR magnitude
    # → freezing actually inflicts real PM on the wrong path).
    source_abs_llr_sum = zeros(Float64, N)
    post_abs_llr_sum   = zeros(Float64, N)
    n_llr_samples      = 0
    considered = 0
    skipped_missing = 0

    for s in _stage1_explicit_samples(res)
        if hasproperty(s, :mode) && getproperty(s, :mode) !== :explicit
            continue
        end
        if hasproperty(s, :D) && Float64(getproperty(s, :D)) > 0.0
            continue
        end

        source = _first_existing_vector(s, (:path_u_scl, :path_u, :u, :U, :path_v_scl))
        post   = _first_existing_vector(s, (:B, :path_x_scl, :path_x, :x, :X, :post_v_scl))
        llr_vec = _first_existing_vector(s, (:LLR, :llr, :path_LLR, :path_llr))

        if source === nothing || post === nothing
            skipped_missing += 1
            continue
        end

        considered += 1
        has_llr = llr_vec !== nothing
        has_llr && (n_llr_samples += 1)
        @inbounds for j in 1:N
            us = _bit01(source, j)
            vp = _bit01(post, j)
            source_count[j] += us
            post_count[j] += vp
            mismatch_count[j] += (us != vp) ? 1 : 0
            both_count[j] += (us == 1 && vp == 1) ? 1 : 0
            if has_llr && j <= length(llr_vec)
                al = abs(Float64(llr_vec[j]))
                source_abs_llr_sum[j] += us * al
                post_abs_llr_sum[j]   += vp * al
            end
        end
    end

    denom_trials = max(1, n_trials)
    denom_paths = max(1, considered)
    source_sigma = Float64[source_count[j] / denom_trials for j in 1:N]
    post_sigma   = Float64[post_count[j]   / denom_trials for j in 1:N]
    mismatch_sigma = Float64[mismatch_count[j] / denom_trials for j in 1:N]
    both_sigma   = Float64[both_count[j]   / denom_trials for j in 1:N]
    source_frac = Float64[source_count[j] / denom_paths for j in 1:N]
    post_frac   = Float64[post_count[j]   / denom_paths for j in 1:N]
    mismatch_frac = Float64[mismatch_count[j] / denom_paths for j in 1:N]

    # sigma_llr per bit, in source and post domains.
    # source_sigma_llr_j ≈ (1/n_trials) · Σ_t (u_j^t = 1) · |LLR_j^t|
    # post_sigma_llr_j   ≈ (1/n_trials) · Σ_t (v_j^t = 1) · |LLR_j^t|
    source_sigma_llr = Float64[source_abs_llr_sum[j] / denom_trials for j in 1:N]
    post_sigma_llr   = Float64[post_abs_llr_sum[j]   / denom_trials for j in 1:N]
    # Conditional mean |LLR| given the support bit was 1 (informative diagnostic
    # — high σ × low conditional |LLR| explains a useless freeze candidate).
    source_cond_abs_llr = zeros(Float64, N)
    post_cond_abs_llr   = zeros(Float64, N)
    @inbounds for j in 1:N
        if source_count[j] > 0
            source_cond_abs_llr[j] = source_abs_llr_sum[j] / source_count[j]
        end
        if post_count[j] > 0
            post_cond_abs_llr[j] = post_abs_llr_sum[j] / post_count[j]
        end
    end

    # rho_j is the conditional propagation rate, with safe division.
    # rho_j = both_count_j / source_count_j when source_count_j > 0, else 0.
    rho = zeros(Float64, N)
    @inbounds for j in 1:N
        if source_count[j] > 0
            rho[j] = both_count[j] / source_count[j]
        end
    end

    return (source_sigma = source_sigma,
            post_sigma = post_sigma,
            mismatch_sigma = mismatch_sigma,
            both_sigma = both_sigma,
            source_sigma_llr = source_sigma_llr,
            post_sigma_llr = post_sigma_llr,
            source_cond_abs_llr = source_cond_abs_llr,
            post_cond_abs_llr = post_cond_abs_llr,
            source_frac = source_frac,
            post_frac = post_frac,
            mismatch_frac = mismatch_frac,
            source_count = source_count,
            post_count = post_count,
            mismatch_count = mismatch_count,
            both_count = both_count,
            rho = rho,
            n_paths = considered,
            n_llr_samples = n_llr_samples,
            skipped_missing = skipped_missing)
end

function _stage1_chosen_sigma_vectors(res, cfg::PipelineConfig)
    n_trials = _stage1_n_trials(res, cfg.n_trials_set)
    support = _stage1_support_vectors(res, cfg.N, n_trials)

    chosen = zeros(Float64, cfg.N)
    chosen_sigma_llr = zeros(Float64, cfg.N)
    if cfg.set_sigma_domain in (:source, :pre, :u)
        chosen .= support.source_sigma
        chosen_sigma_llr .= support.source_sigma_llr
    elseif cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel)
        chosen .= support.post_sigma
        chosen_sigma_llr .= support.post_sigma_llr
    elseif cfg.set_sigma_domain == :hybrid
        chosen .= cfg.set_sigma_pre_weight .* support.source_sigma .+
                  cfg.set_sigma_post_weight .* support.post_sigma
        chosen_sigma_llr .= cfg.set_sigma_pre_weight .* support.source_sigma_llr .+
                            cfg.set_sigma_post_weight .* support.post_sigma_llr
    else
        throw(ArgumentError("Unknown set_sigma_domain $(cfg.set_sigma_domain). Use :source, :post, or :hybrid."))
    end

    return (chosen = chosen,
            chosen_sigma_llr = chosen_sigma_llr,
            source = support.source_sigma,
            post = support.post_sigma,
            source_sigma_llr = support.source_sigma_llr,
            post_sigma_llr = support.post_sigma_llr,
            source_cond_abs_llr = support.source_cond_abs_llr,
            post_cond_abs_llr = support.post_cond_abs_llr,
            mismatch = support.mismatch_sigma,
            both = support.both_sigma,
            rho = support.rho,
            support = support,
            n_trials = n_trials)
end

# ------------------------------------------------------------
# Empirical LLR statistics conditioned on dangerous trials.
#
# For each frozen bit j we compute Zhat_j^dang = E[exp(-|LLR_j|/2)] over the
# dangerous samples, plus a few diagnostic statistics.  Smaller Zhat_j^dang
# means the bit-channel at j is reliably confident on the trials where the
# current code actually fails, so promoting j to info is unlikely to open a
# new first-error mode.  This is the empirical replacement for DEGA on the
# add side: DEGA averages over the whole channel distribution (including
# easy trials), while this metric conditions on the failure tail.
#
# The LLR is the SCL path-conditioned bit-channel LLR at step j.  For a
# frozen bit the correct u_j is 0, so we count "sign errors" as LLR_j < 0
# (the decoder being pulled toward u_j = 1 against the frozen-zero truth).
# ------------------------------------------------------------
function _stage1_llr_dang_stats(res, N::Int)
    samples = _stage1_explicit_samples(res)
    abs_llr_buckets = [Float64[] for _ in 1:N]
    zhat_sum = zeros(Float64, N)
    sign_err = zeros(Int, N)
    n_paths = 0

    for s in samples
        if hasproperty(s, :mode) && getproperty(s, :mode) !== :explicit
            continue
        end
        if hasproperty(s, :D) && Float64(getproperty(s, :D)) > 0.0
            continue
        end
        llr_vec = _first_existing_vector(s, (:LLR, :llr, :path_LLR, :path_llr))
        llr_vec === nothing && continue
        n_paths += 1
        @inbounds for j in 1:min(N, length(llr_vec))
            l = Float64(llr_vec[j])
            al = abs(l)
            push!(abs_llr_buckets[j], al)
            zhat_sum[j] += exp(-0.5 * al)
            if l < 0.0
                sign_err[j] += 1
            end
        end
    end

    denom = max(1, n_paths)
    zhat   = Float64[zhat_sum[j] / denom for j in 1:N]
    mean_abs_llr  = zeros(Float64, N)
    q10_abs_llr   = zeros(Float64, N)
    median_abs_llr = zeros(Float64, N)
    sign_err_rate = Float64[sign_err[j] / denom for j in 1:N]

    @inbounds for j in 1:N
        bucket = abs_llr_buckets[j]
        if !isempty(bucket)
            mean_abs_llr[j]  = mean(bucket)
            median_abs_llr[j] = median(bucket)
            q10_abs_llr[j]    = quantile(bucket, 0.1)
        end
    end

    return (zhat = zhat,
            mean_abs_llr = mean_abs_llr,
            median_abs_llr = median_abs_llr,
            q10_abs_llr = q10_abs_llr,
            sign_err_rate = sign_err_rate,
            n_paths = n_paths)
end

function _stage1_lambda_and_gap_vectors(res, N::Int, n_trials::Int)
    counts = zeros(Int, N)
    D_lists = [Float64[] for _ in 1:N]

    for s in _stage1_events(res)
        hasproperty(s, :first_error) || continue
        i = Int(getproperty(s, :first_error))
        1 <= i <= N || continue
        D = hasproperty(s, :D) ? Float64(getproperty(s, :D)) : 0.0
        counts[i] += 1
        push!(D_lists[i], D)
    end

    lambda = Float64[counts[i] / max(1, n_trials) for i in 1:N]
    Dmean = zeros(Float64, N)
    Dstd = zeros(Float64, N)
    q05 = zeros(Float64, N)
    p_le0 = zeros(Float64, N)
    for i in 1:N
        Ds = D_lists[i]
        if isempty(Ds)
            continue
        end
        Dmean[i] = mean(Ds)
        Dstd[i] = length(Ds) <= 1 ? 0.0 : std(Ds; corrected = false)
        q05[i] = quantile(Ds, 0.05)
        p_le0[i] = Base.count(D -> D <= 0.0, Ds) / length(Ds)
    end

    return (count = counts, lambda = lambda, Dmean = Dmean,
            Dstd = Dstd, q05 = q05, p_le0 = p_le0)
end

function _stage1_priority_value(lambda_i::Float64,
                                sigma_i::Float64,
                                Dmean::Float64,
                                Dstd::Float64,
                                q05::Float64,
                                p_le0::Float64,
                                cfg::PipelineConfig;
                                sigma_llr_i::Float64 = 0.0)
    score_mode = cfg.set_risk_score

    if score_mode in (:lambda, :lambda_true, :lambda_only)
        return lambda_i
    elseif score_mode in (:sigma, :support, :sigma_only)
        return sigma_i
    elseif score_mode in (:sigma_llr, :sigma_x_llr, :sigma_times_llr, :sllr)
        # Per-trial expected freeze PM penalty:
        #   sigma_llr_i = (1/n_trials) * Σ_t (path bit at i on trial t = 1) * |LLR_i^t|
        # This is sigma_i weighted by the magnitude of the LLR being suppressed,
        # so high-support-but-weak-LLR bits get appropriately deprioritized.
        return sigma_llr_i
    elseif score_mode in (:std, :Dstd)
        return Dstd
    elseif score_mode in (:mean, :Dmean)
        return -Dmean
    elseif score_mode in (:q05, :tail, :quantile)
        return -q05
    elseif score_mode in (:prob, :p_le0, :ple0)
        return p_le0
    else
        lambda_weight = Float64(_safe_property(cfg.risk_mix, :lambda, 1.0))
        sigma_weight  = Float64(_safe_property(cfg.risk_mix, :sigma,  1.0))
        return lambda_weight * lambda_i + sigma_weight * sigma_i
    end
end

function _stage1_priority_vectors(res, cfg::PipelineConfig)
    sigma_pack = _stage1_chosen_sigma_vectors(res, cfg)
    gaps = _stage1_lambda_and_gap_vectors(res, cfg.N, sigma_pack.n_trials)
    llr_stats = _stage1_llr_dang_stats(res, cfg.N)
    priority = zeros(Float64, cfg.N)
    @inbounds for i in 1:cfg.N
        priority[i] = _stage1_priority_value(gaps.lambda[i], sigma_pack.chosen[i],
                                             gaps.Dmean[i], gaps.Dstd[i],
                                             gaps.q05[i], gaps.p_le0[i], cfg;
                                             sigma_llr_i = sigma_pack.chosen_sigma_llr[i])
    end

    # rho-weighted removal priority.  Off by default; opt-in via freeze_use_rho.
    #
    # The freeze action is a v-side constraint: forcing v_i = 0 eliminates
    # exactly the dangerous paths with v_i = 1, so the "kill-able mass" is
    # source_sigma_i = P(v_i = 1, dangerous).  rho_i = P(u_i = 1 | v_i = 1)
    # is then the rate at which those kills actually disrupt the BM at step i.
    # Their product is both_sigma_i = P(v_i = 1 AND u_i = 1, dangerous), the
    # rate of "freezable AND BM-relevant" mass.  We use source_sigma as the
    # multiplier base regardless of set_sigma_domain, because post_sigma * rho
    # has no clean probabilistic meaning (it double-counts u_i = 1 events that
    # come from v_<i mixing rather than from v_i itself).  set_sigma_domain
    # still controls the displayed "sigma" column in reports.
    priority_rho = copy(priority)
    if cfg.freeze_use_rho
        floor = cfg.freeze_rho_floor
        if _using_sigma_llr(cfg)
            # When the active score is sigma_llr, the rho weighting uses the
            # source-domain sigma_llr as the multiplier base.  This gives
            #   source_sigma_llr_i * rho_i ≈ (1/n_trials) Σ_t (v_i=1)·|LLR_i^t|
            #                                · P(u_i = 1 | v_i = 1, dangerous)
            # i.e. the v-side freeze-PM mass discounted by the propagation rate
            # to a u-side BM event.  Stays consistent with the source-based
            # justification of rho weighting (freezing acts on v).
            @inbounds for i in 1:cfg.N
                weight = max(sigma_pack.rho[i], floor)
                priority_rho[i] = sigma_pack.source_sigma_llr[i] * weight
            end
        else
            @inbounds for i in 1:cfg.N
                weight = max(sigma_pack.rho[i], floor)
                priority_rho[i] = sigma_pack.source[i] * weight
            end
        end
    end

    return (priority = priority,
            priority_rho = priority_rho,
            lambda = gaps.lambda,
            count = gaps.count,
            Dmean = gaps.Dmean,
            Dstd = gaps.Dstd,
            q05 = gaps.q05,
            p_le0 = gaps.p_le0,
            sigma = sigma_pack.chosen,
            sigma_llr = sigma_pack.chosen_sigma_llr,
            source_sigma = sigma_pack.source,
            post_sigma = sigma_pack.post,
            source_sigma_llr = sigma_pack.source_sigma_llr,
            post_sigma_llr = sigma_pack.post_sigma_llr,
            source_cond_abs_llr = sigma_pack.source_cond_abs_llr,
            post_cond_abs_llr = sigma_pack.post_cond_abs_llr,
            mismatch_sigma = sigma_pack.mismatch,
            both_sigma = sigma_pack.both,
            rho = sigma_pack.rho,
            zhat_dang = llr_stats.zhat,
            mean_abs_llr_dang = llr_stats.mean_abs_llr,
            median_abs_llr_dang = llr_stats.median_abs_llr,
            q10_abs_llr_dang = llr_stats.q10_abs_llr,
            sign_err_rate_dang = llr_stats.sign_err_rate,
            n_paths_llr = llr_stats.n_paths,
            support = sigma_pack.support,
            n_trials = sigma_pack.n_trials)
end


function _stage1_lambda_weight(cfg::PipelineConfig)
    if cfg.set_risk_score in (:lambda, :lambda_true, :lambda_only)
        return 1.0
    elseif cfg.set_risk_score in (:sigma, :support, :sigma_only,
                                  :sigma_llr, :sigma_x_llr, :sigma_times_llr, :sllr,
                                  :std, :Dstd, :mean, :Dmean,
                                  :q05, :tail, :quantile,
                                  :prob, :p_le0, :ple0)
        return 0.0
    else
        return Float64(_safe_property(cfg.risk_mix, :lambda, 1.0))
    end
end

# Helper: is the active sigma-style score the sigma_llr family?
@inline function _using_sigma_llr(cfg::PipelineConfig)
    return cfg.set_risk_score in (:sigma_llr, :sigma_x_llr, :sigma_times_llr, :sllr)
end

function _stage1_report_sort_key(stats, cfg::PipelineConfig, i::Int)
    lambda_w = _stage1_lambda_weight(cfg)
    active_priority = cfg.freeze_use_rho ? stats.priority_rho[i] : stats.priority[i]
    rho_term = cfg.freeze_use_rho ? -stats.rho[i] : 0.0
    if _using_sigma_llr(cfg)
        # When sigma_llr drives the score, sort by it directly so the report
        # ordering matches the candidate-generator ordering, and surface the
        # raw sigma values as tie-breakers/diagnostics.
        return (-stats.sigma_llr[i], rho_term, -stats.sigma[i],
                -active_priority, i)
    elseif lambda_w == 0.0 && (cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel))
        return (-stats.post_sigma[i], rho_term, -active_priority,
                -stats.source_sigma[i], i)
    elseif lambda_w == 0.0
        return (-active_priority, rho_term, -stats.sigma[i], i)
    else
        return (-active_priority, rho_term, -stats.lambda[i], i)
    end
end

function _stage1_remove_sort_key(stats, cfg::PipelineConfig, rel::Vector{Float64}, i::Int)
    lambda_w = _stage1_lambda_weight(cfg)
    # When rho-weighting is enabled, the active priority is priority_rho and
    # rho itself is exposed as the second-stage tie-breaker.  When it is off,
    # the keys collapse to the original behavior.
    active_priority = cfg.freeze_use_rho ? stats.priority_rho[i] : stats.priority[i]
    rho_term = cfg.freeze_use_rho ? -stats.rho[i] : 0.0

    if _using_sigma_llr(cfg)
        # sigma_llr (= sigma * E[|LLR| | bit was 1]) is the primary key for
        # remove selection.  This down-ranks high-σ-low-LLR bits whose freeze
        # would not actually inflict meaningful PM on the wrong path.
        return (-stats.sigma_llr[i], rho_term, -stats.sigma[i],
                -Float64(rel[i]), i)
    elseif lambda_w == 0.0 && (cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel))
        # Post-convolution sigma experiment: rank by post_sigma, with rho as a
        # tie-breaker when enabled.
        return (-stats.post_sigma[i], rho_term, -active_priority,
                -stats.source_sigma[i], i)
    elseif lambda_w == 0.0
        return (-active_priority, rho_term, -stats.sigma[i],
                -Float64(rel[i]), i)
    else
        return (-active_priority, rho_term, -stats.lambda[i],
                -Float64(rel[i]), i)
    end
end

function print_stage1_priority_report(res, cfg::PipelineConfig; max_rows::Int = 32)
    stats = _stage1_priority_vectors(res, cfg)
    active = findall(i -> stats.count[i] > 0 || stats.sigma[i] > 0.0 ||
                         stats.source_sigma[i] > 0.0 || stats.post_sigma[i] > 0.0,
                    1:cfg.N)
    if isempty(active)
        println("No dangerous-path events to summarize.")
        return
    end

    lambda_state = _safe_property(res, :lambda, nothing)
    rows = NamedTuple[]
    for i in active
        eta_star = lambda_state === nothing ? 0.0 :
                   _safe_vector_lookup(lambda_state,
                                       (:eta_star, :eta_opt, :eta_true, :eta, :eta_best),
                                       i, 0.0)
        push!(rows, (
            i = i,
            count = stats.count[i],
            priority = stats.priority[i],
            priority_rho = stats.priority_rho[i],
            lambda = stats.lambda[i],
            sigma = stats.sigma[i],
            sigma_llr = stats.sigma_llr[i],
            source_count = stats.support.source_count[i],
            post_count = stats.support.post_count[i],
            both_count = stats.support.both_count[i],
            mismatch_count = stats.support.mismatch_count[i],
            sigma_source = stats.source_sigma[i],
            sigma_post = stats.post_sigma[i],
            sigma_source_llr = stats.source_sigma_llr[i],
            sigma_post_llr = stats.post_sigma_llr[i],
            source_cond_abs_llr = stats.source_cond_abs_llr[i],
            post_cond_abs_llr = stats.post_cond_abs_llr[i],
            sigma_both = stats.both_sigma[i],
            sigma_mismatch = stats.mismatch_sigma[i],
            rho = stats.rho[i],
            zhat_dang = stats.zhat_dang[i],
            mean_abs_llr_dang = stats.mean_abs_llr_dang[i],
            q10_abs_llr_dang = stats.q10_abs_llr_dang[i],
            sign_err_rate_dang = stats.sign_err_rate_dang[i],
            Dmean = stats.Dmean[i],
            Dstd = stats.Dstd[i],
            q05 = stats.q05[i],
            p_le0 = stats.p_le0[i],
            eta_star = eta_star,
        ))
    end

    sort!(rows; by = r -> _stage1_report_sort_key(stats, cfg, r.i))

    lambda_weight = Float64(_safe_property(cfg.risk_mix, :lambda, 1.0))
    sigma_weight  = Float64(_safe_property(cfg.risk_mix, :sigma,  1.0))
    println("\nStage-1 risk report sorted by active priority")
    @printf("  set_risk_score = %s, risk_mix = (lambda = %.4g, sigma = %.4g), set_sigma_domain = %s\n",
            String(cfg.set_risk_score), lambda_weight, sigma_weight,
            String(cfg.set_sigma_domain))
    if cfg.set_sigma_domain == :hybrid
        @printf("  chosen sigma = %.4g * source_sigma + %.4g * post_sigma\n",
                cfg.set_sigma_pre_weight, cfg.set_sigma_post_weight)
    elseif cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel)
        println("  chosen sigma = post-convolution/channel-facing support")
        println("  removal priority uses post_sigma from explicit path samples; lambda is not used when lambda weight is zero")
    else
        println("  chosen sigma = source-domain/pre-convolution support")
    end
    if !(cfg.set_risk_score in (:lambda, :lambda_true, :lambda_only,
                                :sigma, :support, :sigma_only,
                                :sigma_llr, :sigma_x_llr, :sigma_times_llr, :sllr,
                                :std, :Dstd, :mean, :Dmean, :q05, :tail, :quantile,
                                :prob, :p_le0, :ple0))
        @printf("  priority = %.4g * lambda + %.4g * chosen_sigma\n",
                lambda_weight, sigma_weight)
    end
    if _using_sigma_llr(cfg)
        println("  active score = sigma_llr (= sigma * E[|LLR| | bit was 1])")
        println("  remove pool ranked by sigma_llr in the chosen domain ($(String(cfg.set_sigma_domain)))")
    end
    @printf("  support paths considered = %d, skipped_missing = %d, normalized by trials = %d\n",
            stats.support.n_paths, stats.support.skipped_missing, stats.n_trials)
    @printf("  freeze_use_rho = %s, freeze_rho_floor = %.4g, freeze_rho_min = %.4g\n",
            string(cfg.freeze_use_rho), cfg.freeze_rho_floor, cfg.freeze_rho_min)
    @printf("  add_use_zhat_dang = %s, add_zhat_weight = %.4g, add_zhat_floor = %.4g, n_paths_llr = %d\n",
            string(cfg.add_use_zhat_dang), cfg.add_zhat_weight,
            cfg.add_zhat_floor, stats.n_paths_llr)

    # Column header.  sigma columns and sigma·LLR columns are paired so that
    # the high-σ-low-LLR vs high-σ-high-LLR contrast is visible per bit.
    @printf("%6s %8s %8s %8s %8s %10s %10s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
            "i", "fe_cnt", "src_cnt", "post_cnt", "both",
            "rho", "Zhat_d",
            "priority", "lambda", "sigma",
            "src_sigma", "post_sigma",
            "src_sigLLR", "post_sigLLR",
            "|LLR|_d", "q10|LLR|")
    @printf("%6s %8s %8s %8s %8s %10s %10s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
            "------", "--------", "--------", "--------", "--------",
            "----------", "----------",
            "------------", "------------", "------------",
            "------------", "------------",
            "------------", "------------",
            "------------", "------------")
    for r in first(rows, min(max_rows, length(rows)))
        @printf("%6d %8d %8d %8d %8d %10.4f %10.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                r.i, r.count, r.source_count, r.post_count, r.both_count,
                r.rho, r.zhat_dang,
                r.priority, r.lambda, r.sigma,
                r.sigma_source, r.sigma_post,
                r.sigma_source_llr, r.sigma_post_llr,
                r.mean_abs_llr_dang, r.q10_abs_llr_dang)
    end
end

# ============================================================
# Stage-1 add-candidate report.
#
# Symmetric to print_stage1_priority_report, but iterates over the FROZEN
# bits (the add candidates) and surfaces add-relevant statistics:
#
#   rel(DEGA)       analytic reliability (μ / σ from DEGA).  Higher is more
#                   reliable on the channel marginal.
#   Zhat_d          (1/n_paths) · Σ exp(-|LLR_j|/2) over dangerous samples.
#                   Bhattacharyya-like reliability on the current failure tail.
#                   Lower is better — though see the bias discussion: this is
#                   conditioned on the *current* code's failures, not the
#                   post-add failures.
#   |LLR|_d         mean |LLR_j| on dangerous samples.
#   q10|LLR|_d      10th-percentile |LLR_j| on dangerous samples (robust
#                   to a few outlier-large LLRs).
#   sign_err_d      fraction of dangerous samples where LLR_j < 0 (the bit
#                   channel is sign-flipped against the frozen-zero truth).
#                   High = the bit's LLR points the wrong way under stress.
#   post_sig        post-precoder sigma at this frozen position.  Non-zero
#                   means T-mixing routes other v=1 events into u_j on the
#                   wrong path.  Promoting such a bit substantially changes
#                   the u_j behavior because u_j was being driven by the
#                   precoder, not by v_j (which is zero in the frozen state).
#   post_sigLLR     post-domain sigma * E[|LLR| | post=1] — the σ·|LLR|
#                   metric in the post domain.
#   src_sigLLR      source-domain sigma * E[|LLR| | source=1].  For frozen
#                   bits this is zero by construction (v_j = 0 always),
#                   surfaced only as a sanity-check column.
#   sigma_chosen    chosen-domain sigma (drives the kicker's sigma_low /
#                   sigma_chosen sort variants).
#
# Sorted to match _local_suggest_tail_swaps_kicker's add-pool ordering so
# what you see at the top of this table is what the kicker actually picks
# from at this iteration.
# ============================================================
function print_stage1_add_candidate_report(res, cfg::PipelineConfig,
                                            A::AbstractVector{<:Integer},
                                            rel::AbstractVector{<:Real};
                                            max_rows::Int = 32)
    stats = _stage1_priority_vectors(res, cfg)
    Aset = Set(Int.(A))
    F_full = [i for i in 1:cfg.N if !(i in Aset)]

    if isempty(F_full)
        println("\nStage-1 add-candidate report: no frozen bits to summarize.")
        return
    end

    # Replicate the add-pool post_sig quantile filter from the kicker so the
    # report shows the same pool the kicker is actually picking from.
    F, add_post_sig_threshold, filter_applied = if cfg.add_post_sig_keep_quantile >= 1.0
        F_full, Inf, false
    else
        post_sigs_F = [stats.post_sigma[i] for i in F_full]
        q = clamp(cfg.add_post_sig_keep_quantile, 0.0, 1.0)
        thr = quantile(post_sigs_F, q)
        kept = [i for i in F_full if stats.post_sigma[i] <= thr]
        if isempty(kept)
            F_full, thr, false  # filter would empty the pool; kicker falls back to F_full
        else
            kept, thr, true
        end
    end

    # Replicate the ordering used inside _local_suggest_tail_swaps_kicker so
    # row 1 of the report is the kicker's top add pick.
    add_key_fn, active_label = if cfg.add_use_zhat_dang
        (i -> (max(stats.zhat_dang[i] - cfg.add_zhat_floor, 0.0),
               -Float64(rel[i]), i)),
        "add_use_zhat_dang (lowest Zhat_d - floor, ties → DEGA)"
    elseif cfg.add_strategy in (:low_sigma, :low_support, :low_post_sigma, :low_chosen_sigma)
        (i -> (stats.sigma[i], -Float64(rel[i]), i)),
        "add_strategy = :low_sigma (lowest chosen-domain σ, ties → DEGA)"
    elseif cfg.add_strategy in (:high_reliability_low_sigma, :top_reliability_low_sigma)
        (i -> (-Float64(rel[i]), stats.sigma[i], i)),
        "add_strategy = :high_reliability_low_sigma"
    elseif cfg.add_strategy in (:zhat_dang, :low_zhat_dang)
        (i -> (max(stats.zhat_dang[i] - cfg.add_zhat_floor, 0.0),
               -Float64(rel[i]), i)),
        "add_strategy = :zhat_dang"
    elseif cfg.add_strategy in (:zhat_then_reliability, :low_zhat_then_reliability)
        (i -> (stats.zhat_dang[i], -Float64(rel[i]), i)),
        "add_strategy = :zhat_then_reliability"
    elseif cfg.add_strategy in (:q10_llr_dang, :q10_llr, :top_q10)
        (i -> (-Float64(stats.q10_abs_llr_dang[i]), -Float64(rel[i]), i)),
        "add_strategy = :q10_llr_dang (highest q10|LLR|_d, ties → DEGA)"
    else
        (i -> (-Float64(rel[i]), i)),
        "add_strategy = :top_reliability (DEGA descending)"
    end

    sorted_F = sort(F; by = add_key_fn)

    println("\nStage-1 add-candidate report (frozen bits, in kicker add-pool order)")
    println("  active ranking: ", active_label)
    @printf("  add_use_zhat_dang = %s, add_zhat_weight = %.4g, add_zhat_floor = %.4g\n",
            string(cfg.add_use_zhat_dang), cfg.add_zhat_weight, cfg.add_zhat_floor)
    @printf("  n_paths_llr = %d (dangerous samples backing Zhat / |LLR| stats)\n",
            stats.n_paths_llr)
    if cfg.add_post_sig_keep_quantile >= 1.0
        println("  post_sig filter: OFF (add_post_sig_keep_quantile = 1.0)")
    elseif filter_applied
        @printf("  post_sig filter: keep_quantile = %.3g → post_sig ≤ %.4e, kept %d / %d frozen bits\n",
                cfg.add_post_sig_keep_quantile, add_post_sig_threshold,
                length(F), length(F_full))
    else
        @printf("  post_sig filter: keep_quantile = %.3g → threshold = %.4e, but filter empties pool → fallback to all %d frozen bits\n",
                cfg.add_post_sig_keep_quantile, add_post_sig_threshold, length(F_full))
    end

    @printf("%6s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
            "i", "rel(DEGA)", "Zhat_d", "|LLR|_d", "q10|LLR|_d",
            "sign_err_d", "post_sig", "post_sigLLR", "src_sigLLR", "sigma_chosen")
    @printf("%6s %12s %12s %12s %12s %12s %12s %12s %12s %12s\n",
            "------", "------------", "------------", "------------", "------------",
            "------------", "------------", "------------", "------------", "------------")

    n_show = min(max_rows, length(sorted_F))
    for i in first(sorted_F, n_show)
        @printf("%6d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                i, Float64(rel[i]), stats.zhat_dang[i],
                stats.mean_abs_llr_dang[i], stats.q10_abs_llr_dang[i],
                stats.sign_err_rate_dang[i],
                stats.post_sigma[i], stats.post_sigma_llr[i],
                stats.source_sigma_llr[i], stats.sigma[i])
    end

    # Sanity check: list the top-post_sig bits the filter dropped so the user
    # can see what would have entered the pool without the filter.  These are
    # the positions where the precoder is most actively routing wrong-path
    # mass; promoting them would destroy real parity-check work.
    if filter_applied
        excluded_set = Set(F)
        excluded = [i for i in F_full if !(i in excluded_set)]
        if !isempty(excluded)
            excl_sorted = sort(excluded; by = i -> (-stats.post_sigma[i], i))
            n_excl_show = min(8, length(excl_sorted))
            println("  ... excluded by post_sig filter (top by post_sig):")
            for i in first(excl_sorted, n_excl_show)
                @printf("%6d %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                        i, Float64(rel[i]), stats.zhat_dang[i],
                        stats.mean_abs_llr_dang[i], stats.q10_abs_llr_dang[i],
                        stats.sign_err_rate_dang[i],
                        stats.post_sigma[i], stats.post_sigma_llr[i],
                        stats.source_sigma_llr[i], stats.sigma[i])
            end
        end
    end
end

function _stage1_candidate_generator_mode(cfg::PipelineConfig)
    if cfg.set_candidate_generator == :auto
        return cfg.set_sigma_domain in (:source, :pre, :u) ? :dps : :local
    elseif cfg.set_candidate_generator in (:dps, :local)
        return cfg.set_candidate_generator
    else
        throw(ArgumentError("Unknown set_candidate_generator $(cfg.set_candidate_generator)."))
    end
end

function _combos(v::Vector{Int}, k::Int)
    out = Vector{Vector{Int}}()
    k < 0 && return out
    if k == 0
        push!(out, Int[])
        return out
    end
    length(v) < k && return out
    cur = Int[]
    function rec(start::Int, remaining::Int)
        if remaining == 0
            push!(out, copy(cur))
            return
        end
        last_start = length(v) - remaining + 1
        for idx in start:last_start
            push!(cur, v[idx])
            rec(idx + 1, remaining - 1)
            pop!(cur)
        end
    end
    rec(1, k)
    return out
end

function _local_suggest_tail_swaps_kicker(A::Vector{Int},
                                          N::Int,
                                          res,
                                          cfg::PipelineConfig,
                                          rel::Vector{Float64};
                                          swap_size::Int)
    stats = _stage1_priority_vectors(res, cfg)
    Aset = Set(Int.(A))
    F_full = [i for i in 1:N if !(i in Aset)]

    # Add-pool post_sig quantile filter --------------------------------------
    # Drop frozen positions where the current precoder T is actively routing
    # wrong-path mass (post_sig > q-quantile of the frozen-bit distribution).
    # These positions are doing parity-check work that promotion would
    # destroy; structurally-clean positions are safer adds.  The threshold is
    # within-run adaptive (a quantile rather than an absolute cutoff), so it
    # rescales automatically across SNRs / code dimensions / T densities.
    F, add_post_sig_threshold = if cfg.add_post_sig_keep_quantile >= 1.0 || isempty(F_full)
        F_full, Inf
    else
        post_sigs_F = [stats.post_sigma[i] for i in F_full]
        q = clamp(cfg.add_post_sig_keep_quantile, 0.0, 1.0)
        thr = quantile(post_sigs_F, q)
        kept = [i for i in F_full if stats.post_sigma[i] <= thr]
        # Paranoia fallback: if the filter empties the pool (e.g. q=0 and no
        # bits are exactly at the minimum) fall back to the full set.
        if isempty(kept)
            F_full, thr
        else
            kept, thr
        end
    end

    # Remove pool ----------------------------------------------------------
    # Hard-filter by rho when freeze_rho_min > 0 so bits whose v_i = 1 mostly
    # cancels through T (low rho) are not even considered.  Soft weighting
    # by rho lives inside priority_rho via _stage1_remove_sort_key.
    A_int = Int.(A)
    if cfg.freeze_rho_min > 0.0
        eligible_remove = [i for i in A_int if stats.rho[i] >= cfg.freeze_rho_min]
        # Fall back to the full info set if the rho filter would empty the pool;
        # otherwise the search just stalls without producing any candidates.
        if isempty(eligible_remove)
            eligible_remove = A_int
        end
    else
        eligible_remove = A_int
    end
    remove_pool = sort(eligible_remove; by = i -> _stage1_remove_sort_key(stats, cfg, rel, i))
    remove_pool = first(remove_pool, min(max(swap_size, cfg.nremove), length(remove_pool)))

    # Add pool -------------------------------------------------------------
    # Strategy precedence:
    #   1. add_use_zhat_dang takes over the add ranking entirely (lowest
    #      Zhat_j^dang first, with DEGA reliability as a tie-breaker).
    #   2. Otherwise fall back to the legacy strategies.
    if cfg.add_use_zhat_dang
        # Apply optional numerical floor (treat tiny Zhat as zero).
        zhat_eff = [max(stats.zhat_dang[j] - cfg.add_zhat_floor, 0.0) for j in 1:N]
        add_pool = sort(F; by = j -> (zhat_eff[j], -Float64(rel[j]), j))
    elseif cfg.add_strategy in (:low_sigma, :low_support, :low_post_sigma, :low_chosen_sigma)
        add_pool = sort(F; by = j -> (stats.sigma[j], -Float64(rel[j]), j))
    elseif cfg.add_strategy in (:high_reliability_low_sigma, :top_reliability_low_sigma)
        prelim = sort(F; by = j -> (-Float64(rel[j]), j))
        prelim = first(prelim, min(max(cfg.nadd, swap_size), length(prelim)))
        add_pool = sort(prelim; by = j -> (stats.sigma[j], -Float64(rel[j]), j))
    elseif cfg.add_strategy in (:zhat_dang, :low_zhat_dang)
        zhat_eff = [max(stats.zhat_dang[j] - cfg.add_zhat_floor, 0.0) for j in 1:N]
        add_pool = sort(F; by = j -> (zhat_eff[j], -Float64(rel[j]), j))
    elseif cfg.add_strategy in (:zhat_then_reliability, :low_zhat_then_reliability)
        prelim = sort(F; by = j -> (stats.zhat_dang[j], j))
        prelim = first(prelim, min(max(cfg.nadd, swap_size), length(prelim)))
        add_pool = sort(prelim; by = j -> (-Float64(rel[j]), j))
    elseif cfg.add_strategy in (:q10_llr_dang, :q10_llr, :top_q10)
        # Sort by 10th-percentile |LLR| on dangerous trials, descending.
        # A high q10|LLR|_d means even in the bottom decile of LLR strength
        # on the current failure tail, the channel is still confident.
        # Robust to occasional outlier-huge LLRs (unlike mean |LLR|) and
        # to occasional near-zero LLRs (unlike Zhat, which exponentially
        # amplifies tiny LLR events).  DEGA breaks ties so bits with the
        # same q10 are ordered by their channel-marginal reliability.
        add_pool = sort(F; by = j -> (-Float64(stats.q10_abs_llr_dang[j]),
                                       -Float64(rel[j]), j))
    else
        add_pool = sort(F; by = j -> (-Float64(rel[j]), j))
    end
    add_pool = first(add_pool, min(max(swap_size, cfg.nadd), length(add_pool)))

    remove_sets = _combos(remove_pool, swap_size)
    add_sets = _combos(add_pool, swap_size)
    cands = NamedTuple[]
    for rem in remove_sets
        remset = Set(rem)
        base = [x for x in A if !(x in remset)]
        remove_risk = cfg.freeze_use_rho ?
                      sum(stats.priority_rho[i] for i in rem) :
                      sum(stats.priority[i] for i in rem)
        remove_lambda = sum(stats.lambda[i] for i in rem)
        remove_sigma = sum(stats.sigma[i] for i in rem)
        remove_source_sigma = sum(stats.source_sigma[i] for i in rem)
        remove_post_sigma = sum(stats.post_sigma[i] for i in rem)
        remove_rho_avg = isempty(rem) ? 0.0 : mean(Float64(stats.rho[i]) for i in rem)
        for add in add_sets
            Anew = sort(vcat(base, add))
            length(Anew) == length(A) || continue
            add_zhat_sum = sum(Float64(stats.zhat_dang[j]) for j in add)
            push!(cands, (
                A = Anew,
                removes = copy(rem),
                adds = copy(add),
                remove_risk = remove_risk,
                remove_lambda = remove_lambda,
                remove_sigma = remove_sigma,
                remove_source_sigma = remove_source_sigma,
                remove_post_sigma = remove_post_sigma,
                remove_rho_avg = remove_rho_avg,
                add_reliability = sum(Float64(rel[j]) for j in add),
                add_sigma = sum(stats.sigma[j] for j in add),
                add_source_sigma = sum(stats.source_sigma[j] for j in add),
                add_post_sigma = sum(stats.post_sigma[j] for j in add),
                add_zhat_dang = add_zhat_sum,
                add_zhat_avg = isempty(add) ? 0.0 : add_zhat_sum / length(add),
                composite_score = Float64(remove_risk) - cfg.add_zhat_weight * add_zhat_sum,
                sigma_domain = cfg.set_sigma_domain,
                generator = :local,
            ))
        end
    end

    # Composite ranking.  When add_use_zhat_dang is enabled, the primary key
    # is (remove_risk - beta * Zhat_sum) descending; this is the "good freeze
    # benefit, low add cost" pair score discussed in the design notes.  Off
    # by default to preserve the original behavior.
    if cfg.add_use_zhat_dang
        sort!(cands; by = c -> (-Float64(c.composite_score),
                                -Float64(c.remove_risk),
                                Float64(c.add_zhat_dang),
                                -Float64(c.add_reliability),
                                string(c.removes), string(c.adds)))
    else
        sort!(cands; by = c -> (-Float64(c.remove_risk), -Float64(c.add_reliability),
                                Float64(c.add_sigma), string(c.removes), string(c.adds)))
    end
    return first(cands, min(cfg.max_kicker_candidates, length(cands)))
end

function _stage1_candidate_sets(A::Vector{Int},
                                N::Int,
                                res,
                                cfg::PipelineConfig,
                                rel::Vector{Float64};
                                swap_size::Int)
    mode = _stage1_candidate_generator_mode(cfg)
    if mode == :dps
        return DPS.suggest_tail_swaps_kicker(
            A, N, res;
            reliability    = rel,
            swap_size      = swap_size,
            mode           = :explicit,
            risk_mix       = cfg.risk_mix,
            score          = cfg.set_risk_score,
            add_strategy   = cfg.add_strategy,
            nremove        = cfg.nremove,
            nadd           = cfg.nadd,
            max_candidates = cfg.max_kicker_candidates,
        )
    else
        return _local_suggest_tail_swaps_kicker(A, N, res, cfg, rel;
                                                swap_size = swap_size)
    end
end

function _candidate_sort_key(c)
    return (-Float64(c.remove_risk), -Float64(c.add_reliability))
end

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function save_vector(path::AbstractString, v)
    open(path, "w") do io
        for x in v
            println(io, x)
        end
    end
end

function save_set_history_csv(path::AbstractString, history)
    open(path, "w") do io
        println(io, "iter,swap_size,accepted,reverted,removes,adds,old_score,new_score,n_events")
        for h in history
            println(io, join((
                h.iter, h.swap_size, h.accepted, h.reverted,
                join(h.removes, "|"), join(h.adds, "|"),
                h.old_score, h.new_score, h.n_events,
            ), ","))
        end
    end
end

function save_edges_csv(path::AbstractString, edges)
    open(path, "w") do io
        println(io, "step,move_type,kicker_size,edges_added,saved_paths_estimate,saved_kept_l0,saved_lost_l0_urgent,saved_lost_l0_late,small_score,full_score,working_score,best_score")
        for e in edges
            edge_repr = join(["$(es.i)->$(es.j)" for es in e.edges], "|")
            println(io, join((e.step, e.move_type, e.kicker_size, edge_repr,
                              e.saved_paths_estimate, e.saved_kept_l0,
                              e.saved_lost_l0_urgent, e.saved_lost_l0_late,
                              e.small_score, e.full_score, e.working_score,
                              e.best_score), ","))
        end
    end
end

# ============================================================
# Stage 1: non-greedy information-set design with top-k full evaluation
# ============================================================

# Shared screen-and-confirm logic for Stage-1 set search.
#
# Given a pre-built pool of candidate_items (each with fields swap_size, k, cand),
# small-screens all of them, then fully evaluates the top set_full_eval_top_k
# candidates and returns the best.  Used by both batch mode (all kicker sizes
# pooled) and escalation mode (one size at a time).
function _screen_and_confirm_stage1_pool(
        cfg::PipelineConfig,
        iter::Int,
        candidate_items,
        A::Vector{Int},
        res,
        score::Float64,
        Tgeneric::AbstractMatrix{<:Integer}
)
    pool_history = NamedTuple[]

    if isempty(candidate_items)
        cfg.verbose && println("\nNo information-set swap candidates were generated.")
        return (accepted = false, A = A, res = res, score = score, history = pool_history)
    end

    cfg.verbose && begin
        println("\n>>> iter $iter  non-greedy information-set swap screen  <<<")
        @printf("  small-evaluating %d total candidate(s) at %d trials\n",
                length(candidate_items), cfg.n_trials_candidate)
        @printf("  then full-evaluating up to top %d candidate(s) at %d trials\n",
                max(1, cfg.set_full_eval_top_k), cfg.n_trials_set)
    end

    small_results = NamedTuple[]

    for (global_k, item) in enumerate(candidate_items)
        cand = item.cand
        label = @sprintf("iter=%d pooled=%d sz=%d cand=%d remove=%s add=%s",
                         iter, global_k, item.swap_size, item.k,
                         string(cand.removes), string(cand.adds))
        res_try = spectrum_for_A(cfg, cand.A, Tgeneric;
                                 n_trials = cfg.n_trials_candidate,
                                 seed = cfg.seed + 10_000 * iter + 1_000 * item.swap_size + item.k,
                                 label = cfg.verbose ? label : "")
        s_try = tail_score(res_try, cfg)

        cfg.verbose && @printf("    pool %3d sz=%d local=%3d: rm %s  ad %s  score(small) = %.8e\n",
                                global_k, item.swap_size, item.k,
                                string(cand.removes), string(cand.adds), s_try)

        push!(small_results, (
            global_k = global_k,
            swap_size = item.swap_size,
            k = item.k,
            cand = cand,
            small_score = s_try,
            small_improves = s_try < score - cfg.min_set_improvement,
        ))
    end

    ranked_small = sort(small_results; by = x -> (x.small_score, x.swap_size, x.k))
    full_pool = cfg.set_full_eval_only_small_improvers ?
                filter(x -> x.small_improves, ranked_small) :
                ranked_small

    if isempty(full_pool)
        cfg.verbose && println("  No small-screen improvers.")
        push!(pool_history, (
            iter = iter, swap_size = 0,
            accepted = false, reverted = false,
            removes = Int[], adds = Int[],
            old_score = score, new_score = score,
            n_events = length(res.events),
        ))
        return (accepted = false, A = A, res = res, score = score, history = pool_history)
    end

    n_full = min(max(1, cfg.set_full_eval_top_k), length(full_pool))
    full_pool = first(full_pool, n_full)

    cfg.verbose && begin
        println("  Full-budget confirmation pool:")
        for (rank, item) in enumerate(full_pool)
            @printf("    rank %2d: sz=%d local=%d small=%.8e rm %s ad %s\n",
                    rank, item.swap_size, item.k, item.small_score,
                    string(item.cand.removes), string(item.cand.adds))
        end
    end

    best_full = nothing
    best_full_score = Inf
    best_full_res = nothing
    best_full_small = Inf
    best_full_swap_size = 0

    for (rank, item) in enumerate(full_pool)
        cand = item.cand
        res_try_full = spectrum_for_A(cfg, cand.A, Tgeneric;
                                      n_trials = cfg.n_trials_set,
                                      seed = cfg.seed + 100_000 * iter + 10_000 * item.swap_size + item.k,
                                      label = @sprintf("confirm iter=%d pooled_rank=%d sz=%d cand=%d rm=%s ad=%s",
                                                       iter, rank, item.swap_size, item.k,
                                                       string(cand.removes), string(cand.adds)))
        score_full = tail_score(res_try_full, cfg)

        cfg.verbose && @printf("    full rank %2d sz=%d cand %3d: small=%.8e  full=%.8e  prev=%.8e  rm %s  ad %s\n",
                                rank, item.swap_size, item.k, item.small_score,
                                score_full, score,
                                string(cand.removes), string(cand.adds))

        if score_full < best_full_score
            best_full = cand
            best_full_score = score_full
            best_full_res = res_try_full
            best_full_small = item.small_score
            best_full_swap_size = item.swap_size
        end
    end

    if best_full !== nothing && best_full_score < score - cfg.min_set_improvement
        @printf("  ACCEPTED iter=%d swap_size=%d  rm=%s  add=%s: small=%.8e  full %.8e -> %.8e\n",
                iter, best_full_swap_size,
                string(best_full.removes), string(best_full.adds),
                best_full_small, score, best_full_score)
        push!(pool_history, (
            iter = iter, swap_size = best_full_swap_size,
            accepted = true, reverted = false,
            removes = Int.(best_full.removes), adds = Int.(best_full.adds),
            old_score = score, new_score = best_full_score,
            n_events = length(best_full_res.events),
        ))
        return (accepted = true, A = best_full.A, res = best_full_res,
                score = best_full_score, history = pool_history)
    else
        if best_full === nothing
            cfg.verbose && println("  No full candidates evaluated.")
            removes = Int[]
            adds = Int[]
            new_score = score
            swap_size_rec = 0
        else
            cfg.verbose && @printf("  REVERTED best full candidate from swap_size=%d (full=%.8e vs prev=%.8e).\n",
                                    best_full_swap_size, best_full_score, score)
            removes = Int.(best_full.removes)
            adds = Int.(best_full.adds)
            new_score = best_full_score
            swap_size_rec = best_full_swap_size
        end
        push!(pool_history, (
            iter = iter, swap_size = swap_size_rec,
            accepted = false, reverted = best_full !== nothing,
            removes = removes, adds = adds,
            old_score = score, new_score = new_score,
            n_events = length(res.events),
        ))
        return (accepted = false, A = A, res = res, score = score, history = pool_history)
    end
end

function _try_kicker_escalation(cfg::PipelineConfig,
                                  iter::Int,
                                  A::Vector{Int},
                                  res,
                                  score::Float64,
                                  rel::Vector{Float64},
                                  Tgeneric::AbstractMatrix{<:Integer})
    new_history = NamedTuple[]

    if cfg.set_kicker_escalate
        # Escalation mode: try kicker sizes one at a time in order.
        # Commit the first size that produces an improvement; only move on to a
        # larger kicker size when the current size finds no improvement.
        for swap_size in cfg.kicker_sizes
            cfg.verbose && println("\n>>> iter $iter  escalation: trying swap_size = $swap_size  <<<")

            cands = _stage1_candidate_sets(A, cfg.N, res, cfg, rel; swap_size = swap_size)

            if isempty(cands)
                cfg.verbose && @printf("  swap_size=%d: no candidates; escalating to next kicker size.\n", swap_size)
                continue
            end

            cands = first(cands, min(cfg.max_candidates_per_iter, length(cands)))
            cfg.verbose && @printf("  swap_size=%d: keeping %d heuristic candidate(s)\n",
                                    swap_size, length(cands))
            if cfg.verbose && (cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel))
                println("  Stage-1 removal priority: post-convolution sigma; lambda is not used when lambda weight is zero.")
            end

            items = [(swap_size = swap_size, k = k, cand = cand) for (k, cand) in enumerate(cands)]
            result = _screen_and_confirm_stage1_pool(cfg, iter, items, A, res, score, Tgeneric)
            append!(new_history, result.history)

            if result.accepted
                return (accepted = true, A = result.A, res = result.res,
                        score = result.score, history = new_history)
            end

            cfg.verbose && @printf("  swap_size=%d: no improvement found; escalating to next kicker size.\n", swap_size)
        end

        cfg.verbose && println("\nAll kicker sizes exhausted (escalation) at iter ", iter, "; stopping search.")
        if isempty(new_history)
            push!(new_history, (
                iter = iter, swap_size = 0, accepted = false, reverted = false,
                removes = Int[], adds = Int[], old_score = score, new_score = score,
                n_events = length(res.events),
            ))
        end
        return (accepted = false, A = A, res = res, score = score, history = new_history)
    end

    # Batch mode (default): pool candidates from all configured kicker sizes,
    # small-screen the combined pool, then fully evaluate the top candidates.
    # This lets a 2-swap beat a 1-swap when the 2-swap is genuinely better.
    candidate_items = NamedTuple[]

    for swap_size in cfg.kicker_sizes
        cfg.verbose && println("\n>>> iter $iter  generating swap_size = $swap_size candidates  <<<")

        cands = _stage1_candidate_sets(A, cfg.N, res, cfg, rel; swap_size = swap_size)

        if isempty(cands)
            cfg.verbose && @printf("  swap_size=%d: no candidates.\n", swap_size)
            continue
        end

        cands = first(cands, min(cfg.max_candidates_per_iter, length(cands)))
        cfg.verbose && @printf("  swap_size=%d: keeping %d heuristic candidate(s) for the shared pool\n",
                                swap_size, length(cands))
        if cfg.verbose && (cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel))
            println("  Stage-1 removal priority: post-convolution sigma; lambda is not used when lambda weight is zero.")
        end

        for (k, cand) in enumerate(cands)
            push!(candidate_items, (swap_size = swap_size, k = k, cand = cand))
        end
    end

    result = _screen_and_confirm_stage1_pool(cfg, iter, candidate_items, A, res, score, Tgeneric)
    append!(new_history, result.history)
    return (accepted = result.accepted, A = result.A, res = result.res,
            score = result.score, history = new_history)
end

function run_generic_pac_set_design(cfg::PipelineConfig)
    rel = reliability_vector(cfg)
    A = initial_information_set(cfg, rel)

    Tgeneric = build_stage1_Tmat(cfg)

    res = spectrum_for_A(cfg, A, Tgeneric;
                         n_trials = cfg.n_trials_set,
                         seed = cfg.seed,
                         label = "initial generic PAC set")
    score = tail_score(res, cfg)

    cfg.verbose && begin
        println("Initial A = ", A)
        @printf("Initial tail objective (%s) = %.8e\n", String(cfg.set_objective), score)
        print_stage1_priority_report(res, cfg)
        print_stage1_add_candidate_report(res, cfg, A, rel)
    end

    history = NamedTuple[]

    for iter in 1:cfg.set_iters
        result = _try_kicker_escalation(cfg, iter, A, res, score, rel, Tgeneric)
        append!(history, result.history)

        if result.accepted
            A = result.A
            res = result.res
            score = result.score
            cfg.verbose && begin
                println("Current A = ", A)
                print_stage1_priority_report(res, cfg)
                print_stage1_add_candidate_report(res, cfg, A, rel)
            end
        else
            cfg.verbose && println("\nAll kicker sizes exhausted at iter ", iter, "; stopping search.")
            break
        end
    end

    return (A = A, result = res, score = score, reliability = rel,
            Tgeneric = Tgeneric, depths_generic = DPS.infer_depths(Tgeneric),
            history = history)
end

# ============================================================
# Stage 2: from-scratch PAC precoder construction
# ============================================================

@inline function _sample_l0_survived(s)
    return isfinite(Float64(s.scl_gap))
end

@inline function _lost_l0_deadline(s, N::Int, lost_window::Int)
    return min(N, Int(s.first_error) + max(0, lost_window))
end

"""
For each candidate single edge (i, j), count paths that would be pushed to
PM-gap > threshold if the edge were added.

L0-aware behavior:
  * If the all-zero/correct path survives the final SCL list, the edge gets
    ordinary saved-path credit for any allowed frozen target j.
  * If the all-zero/correct path is absent from the final list, the edge gets
    credit only when j <= first_error + lost_window.  If a later j would have
    helped according to the final-PM surrogate, it is counted separately as
    saved_lost_late for diagnostics but does not drive the ranking.
"""
function _score_single_edges(res::DPS.SpectrumResult,
                              T_current::AbstractMatrix{<:Integer},
                              N::Int,
                              A::AbstractVector{<:Integer};
                              max_span::Int = 24,
                              target::Symbol = :frozen,
                              only_dangerous::Bool = true,
                              threshold::Float64 = 0.0,
                              l0_aware::Bool = true,
                              lost_window::Int = 32,
                              kept_weight::Float64 = 1.0,
                              lost_weight::Float64 = 1.0,
                              late_penalty::Float64 = 0.0)
    samples = res.samples_by_mode[:explicit]
    Aset = Set(Int.(A))

    score = zeros(Float64, N, N)
    saved_kept = zeros(Float64, N, N)
    saved_lost_urgent = zeros(Float64, N, N)
    saved_lost_late = zeros(Float64, N, N)
    n_filtered = 0
    n_l0_kept = 0
    n_l0_lost = 0

    @inbounds for s in samples
        s.mode === :explicit || continue
        if only_dangerous && Float64(s.D) > 0.0
            continue
        end
        n_filtered += 1

        l0_survived = _sample_l0_survived(s)
        if l0_survived
            n_l0_kept += 1
        else
            n_l0_lost += 1
        end
        deadline = l0_survived ? N : _lost_l0_deadline(s, N, lost_window)

        path_v = s.path_v_scl
        B      = s.B
        LLR    = s.LLR
        D      = Float64(s.D)

        for i in A
            path_v[i] == 0 && continue
            j_max = min(N, i + max_span)
            for j in (i + 1):j_max
                if target == :frozen
                    in(j, Aset) && continue
                elseif target == :all
                    # allow info -> info too
                else
                    throw(ArgumentError("pac_target must be :frozen or :all."))
                end
                T_current[i, j] != 0 && continue

                Δ = (1.0 - 2.0 * Float64(B[j])) * Float64(LLR[j])
                if D + Δ > threshold
                    if l0_survived || !l0_aware
                        saved_kept[i, j] += 1.0
                        score[i, j] += kept_weight
                    elseif j <= deadline
                        saved_lost_urgent[i, j] += 1.0
                        score[i, j] += lost_weight
                    else
                        saved_lost_late[i, j] += 1.0
                        score[i, j] -= late_penalty
                    end
                end
            end
        end
    end

    return (score = score,
            saved = score,
            saved_kept = saved_kept,
            saved_lost_urgent = saved_lost_urgent,
            saved_lost_late = saved_lost_late,
            n_paths_considered = n_filtered,
            n_l0_kept = n_l0_kept,
            n_l0_lost = n_l0_lost)
end

function _edge_repr(edges)
    return join(["$(e.i)->$(e.j)" for e in edges], ",")
end

function _apply_edges(T::AbstractMatrix{<:Integer}, edges)
    T_try = copy(T)
    for e in edges
        T_try[e.i, e.j] = 1
    end
    return T_try
end

function _combo_indices(n::Int, k::Int)
    out = Vector{Vector{Int}}()
    k <= 0 && return out
    k > n && return out
    cur = Int[]

    function rec(start::Int, left::Int)
        if left == 0
            push!(out, copy(cur))
            return
        end
        max_i = n - left + 1
        for idx in start:max_i
            push!(cur, idx)
            rec(idx + 1, left - 1)
            pop!(cur)
        end
    end

    rec(1, k)
    return out
end

function _edge_set_saved_paths(res::DPS.SpectrumResult,
                                edges;
                                only_dangerous::Bool = true,
                                threshold::Float64 = 0.0,
                                l0_aware::Bool = true,
                                lost_window::Int = 32,
                                kept_weight::Float64 = 1.0,
                                lost_weight::Float64 = 1.0,
                                late_penalty::Float64 = 0.0,
                                N::Int = 0)
    samples = res.samples_by_mode[:explicit]
    score = 0.0
    saved_kept = 0
    saved_lost_urgent = 0
    saved_lost_late = 0
    Nlocal = N <= 0 ? length(first(samples).B) : N

    @inbounds for s in samples
        s.mode === :explicit || continue
        if only_dangerous && Float64(s.D) > 0.0
            continue
        end

        l0_survived = _sample_l0_survived(s)
        deadline = l0_survived ? Nlocal : _lost_l0_deadline(s, Nlocal, lost_window)

        Δ_effective = 0.0
        Δ_all = 0.0
        for e in edges
            if s.path_v_scl[e.i] != 0
                Δe = (1.0 - 2.0 * Float64(s.B[e.j])) * Float64(s.LLR[e.j])
                Δ_all += Δe
                if l0_survived || !l0_aware || e.j <= deadline
                    Δ_effective += Δe
                end
            end
        end

        D = Float64(s.D)
        if D + Δ_effective > threshold
            if l0_survived || !l0_aware
                saved_kept += 1
                score += kept_weight
            else
                saved_lost_urgent += 1
                score += lost_weight
            end
        elseif !l0_survived && l0_aware && D + Δ_all > threshold
            saved_lost_late += 1
            score -= late_penalty
        end
    end

    return (score = score,
            saved_paths = score,
            saved_kept = Float64(saved_kept),
            saved_lost_urgent = Float64(saved_lost_urgent),
            saved_lost_late = Float64(saved_lost_late))
end

# Shared trial grouping for the trial-level edge scorers.
function _edge_trial_ranges(samples, only_dangerous::Bool)
    sel = [s for s in samples if s.mode === :explicit &&
                                 (!only_dangerous || Float64(s.D) <= 0.0)]
    isempty(sel) && return sel, UnitRange{Int}[]
    sort!(sel, by = s -> (s.trial, s.rank))
    ranges = UnitRange{Int}[]
    start = 1
    for p in 2:(length(sel) + 1)
        if p == length(sel) + 1 || sel[p].trial != sel[start].trial
            push!(ranges, start:(p - 1))
            start = p
        end
    end
    return sel, ranges
end

"""
Trial-level replacement for `_score_single_edges` (selected with
pac_edge_metric = :trials).  An edge (i, j) is credited for a TRIAL only
when every dangerous path recorded for that trial clears the threshold
after the edge's counterfactual Delta.  Per-path crediting overstates FER
impact whenever several wrong paths beat l0 in the same trial — the trial
still fails unless all of them are lifted — and those multi-path trials are
exactly what remains late in the search.  Interface, return shape, and
L0-aware bookkeeping match the legacy scorer (n_paths_considered counts
trials here).
"""
function _score_single_edges_trialwise(res::DPS.SpectrumResult,
                                       T_current::AbstractMatrix{<:Integer},
                                       N::Int,
                                       A::AbstractVector{<:Integer};
                                       max_span::Int = 24,
                                       target::Symbol = :frozen,
                                       only_dangerous::Bool = true,
                                       threshold::Float64 = 0.0,
                                       l0_aware::Bool = true,
                                       lost_window::Int = 32,
                                       kept_weight::Float64 = 1.0,
                                       lost_weight::Float64 = 1.0,
                                       late_penalty::Float64 = 0.0)
    Aset = Set(Int.(A))

    score = zeros(Float64, N, N)
    saved_kept = zeros(Float64, N, N)
    saved_lost_urgent = zeros(Float64, N, N)
    saved_lost_late = zeros(Float64, N, N)

    sel, ranges = _edge_trial_ranges(res.samples_by_mode[:explicit], only_dangerous)
    if isempty(ranges)
        return (score = score, saved = score, saved_kept = saved_kept,
                saved_lost_urgent = saved_lost_urgent,
                saved_lost_late = saved_lost_late,
                n_paths_considered = 0, n_l0_kept = 0, n_l0_lost = 0)
    end

    n_l0_kept = 0
    n_l0_lost = 0

    @inbounds for rg in ranges
        l0_survived = _sample_l0_survived(sel[first(rg)])
        l0_survived ? (n_l0_kept += 1) : (n_l0_lost += 1)

        for i in A
            # The edge can only act on paths carrying v_i = 1; skip sources
            # absent from every path of this trial.
            any_v = false
            for p in rg
                if sel[p].path_v_scl[i] != 0
                    any_v = true
                    break
                end
            end
            any_v || continue

            j_max = min(N, i + max_span)
            for j in (i + 1):j_max
                if target == :frozen
                    in(j, Aset) && continue
                elseif target != :all
                    throw(ArgumentError("pac_target must be :frozen or :all."))
                end
                T_current[i, j] != 0 && continue

                saved_gated = true    # deadline-gated Delta drives the score
                saved_ungated = true  # ungated Delta feeds the late diagnostic
                for p in rg
                    s = sel[p]
                    Δ = 0.0
                    if s.path_v_scl[i] != 0
                        Δ = (1.0 - 2.0 * Float64(s.B[j])) * Float64(s.LLR[j])
                    end
                    D = Float64(s.D)
                    if D + Δ <= threshold
                        saved_ungated = false
                    end
                    elig = !l0_aware || l0_survived ||
                           j <= s.first_error + lost_window
                    if D + (elig ? Δ : 0.0) <= threshold
                        saved_gated = false
                    end
                    saved_gated || saved_ungated || break
                end

                if saved_gated
                    if l0_survived || !l0_aware
                        saved_kept[i, j] += 1.0
                        score[i, j] += kept_weight
                    else
                        saved_lost_urgent[i, j] += 1.0
                        score[i, j] += lost_weight
                    end
                elseif !l0_survived && l0_aware && saved_ungated
                    saved_lost_late[i, j] += 1.0
                    score[i, j] -= late_penalty
                end
            end
        end
    end

    return (score = score, saved = score, saved_kept = saved_kept,
            saved_lost_urgent = saved_lost_urgent,
            saved_lost_late = saved_lost_late,
            n_paths_considered = length(ranges),
            n_l0_kept = n_l0_kept, n_l0_lost = n_l0_lost)
end

# Trial-level analogue of `_edge_set_saved_paths` for k-edge candidate sets.
function _edge_set_saved_trials(res::DPS.SpectrumResult,
                                edges;
                                only_dangerous::Bool = true,
                                threshold::Float64 = 0.0,
                                l0_aware::Bool = true,
                                lost_window::Int = 32,
                                kept_weight::Float64 = 1.0,
                                lost_weight::Float64 = 1.0,
                                late_penalty::Float64 = 0.0,
                                N::Int = 0)
    sel, ranges = _edge_trial_ranges(res.samples_by_mode[:explicit], only_dangerous)
    score = 0.0
    saved_kept = 0
    saved_lost_urgent = 0
    saved_lost_late = 0

    @inbounds for rg in ranges
        l0_survived = _sample_l0_survived(sel[first(rg)])
        saved_gated = true
        saved_ungated = true
        for p in rg
            s = sel[p]
            Δg = 0.0
            Δa = 0.0
            for e in edges
                s.path_v_scl[e.i] == 0 && continue
                Δe = (1.0 - 2.0 * Float64(s.B[e.j])) * Float64(s.LLR[e.j])
                Δa += Δe
                elig = !l0_aware || l0_survived ||
                       e.j <= s.first_error + lost_window
                elig && (Δg += Δe)
            end
            D = Float64(s.D)
            D + Δa <= threshold && (saved_ungated = false)
            D + Δg <= threshold && (saved_gated = false)
            saved_gated || saved_ungated || break
        end
        if saved_gated
            if l0_survived || !l0_aware
                saved_kept += 1
                score += kept_weight
            else
                saved_lost_urgent += 1
                score += lost_weight
            end
        elseif !l0_survived && l0_aware && saved_ungated
            saved_lost_late += 1
            score -= late_penalty
        end
    end

    return (score = score,
            saved_paths = score,
            saved_kept = Float64(saved_kept),
            saved_lost_urgent = Float64(saved_lost_urgent),
            saved_lost_late = Float64(saved_lost_late))
end

"""
Score k-edge candidate sets using combined saved-path count.

This replaces the previous 2-edge kicker behavior.  The previous code only kept
pairs with positive saved-excess beyond the sum of the individual saved counts.
That made the kicker nearly invisible: two edges can be a good joint move even
when their saved-path sets overlap or when the eventual FER gain is not captured
by positive saved-excess.  Here every candidate set with positive combined saved
count is eligible and is sorted by combined saved count, with saved-excess only a
tie-breaker/diagnostic.
"""
function _score_k_edge_sets(res::DPS.SpectrumResult,
                             top_single::Vector{<:NamedTuple};
                             kicker_size::Int = 2,
                             only_dangerous::Bool = true,
                             threshold::Float64 = 0.0,
                             max_sets::Int = 96,
                             l0_aware::Bool = true,
                             lost_window::Int = 32,
                             kept_weight::Float64 = 1.0,
                             lost_weight::Float64 = 1.0,
                             late_penalty::Float64 = 0.0,
                             N::Int = 0,
                             metric::Symbol = :paths)
    set_scorer = metric == :trials ? _edge_set_saved_trials : _edge_set_saved_paths
    sets = NamedTuple[]

    if kicker_size == 1
        for c in top_single
            push!(sets, (edges = [c],
                         kicker_size = 1,
                         saved_paths = Float64(c.saved_paths),
                         saved_kept = Float64(c.saved_kept),
                         saved_lost_urgent = Float64(c.saved_lost_urgent),
                         saved_lost_late = Float64(c.saved_lost_late),
                         saved_excess = 0.0,
                         score = Float64(c.saved_paths)))
        end
        sort!(sets, by = p -> -p.score)
        return first(sets, min(max_sets, length(sets)))
    end

    combos = _combo_indices(length(top_single), kicker_size)
    isempty(combos) && return sets

    for idxs in combos
        edges = [top_single[t] for t in idxs]
        stats = set_scorer(res, edges;
                                      only_dangerous = only_dangerous,
                                      threshold = threshold,
                                      l0_aware = l0_aware,
                                      lost_window = lost_window,
                                      kept_weight = kept_weight,
                                      lost_weight = lost_weight,
                                      late_penalty = late_penalty,
                                      N = N)
        stats.score <= 0 && continue

        indiv_sum = sum(Float64(e.saved_paths) for e in edges)
        saved_excess = Float64(stats.score) - indiv_sum
        push!(sets, (edges = edges,
                     kicker_size = kicker_size,
                     saved_paths = Float64(stats.score),
                     saved_kept = Float64(stats.saved_kept),
                     saved_lost_urgent = Float64(stats.saved_lost_urgent),
                     saved_lost_late = Float64(stats.saved_lost_late),
                     saved_excess = saved_excess,
                     score = Float64(stats.score)))
    end

    sort!(sets, by = p -> (-p.score, -p.saved_kept, -p.saved_lost_urgent, p.saved_lost_late, p.kicker_size))
    return first(sets, min(max_sets, length(sets)))
end

function _pac_small_slack(cfg::PipelineConfig)
    return cfg.pac_small_screen_slack_events <= 0.0 ? 0.0 :
           cfg.pac_small_screen_slack_events / max(1, cfg.pac_n_trials_small)
end

function _pac_explore_limit_from_current(cfg::PipelineConfig, current_score::Float64)
    return current_score + cfg.pac_explore_max_abs_worse +
           cfg.pac_explore_max_rel_worse * abs(current_score)
end

function _pac_explore_limit_from_best(cfg::PipelineConfig, best_score::Float64)
    return best_score + cfg.pac_explore_max_abs_above_best +
           cfg.pac_explore_max_rel_above_best * abs(best_score)
end

function _pac_confirm_limit(cfg::PipelineConfig,
                             current_score::Float64,
                             best_score::Float64)
    limit = current_score - cfg.pac_min_improvement + _pac_small_slack(cfg)
    if cfg.pac_explore_when_stalled
        limit = max(limit,
                    min(_pac_explore_limit_from_current(cfg, current_score),
                        _pac_explore_limit_from_best(cfg, best_score)) + _pac_small_slack(cfg))
    end
    return limit
end

function _pac_is_improvement(cfg::PipelineConfig,
                              score_new::Float64,
                              current_score::Float64)
    return score_new < current_score - cfg.pac_min_improvement
end

function _pac_explore_allowed(cfg::PipelineConfig,
                               score_new::Float64,
                               current_score::Float64,
                               best_score::Float64,
                               explore_used::Int)
    cfg.pac_explore_when_stalled || return false
    explore_used < cfg.pac_max_explore_steps || return false
    score_new <= _pac_explore_limit_from_current(cfg, current_score) || return false
    score_new <= _pac_explore_limit_from_best(cfg, best_score) || return false
    return true
end

function _evaluate_edge_set_small(cfg::PipelineConfig,
                                   A::AbstractVector{Int},
                                   T::AbstractMatrix{<:Integer},
                                   cset,
                                   step::Int,
                                   k_index::Int)
    T_try = _apply_edges(T, cset.edges)
    edge_repr = _edge_repr(cset.edges)
    label = cfg.verbose ?
            @sprintf("step=%d sz=%d k=%d edges=%s",
                     step, cset.kicker_size, k_index, edge_repr) :
            ""

    res_small = spectrum_for_A(cfg, A, T_try;
                               n_trials = cfg.pac_n_trials_small,
                               seed = cfg.seed + 1_000 * step + 100 * cset.kicker_size + k_index,
                               label = label)
    score_small = tail_score(res_small, cfg)

    cfg.verbose && @printf("    small  (%dk trials)  sz=%d edges=%s : %.8e\n",
                           cfg.pac_n_trials_small ÷ 1000, cset.kicker_size,
                           edge_repr, score_small)

    return (cset = cset, res = res_small, score = score_small,
            edge_repr = edge_repr, small_rank = k_index)
end

function _evaluate_edge_set_full(cfg::PipelineConfig,
                                  A::AbstractVector{Int},
                                  T::AbstractMatrix{<:Integer},
                                  small_eval,
                                  current_score::Float64,
                                  step::Int,
                                  confirm_index::Int)
    cset = small_eval.cset
    T_try = _apply_edges(T, cset.edges)
    edge_repr = small_eval.edge_repr
    label = cfg.verbose ?
            @sprintf("confirm step=%d sz=%d small_rank=%d confirm=%d edges=%s",
                     step, cset.kicker_size, small_eval.small_rank,
                     confirm_index, edge_repr) :
            ""

    res_full = spectrum_for_A(cfg, A, T_try;
                              n_trials = cfg.pac_n_trials_full,
                              seed = cfg.seed + 10_000 * step + 1_000 * cset.kicker_size + confirm_index,
                              label = label,
                              collect_replay = cfg.pac_replay_screen)
    score_full = tail_score(res_full, cfg)

    cfg.verbose && @printf("    full   (%dk trials)  sz=%d edges=%s : %.8e (prev %.8e, small %.8e)\n",
                           cfg.pac_n_trials_full ÷ 1000, cset.kicker_size,
                           edge_repr, score_full, current_score, small_eval.score)

    return (cset = cset, res = res_full, score = score_full,
            score_small = small_eval.score, edge_repr = edge_repr,
            small_rank = small_eval.small_rank, confirm_rank = confirm_index)
end

function _initial_stage2_tmat(cfg::PipelineConfig, set_out)
    if cfg.pac_initial == :identity
        return identity_tmat(cfg.N)
    elseif cfg.pac_initial == :generic
        return copy(set_out.Tgeneric)
    elseif cfg.pac_initial == :custom
        if cfg.pac_custom_Tmat !== nothing
            T = UInt8.(cfg.pac_custom_Tmat)
        elseif cfg.pac_custom_Tmat_path !== nothing
            T = UInt8.(round.(Int, readdlm(cfg.pac_custom_Tmat_path, ',')))
        else
            throw(ArgumentError("pac_initial=:custom requires pac_custom_Tmat or pac_custom_Tmat_path."))
        end
        size(T, 1) == cfg.N && size(T, 2) == cfg.N ||
            throw(ArgumentError("custom Tmat must be N x N."))
        @inbounds for i in 1:cfg.N
            T[i, i] == 1 || throw(ArgumentError("custom Tmat must have all diagonal entries equal to 1."))
            for j in 1:(i - 1)
                T[i, j] == 0 || throw(ArgumentError("custom Tmat must be upper triangular under the i -> j convention."))
            end
        end
        return T
    else
        throw(ArgumentError("pac_initial must be :identity, :generic, or :custom."))
    end
end

# ============================================================
# Stage 2 (:parity): per-target column design with replay screening
#
# Rationale.  A precoder column S_j forces u_j = XOR_{i in S_j} v_i, so a
# wrong path pays the |LLR_j| penalty iff its error pattern has ODD overlap
# with S_j.  Coverage is therefore parity-structured: adding a source to S_j
# XOR-flips the covered/uncovered status of every dangerous path containing
# that source.  Per-edge greedy can only walk monotone chains through
# subsets and stalls exactly when a target gets overloaded (even-parity
# evasion).  Here each move is a whole-column redesign found by local flip
# search against a TRIAL-level saved count (a trial is saved only if every
# one of its dangerous paths clears the threshold), then screened by an
# exact SCL replay of the stored failure/near-miss trials, then confirmed
# with fresh Monte Carlo.
#
# Division of labor, made explicit because it is the crux:
#   * surrogate (frozen-sample counterfactual) — proposes columns.  It can
#     only ever see "add a penalty to a dangerous path"; it is structurally
#     unable to justify pruning (removals only delete penalties that wrong
#     paths are currently paying) and it cannot see harm to safe trials.
#   * replay — disposes.  Exact under the modified T on the population that
#     carries the FER delta, paired across candidates, and two-sided (sees
#     broken near-misses).  All pruning credit flows through replay.
#   * fresh MC — final arbiter for the commit, as in the legacy driver.
# ============================================================

# Group the explicit dangerous samples by trial.  Trial-level scoring is the
# point: crediting individual saved paths overstates FER impact whenever a
# trial holds several dangerous paths (the trial fails unless ALL are lifted).
function _parity_trial_data(res::DPS.SpectrumResult, cfg::PipelineConfig)
    samples = res.samples_by_mode[:explicit]
    sel = [s for s in samples
           if s.mode === :explicit &&
              (!cfg.pac_only_dangerous || Float64(s.D) <= 0.0)]
    isempty(sel) && return nothing
    sort!(sel, by = s -> (s.trial, s.rank))

    ranges = UnitRange{Int}[]
    w = Float64[]
    lost = Bool[]
    start = 1
    for p in 2:(length(sel) + 1)
        if p == length(sel) + 1 || sel[p].trial != sel[start].trial
            push!(ranges, start:(p - 1))
            l0lost = !_sample_l0_survived(sel[start])
            push!(lost, l0lost)
            push!(w, l0lost ? cfg.pac_l0_lost_weight : cfg.pac_l0_kept_weight)
            start = p
        end
    end
    return (paths = sel, ranges = ranges, w = w, lost = lost)
end

# Upper bound on the trial-level gain any single column at target j can
# deliver: give every path its best-case Delta (fire only when helpful).
# Used to shortlist targets before running the flip search.
function _parity_target_upper_bound(data, j::Int, cfg::PipelineConfig)
    ub = 0.0
    thr = cfg.pac_parity_threshold
    @inbounds for (t, rg) in enumerate(data.ranges)
        ok = true
        for p in rg
            s = data.paths[p]
            elig = !cfg.pac_l0_aware_scoring ||
                   _sample_l0_survived(s) ||
                   j <= s.first_error + cfg.pac_l0_lost_window
            best_delta = 0.0
            if elig
                d = (1.0 - 2.0 * Float64(s.B[j])) * Float64(s.LLR[j])
                best_delta = max(0.0, d)
            end
            if Float64(s.D) + best_delta <= thr
                ok = false
                break
            end
        end
        ok && (ub += data.w[t])
    end
    return ub
end

# Replace column j of T (rows above the diagonal) with `sources`.
# Note: this also clears inert taps whose source row is frozen (v_r = 0 on
# every path, correct or wrong, so they never had any effect).
function _apply_column(T::AbstractMatrix{<:Integer}, j::Int,
                       sources::AbstractVector{<:Integer})
    T_try = copy(T)
    @inbounds for i in 1:(j - 1)
        T_try[i, j] = 0
    end
    for i in sources
        1 <= i < j || throw(ArgumentError("column source $i must precede target $j."))
        T_try[i, j] = 1
    end
    return T_try
end

"""
Local flip search over the source set of column j.

Pool = current (effective, info-row) sources of the column, plus the
highest-support info bits in span.  Capped at 64 so each path's pattern is
one UInt64 mask and a parity test is a single AND + count_ones.

Starts from the CURRENT column, so with pac_initial = :generic the search
seeds at the generic taps and flips can remove as well as add — pruning and
growing are the same move, which is what makes the two directions cost the
same number of committed steps.

Returns nothing when the pool is empty.
"""
function _parity_column_search(cfg::PipelineConfig, data,
                               A::AbstractVector{Int}, Aset::Set{Int},
                               T::AbstractMatrix{<:Integer}, j::Int)
    lo = max(1, j - cfg.pac_max_span)
    paths = data.paths
    np = length(paths)

    cur_sources = Int[i for i in lo:(j - 1) if in(i, Aset) && isodd(Int(T[i, j]))]
    curset = Set(cur_sources)
    # Live info taps below the span window are outside the design pool.
    # They are RETAINED verbatim in the redesigned column, and the flip
    # condition below is computed against the symmetric difference S ⊻ S0,
    # which is exact whether or not such taps exist.
    retained = Int[i for i in 1:(lo - 1) if in(i, Aset) && isodd(Int(T[i, j]))]

    # Rank in-span info bits by dangerous-path support.
    cand = Int[i for i in A if lo <= i < j]
    isempty(cand) && return nothing
    supp = Dict{Int, Int}()
    for i in cand
        c = 0
        @inbounds for s in paths
            c += Int(s.path_v_scl[i])
        end
        supp[i] = c
    end
    sort!(cand, by = i -> (-supp[i], i))

    cap = min(max(cfg.pac_parity_source_pool, 1), 64)
    pool = copy(cur_sources)                       # current sources always prunable
    for i in cand
        length(pool) >= cap && break
        (i in curset) && continue
        supp[i] > 0 || continue
        push!(pool, i)
    end
    length(pool) > 64 && (pool = pool[1:64])
    isempty(pool) && return nothing
    sort!(pool)
    npool = length(pool)

    # Per-path precomputation for O(1) parity evaluation.
    masks = Vector{UInt64}(undef, np)
    lam   = Vector{Float64}(undef, np)
    b0    = Vector{Int}(undef, np)
    Dv    = Vector{Float64}(undef, np)
    elig  = Vector{Bool}(undef, np)
    @inbounds for (p, s) in enumerate(paths)
        m = UInt64(0)
        for (k, i) in enumerate(pool)
            if s.path_v_scl[i] != 0
                m |= UInt64(1) << (k - 1)
            end
        end
        masks[p] = m
        lam[p]   = Float64(s.LLR[j])
        b0[p]    = Int(s.B[j])
        Dv[p]    = Float64(s.D)
        # L0-aware gating, matching the legacy edge scorer: on a lost-L0
        # trial a target past the urgency deadline can neither help nor,
        # by convention, harm (Delta treated as 0).
        elig[p]  = !cfg.pac_l0_aware_scoring ||
                   _sample_l0_survived(s) ||
                   j <= s.first_error + cfg.pac_l0_lost_window
    end

    S0 = UInt64(0)
    for (k, i) in enumerate(pool)
        (i in curset) && (S0 |= UInt64(1) << (k - 1))
    end

    thr = cfg.pac_parity_threshold
    ranges = data.ranges
    w = data.w

    # Trial-level saved weight for a hypothesized column mask.
    # Exact flip test: replacing the pool part of the column S0 -> S flips
    # u_j on path p iff parity(v_p over S ⊻ S0) is odd; retained
    # below-window taps cancel out of the difference.  When u_j flips, the
    # BM change is (1 - 2 b0) * LLR_j exactly (one branch of the max is 0).
    function col_score(Smask::UInt64)
        dmask = Smask ⊻ S0
        total = 0.0
        @inbounds for (t, rg) in enumerate(ranges)
            saved = true
            for p in rg
                Δ = 0.0
                if elig[p] && isodd(count_ones(masks[p] & dmask))
                    Δ = (1.0 - 2.0 * b0[p]) * lam[p]
                end
                if Dv[p] + Δ <= thr
                    saved = false
                    break
                end
            end
            saved && (total += w[t])
        end
        return total
    end

    base = col_score(S0)

    # Sparsity-aware search objective: each designed tap costs tap_penalty
    # (weighted trials), and cap_w > 0 hard-limits the designed column
    # weight.  With both at their defaults the objective is the raw score.
    cap_w = cfg.pac_parity_max_col_weight
    pen = cfg.pac_parity_tap_penalty
    obj(mask::UInt64) = col_score(mask) - pen * count_ones(mask)

    pool_mask = npool >= 64 ? typemax(UInt64) : (UInt64(1) << npool) - UInt64(1)

    # Clamp a start mask to the weight cap by clearing lowest set bits
    # (which taps survive is immaterial — starts are random restarts).
    function clamp_cap(mask::UInt64)
        cap_w <= 0 && return mask
        while count_ones(mask) > cap_w
            mask &= mask - UInt64(1)
        end
        return mask
    end

    # Best-improvement flip search from a given start mask.
    function run_flips(start::UInt64)
        mask = clamp_cap(start & pool_mask)
        sc = obj(mask)
        for _ in 1:cfg.pac_parity_max_flip_rounds
            flip = -1
            fbest = sc
            for k in 0:(npool - 1)
                nm = mask ⊻ (UInt64(1) << k)
                # Skip ON-flips that would exceed the weight cap.
                if cap_w > 0 && (nm & (UInt64(1) << k)) != 0 && count_ones(nm) > cap_w
                    continue
                end
                s2 = obj(nm)
                if s2 > fbest
                    fbest = s2
                    flip = k
                end
            end
            flip < 0 && break
            mask ⊻= UInt64(1) << flip
            sc = fbest
        end
        return mask, sc
    end

    # Multi-start: the current column plus deterministic pseudo-random masks
    # (splitmix64 keyed on seed and target).  Single-start flip search can
    # sit in a parity local maximum exactly the way add-only greedy does;
    # restarts are nearly free at this problem size.
    best_mask, best_obj = run_flips(S0)
    x = (UInt64(cfg.seed & typemax(Int32)) + UInt64(j)) * 0x9E3779B97F4A7C15
    for _ in 1:max(cfg.pac_parity_restarts, 0)
        x += 0x9E3779B97F4A7C15
        z = x
        z = (z ⊻ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ⊻ (z >> 27)) * 0x94D049BB133111EB
        z = z ⊻ (z >> 31)
        m, s = run_flips(z)
        if s > best_obj
            best_obj = s
            best_mask = m
        end
    end

    # Backward elimination on the RAW score: strip every tap whose removal
    # costs nothing.  The flip search only takes strict improvements, so it
    # cannot remove plateau taps (taps firing on canceling trial pairs, or
    # on none); leaving them in buys density for zero FER benefit — the
    # opposite of the sparse-precoder story.
    best = col_score(best_mask)
    changed = true
    while changed
        changed = false
        for k in 0:(npool - 1)
            bit = UInt64(1) << k
            (best_mask & bit) == 0 && continue
            if col_score(best_mask ⊻ bit) >= best
                best_mask ⊻= bit
                best = col_score(best_mask)
                changed = true
            end
        end
    end

    designed = Int[pool[k] for k in 1:npool if ((best_mask >> (k - 1)) & 1) == 1]
    sources  = sort(vcat(retained, designed))
    cur_full = Set(vcat(retained, cur_sources))
    added    = sort(collect(setdiff(Set(sources), cur_full)))
    removed  = sort(collect(setdiff(cur_full, Set(sources))))

    return (j = j, sources = sources, added = added, removed = removed,
            gain = best - base, base = base, surrogate = best,
            pool_size = npool)
end

function _replay_screen(cfg::PipelineConfig, A::AbstractVector{Int},
                        T_try::AbstractMatrix{<:Integer}, buffer)
    cfg_scl = DPS.make_config(cfg.N, sort(Int.(A)), cfg.L, T_try;
                              depths = DPS.infer_depths(T_try))
    out = DPS.replay_decode(cfg_scl, buffer)
    return (n_errors = out.n_errors,
            fer_est = out.n_errors / max(buffer.n_source_trials, 1))
end

# One pass of replay-verified column clearing.  Sequential: each accepted
# clear updates the working T and the acceptance base, so with slack = 0 the
# buffer error count is monotone non-increasing over the pass.  Info-target
# columns are included: the replay decides whether generic's info->info taps
# are load-bearing, which no dangerous-sample surrogate can.
function _parity_bulk_prune(cfg::PipelineConfig, A::Vector{Int},
                            T::AbstractMatrix{<:Integer}, buffer)
    Aset = Set(A)
    base = _replay_screen(cfg, A, T, buffer).n_errors
    cfg.verbose && @printf("Bulk prune: baseline replay errors = %d over %d buffered trials\n",
                           base, length(buffer.trial_ids))
    T_work = copy(T)
    cleared = Int[]
    for j in 2:cfg.N
        has_live = any(isodd(Int(T_work[i, j])) && in(i, Aset) for i in 1:(j - 1))
        has_live || continue
        T_try = _apply_column(T_work, j, Int[])
        n_err = _replay_screen(cfg, A, T_try, buffer).n_errors
        if n_err <= base + cfg.pac_parity_bulk_prune_slack
            cfg.verbose && @printf("  clear col %3d : replay errors %d -> %d  (accepted)\n",
                                   j, base, n_err)
            T_work = T_try
            base = n_err
            push!(cleared, j)
        end
    end
    return (T = T_work, cleared = cleared, replay_errors = base)
end

function run_iterative_pac_design_parity(cfg::PipelineConfig, set_out)
    A = Vector{Int}(set_out.A)
    N = cfg.N
    Aset = Set(A)
    T = Matrix{Int}(_initial_stage2_tmat(cfg, set_out))

    cfg.verbose && begin
        println("\n=== Stage 2 (parity): per-target column design ===")
        println("selected A                   = ", A)
        println("pac_initial                  = ", cfg.pac_initial)
        println("pac_replay_screen            = ", cfg.pac_replay_screen,
                "  (delta = ", cfg.pac_replay_delta, ")")
        println("pac_parity_max_steps         = ", cfg.pac_parity_max_steps)
        println("pac_parity_source_pool       = ", min(cfg.pac_parity_source_pool, 64))
        println("pac_parity_targets_screened  = ", cfg.pac_parity_targets_screened)
        println("pac_parity_candidates        = ", cfg.pac_parity_candidates_per_step)
        println("pac_parity_full_confirm      = ", cfg.pac_parity_full_confirm)
        println("pac_parity_min_trial_gain    = ", cfg.pac_parity_min_trial_gain)
        println("pac_parity_bulk_prune        = ", cfg.pac_parity_bulk_prune)
        println("pac_l0_aware_scoring         = ", cfg.pac_l0_aware_scoring,
                "  lost_window = ", cfg.pac_l0_lost_window)
        cfg.pac_target != :frozen &&
            println("note: :parity mode designs frozen-target columns only; pac_target=:all is ignored.")
    end

    res = spectrum_for_A(cfg, A, T;
                         n_trials = cfg.pac_n_trials_full,
                         seed = cfg.seed + 7,
                         label = "stage-2 (parity) baseline",
                         collect_replay = cfg.pac_replay_screen)
    score = tail_score(res, cfg)
    buffer = res.replay

    cfg.verbose && @printf("\nStage-2 baseline objective (%s) = %.8e\n",
                           String(cfg.set_objective), score)
    cfg.verbose && buffer !== nothing &&
        @printf("Replay buffer: %d of %d trials retained (margin <= %.2f)\n",
                length(buffer.trial_ids), buffer.n_source_trials, buffer.delta)

    edges_log = NamedTuple[]
    step0 = 0

    # ── optional one-pass bulk prune (the :generic entry point) ──
    if cfg.pac_parity_bulk_prune
        if buffer === nothing
            cfg.verbose && println("pac_parity_bulk_prune requires pac_replay_screen = true; skipping.")
        else
            bp = _parity_bulk_prune(cfg, A, T, buffer)
            if isempty(bp.cleared)
                cfg.verbose && println("Bulk prune: no clearable columns at the replay slack.")
            else
                res_try = spectrum_for_A(cfg, A, bp.T;
                                         n_trials = cfg.pac_n_trials_full,
                                         seed = cfg.seed + 77,
                                         label = "bulk-prune confirm",
                                         collect_replay = cfg.pac_replay_screen)
                score_try = tail_score(res_try, cfg)
                if score_try <= score * (1.0 + cfg.pac_parity_bulk_prune_rel_slack)
                    cfg.verbose && @printf("Bulk prune KEPT: cleared %d column(s) %s  full %.8e -> %.8e\n",
                                           length(bp.cleared), string(bp.cleared), score, score_try)
                    step0 += 1
                    push!(edges_log, (
                        step = step0, move_type = :bulk_prune,
                        kicker_size = length(bp.cleared),
                        edges = [(i = 0, j = j) for j in bp.cleared],
                        saved_paths_estimate = 0.0,
                        saved_kept_l0 = 0.0, saved_lost_l0_urgent = 0.0,
                        saved_lost_l0_late = 0.0,
                        small_score = bp.replay_errors / max(buffer.n_source_trials, 1),
                        full_score = score_try, working_score = score_try,
                        best_score = min(score, score_try)))
                    T = Matrix{Int}(bp.T)
                    res = res_try
                    score = score_try
                    buffer = res_try.replay
                else
                    cfg.verbose && @printf("Bulk prune REVERTED at full budget: %.8e vs baseline %.8e\n",
                                           score_try, score)
                end
            end
        end
    end

    best_T = copy(T)
    best_res = res
    best_score = score
    best_step = step0

    for step in (step0 + 1):(step0 + cfg.pac_parity_max_steps)
        cfg.verbose && @printf("\n--- Column step %d  (working %.8e, best %.8e, dangerous events = %d) ---\n",
                               step, score, best_score, length(res.events))

        if length(res.events) < cfg.pac_min_events_for_edges
            cfg.verbose && @printf("Too few dangerous events (%d < %d) to rank columns; stopping.\n",
                                   length(res.events), cfg.pac_min_events_for_edges)
            break
        end

        data = _parity_trial_data(res, cfg)
        if data === nothing || isempty(data.ranges)
            cfg.verbose && println("No dangerous trials to score; stopping.")
            break
        end

        # Shortlist frozen targets by the single-column upper bound.
        targets = Int[]
        ubs = Float64[]
        for j in 2:N
            in(j, Aset) && continue
            ub = _parity_target_upper_bound(data, j, cfg)
            if ub > 0.0
                push!(targets, j)
                push!(ubs, ub)
            end
        end
        if isempty(targets)
            cfg.verbose && println("No frozen target with positive upper bound; stopping.")
            break
        end
        ord = sortperm(ubs, rev = true)
        targets = targets[ord]
        targets = first(targets, min(cfg.pac_parity_targets_screened, length(targets)))

        # Flip search per shortlisted target.
        cands = NamedTuple[]
        for j in targets
            c = _parity_column_search(cfg, data, A, Aset, T, j)
            c === nothing && continue
            c.gain >= cfg.pac_parity_min_trial_gain || continue
            push!(cands, c)
        end
        if isempty(cands)
            cfg.verbose && @printf("No column with surrogate trial-gain >= %.1f; stopping.\n",
                                   cfg.pac_parity_min_trial_gain)
            break
        end
        sort!(cands, by = c -> -c.gain)

        cfg.verbose && begin
            println("Top column candidates over $(length(data.ranges)) dangerous trials [surrogate trial-level gain]:")
            for c in first(cands, min(12, length(cands)))
                @printf("  col %3d  gain=%.1f  |S|=%d  add=%s  remove=%s\n",
                        c.j, c.gain, length(c.sources), string(c.added), string(c.removed))
            end
        end

        cands = first(cands, min(cfg.pac_parity_candidates_per_step, length(cands)))

        # Replay screen (exact, paired, two-sided).
        screened = NamedTuple[]
        if cfg.pac_replay_screen && buffer !== nothing
            base_err = count(m -> m <= 0.0, buffer.margins)
            for c in cands
                T_try = _apply_column(T, c.j, c.sources)
                r = _replay_screen(cfg, A, T_try, buffer)
                push!(screened, merge(c, (replay_errors = r.n_errors,
                                          replay_fer = r.fer_est)))
                cfg.verbose && @printf("  replay col %3d : buffer errors %d -> %d  (fer_est %.4e)\n",
                                       c.j, base_err, r.n_errors, r.fer_est)
            end
            sort!(screened, by = c -> (c.replay_errors, -c.gain))
            screened = [c for c in screened if c.replay_errors < base_err]
            if isempty(screened)
                cfg.verbose && println("Replay screen: no candidate reduces buffer errors; stopping.")
                break
            end
        else
            screened = [merge(c, (replay_errors = -1, replay_fer = NaN)) for c in cands]
        end

        confirm_pool = first(screened, min(cfg.pac_parity_full_confirm, length(screened)))

        # Fresh-MC confirmation of the shortlist.
        full_evals = NamedTuple[]
        for (cidx, c) in enumerate(confirm_pool)
            T_try = _apply_column(T, c.j, c.sources)
            label = cfg.verbose ?
                    @sprintf("confirm step=%d confirm=%d col=%d", step, cidx, c.j) : ""
            res_try = spectrum_for_A(cfg, A, T_try;
                                     n_trials = cfg.pac_n_trials_full,
                                     seed = cfg.seed + 10_000 * step + cidx,
                                     label = label,
                                     collect_replay = cfg.pac_replay_screen)
            sc = tail_score(res_try, cfg)
            cfg.verbose && @printf("    full   (%dk trials)  col=%d sources=%s : %.8e (prev %.8e, replay_fer %s)\n",
                                   cfg.pac_n_trials_full ÷ 1000, c.j, string(c.sources),
                                   sc, score,
                                   isnan(c.replay_fer) ? "n/a" : @sprintf("%.4e", c.replay_fer))
            push!(full_evals, (cand = c, res = res_try, score = sc, T_try = T_try))
        end
        sort!(full_evals, by = e -> e.score)
        best_full = first(full_evals)

        if !_pac_is_improvement(cfg, best_full.score, score)
            cfg.verbose && @printf("No confirmed improving column at step %d (best %.8e vs %.8e); stopping.\n",
                                   step, best_full.score, score)
            break
        end

        c = best_full.cand
        T = Matrix{Int}(best_full.T_try)
        res = best_full.res
        score = best_full.score
        buffer = res.replay !== nothing ? res.replay : buffer

        if score < best_score
            best_T = copy(T)
            best_res = res
            best_score = score
            best_step = step
        end

        push!(edges_log, (
            step = step, move_type = :column, kicker_size = length(c.sources),
            edges = [(i = i, j = c.j) for i in c.sources],
            saved_paths_estimate = Float64(c.gain),
            saved_kept_l0 = 0.0, saved_lost_l0_urgent = 0.0, saved_lost_l0_late = 0.0,
            small_score = isnan(c.replay_fer) ? NaN : c.replay_fer,
            full_score = score, working_score = score, best_score = best_score))

        cfg.verbose && @printf("ACCEPTED step %d  col=%d  sources=%s  (add %s, remove %s)  working -> %.8e  best -> %.8e\n",
                               step, c.j, string(c.sources), string(c.added),
                               string(c.removed), score, best_score)
    end

    T_return = cfg.pac_return_best_seen ? best_T : T
    res_return = cfg.pac_return_best_seen ? best_res : res
    score_return = cfg.pac_return_best_seen ? best_score : score

    cfg.verbose && begin
        println("\n--- Stage 2 (parity) summary ---")
        @printf("column moves         = %d\n", length(edges_log))
        @printf("working objective    = %.8e\n", score)
        @printf("best objective       = %.8e at step %d\n", best_score, best_step)
        @printf("returned objective   = %.8e\n", score_return)
        if !isempty(edges_log)
            println("column-move history:")
            for e in edges_log
                edge_repr = join(["$(es.i)->$(es.j)" for es in e.edges], ",")
                @printf("  step %2d  move=%s  |S|=%d  %s  full=%.4e  best=%.4e\n",
                        e.step, string(e.move_type), e.kicker_size, edge_repr,
                        e.full_score, e.best_score)
            end
        end
    end

    return (Tmat = Int.(T_return),
            Tmat_working = Int.(T),
            depths = DPS.infer_depths(T_return),
            edges = edges_log,
            result = res_return,
            score = score_return,
            working_result = res,
            working_score = score,
            best_step = best_step,
            explore_used = 0)
end

function run_iterative_pac_design(cfg::PipelineConfig, set_out)
    A = set_out.A
    N = cfg.N
    T = _initial_stage2_tmat(cfg, set_out)

    cfg.verbose && begin
        println("\n=== Stage 2: iterative PAC precoder construction ===")
        println("selected A                        = ", A)
        println("pac_initial                       = ", cfg.pac_initial)
        println("pac_max_edges                     = ", cfg.pac_max_edges)
        println("pac_max_span                      = ", cfg.pac_max_span)
        println("pac_target                        = ", cfg.pac_target)
        println("pac_kicker_sizes                  = ", cfg.pac_kicker_sizes)
        println("pac_top_edges_per_step            = ", cfg.pac_top_edges_per_step)
        println("pac_kicker_single_pool            = ", cfg.pac_kicker_single_pool)
        println("pac_candidate_sets_per_step       = ", cfg.pac_candidate_sets_per_step)
        println("pac_full_confirm_top_candidates   = ", cfg.pac_full_confirm_top_candidates)
        println("pac_explore_when_stalled          = ", cfg.pac_explore_when_stalled)
        println("pac_return_best_seen              = ", cfg.pac_return_best_seen)
        println("pac_l0_aware_scoring              = ", cfg.pac_l0_aware_scoring)
        println("pac_l0_lost_window                = ", cfg.pac_l0_lost_window)
    end

    res = spectrum_for_A(cfg, A, T;
                         n_trials = cfg.pac_n_trials_full,
                         seed = cfg.seed + 7,
                         label = "stage-2 baseline",
                         collect_replay = cfg.pac_replay_screen)
    score = tail_score(res, cfg)
    buffer = res.replay

    cfg.verbose && buffer !== nothing &&
        @printf("Replay buffer: %d of %d trials retained (margin <= %.2f)\n",
                length(buffer.trial_ids), buffer.n_source_trials, buffer.delta)

    best_T = copy(T)
    best_res = res
    best_score = score
    best_step = 0
    explore_used = 0

    cfg.verbose && @printf("\nStage-2 baseline objective (%s) = %.8e\n",
                           String(cfg.set_objective), score)

    edges_log = NamedTuple[]

    for step in 1:cfg.pac_max_edges
        cfg.verbose && @printf("\n--- Edge step %d  (working %.8e, best %.8e, dangerous events = %d) ---\n",
                               step, score, best_score, length(res.events))

        if length(res.events) < cfg.pac_min_events_for_edges
            cfg.verbose && @printf("Too few dangerous events (%d < %d) to rank edges; stopping.\n",
                                   length(res.events), cfg.pac_min_events_for_edges)
            break
        end

        edge_scorer = cfg.pac_edge_metric == :trials ?
                      _score_single_edges_trialwise : _score_single_edges
        single = edge_scorer(res, T, N, A;
                             max_span = cfg.pac_max_span,
                             target = cfg.pac_target,
                             only_dangerous = cfg.pac_only_dangerous,
                             l0_aware = cfg.pac_l0_aware_scoring,
                             lost_window = cfg.pac_l0_lost_window,
                             kept_weight = cfg.pac_l0_kept_weight,
                             lost_weight = cfg.pac_l0_lost_weight,
                             late_penalty = cfg.pac_l0_late_penalty)
        saved = single.score

        # σ_source at every bit (computed from the current res under the
        # current T).  Used by :source_first ranking; also surfaced as a
        # diagnostic column in :saved_paths mode so the source-firing rate is
        # visible per candidate edge.
        n_trials_for_sigma = _stage1_n_trials(res, cfg.pac_n_trials_full)
        support_for_sigma = _stage1_support_vectors(res, N, n_trials_for_sigma)
        sigma_source = support_for_sigma.source_sigma

        single_cands = NamedTuple[]
        Aset = Set(Int.(A))

        if cfg.pac_edge_ranking == :source_first
            # Source-first: for each info source i, find argmax_j saved[i, j]
            # subject to the min-saved floor; rank pairs by σ_source descending.
            for i in A
                sigma_i = Float64(sigma_source[i])
                sigma_i > 0.0 || continue
                best_j = 0
                best_saved = 0.0
                j_max = min(N, i + cfg.pac_max_span)
                for j in (i + 1):j_max
                    if cfg.pac_target == :frozen
                        in(j, Aset) && continue
                    elseif cfg.pac_target == :all
                        # allow info -> info too
                    else
                        throw(ArgumentError("pac_target must be :frozen or :all."))
                    end
                    T[i, j] != 0 && continue
                    sv = Float64(saved[i, j])
                    if sv > best_saved
                        best_saved = sv
                        best_j = j
                    end
                end
                best_j > 0 || continue
                best_saved >= Float64(cfg.pac_source_first_min_saved) || continue
                push!(single_cands, (
                    i = Int(i), j = Int(best_j),
                    source_sigma = sigma_i,
                    saved_paths = best_saved,
                    saved_kept = Float64(single.saved_kept[i, best_j]),
                    saved_lost_urgent = Float64(single.saved_lost_urgent[i, best_j]),
                    saved_lost_late = Float64(single.saved_lost_late[i, best_j])))
            end
        elseif cfg.pac_edge_ranking == :saved_paths
            for i in A
                j_max = min(N, i + cfg.pac_max_span)
                for j in (i + 1):j_max
                    if cfg.pac_target == :frozen
                        in(j, Aset) && continue
                    elseif cfg.pac_target == :all
                        # allow info -> info too
                    else
                        throw(ArgumentError("pac_target must be :frozen or :all."))
                    end
                    T[i, j] != 0 && continue
                    saved[i, j] > 0 || continue
                    push!(single_cands, (i = Int(i), j = Int(j),
                                         source_sigma = Float64(sigma_source[i]),
                                         saved_paths = Float64(saved[i, j]),
                                         saved_kept = Float64(single.saved_kept[i, j]),
                                         saved_lost_urgent = Float64(single.saved_lost_urgent[i, j]),
                                         saved_lost_late = Float64(single.saved_lost_late[i, j])))
                end
            end
        else
            throw(ArgumentError("pac_edge_ranking must be :saved_paths or :source_first."))
        end

        if isempty(single_cands)
            cfg.verbose && println("No single-edge candidates with positive saved-paths; stopping.")
            break
        end

        if cfg.pac_edge_ranking == :source_first
            # Primary: σ_source descending → "fix the most-firing info bit first."
            # Tie-break: saved-paths, then kept, then urgent.
            sort!(single_cands, by = c -> (-c.source_sigma, -c.saved_paths,
                                            -c.saved_kept, -c.saved_lost_urgent,
                                            c.saved_lost_late))
        else
            sort!(single_cands, by = c -> (-c.saved_paths, -c.saved_kept,
                                            -c.saved_lost_urgent, c.saved_lost_late))
        end
        top_single_print = first(single_cands, min(cfg.pac_top_edges_per_step, length(single_cands)))
        top_single_pool  = first(single_cands, min(max(cfg.pac_top_edges_per_step,
                                                       cfg.pac_kicker_single_pool),
                                                   length(single_cands)))

        cfg.verbose && begin
            ranking_tag = cfg.pac_edge_ranking == :source_first ?
                          "source_first: rank by σ_source descending, best target per source" :
                          "saved_paths: global rank by saved-paths count"
            println("Top single-edge candidates [", ranking_tag, "]")
            println("  L0-aware over $(single.n_paths_considered) dangerous paths; kept=$(single.n_l0_kept), lost=$(single.n_l0_lost)")
            for c in top_single_print
                @printf("  (%3d -> %3d)  σ_src=%.4f  score=%.0f  kept=%.0f  lost_urgent=%.0f  lost_late=%.0f\n",
                        c.i, c.j, c.source_sigma, c.saved_paths,
                        c.saved_kept, c.saved_lost_urgent, c.saved_lost_late)
            end
        end

        candidate_sets = NamedTuple[]
        for kicker_size in cfg.pac_kicker_sizes
            kicker_size <= 0 && continue
            sets_k = _score_k_edge_sets(res, top_single_pool;
                                        kicker_size = kicker_size,
                                        only_dangerous = cfg.pac_only_dangerous,
                                        max_sets = cfg.pac_kicker_max_candidates,
                                        l0_aware = cfg.pac_l0_aware_scoring,
                                        lost_window = cfg.pac_l0_lost_window,
                                        kept_weight = cfg.pac_l0_kept_weight,
                                        lost_weight = cfg.pac_l0_lost_weight,
                                        late_penalty = cfg.pac_l0_late_penalty,
                                        N = N,
                                        metric = cfg.pac_edge_metric)
            append!(candidate_sets, sets_k)
        end

        if isempty(candidate_sets)
            cfg.verbose && println("No candidate edge sets were generated; stopping.")
            break
        end

        sort!(candidate_sets, by = c -> (-c.score, -c.saved_kept, -c.saved_lost_urgent, c.saved_lost_late, c.kicker_size, -c.saved_excess))
        candidate_sets = first(candidate_sets, min(cfg.pac_candidate_sets_per_step, length(candidate_sets)))

        cfg.verbose && begin
            println("Candidate edge sets in this non-greedy batch:")
            nshow = min(12, length(candidate_sets))
            for idx in 1:nshow
                c = candidate_sets[idx]
                @printf("  batch %3d  sz=%d  edges=%s  score=%.0f  kept=%.0f  lost_urgent=%.0f  lost_late=%.0f  excess=%.0f\n",
                        idx, c.kicker_size, _edge_repr(c.edges), c.saved_paths,
                        c.saved_kept, c.saved_lost_urgent, c.saved_lost_late, c.saved_excess)
            end
            length(candidate_sets) > nshow && @printf("  ... %d more candidate sets\n", length(candidate_sets) - nshow)
        end

        confirm_pool = NamedTuple[]
        if cfg.pac_replay_screen && buffer !== nothing
            # Replay screen: exact SCL re-decode of the stored error +
            # near-miss trials under each candidate T.  Replaces the 50k
            # fresh-MC small screens: paired (same noise for every
            # candidate), unbiased under the new T (wrong paths re-optimize,
            # so the counterfactual's optimism is priced out here), and it
            # sees candidates that BREAK currently-safe near-miss trials.
            base_err = count(m -> m <= 0.0, buffer.margins)
            screen_evals = NamedTuple[]
            for (k_idx, cset) in enumerate(candidate_sets)
                T_try = _apply_edges(T, cset.edges)
                r = _replay_screen(cfg, A, T_try, buffer)
                cfg.verbose && @printf("    replay sz=%d edges=%s : buffer errors %d -> %d  (fer_est %.8e)\n",
                                       cset.kicker_size, _edge_repr(cset.edges),
                                       base_err, r.n_errors, r.fer_est)
                push!(screen_evals, (cset = cset, score = r.fer_est,
                                     replay_errors = r.n_errors,
                                     edge_repr = _edge_repr(cset.edges),
                                     small_rank = k_idx))
            end
            sort!(screen_evals, by = e -> (e.replay_errors, -e.cset.score))
            confirm_pool = [e for e in screen_evals if e.replay_errors < base_err]

            if isempty(confirm_pool) && cfg.pac_confirm_best_even_if_screen_fails &&
               !isempty(screen_evals)
                confirm_pool = first(screen_evals, 1)
                cfg.verbose && println("No candidate reduces replay errors; force-confirming the best.")
            end
            if isempty(confirm_pool)
                cfg.verbose && println("Replay screen: no candidate reduces buffer errors; stopping.")
                break
            end
        else
            small_evals = NamedTuple[]
            for (k_idx, cset) in enumerate(candidate_sets)
                push!(small_evals, _evaluate_edge_set_small(cfg, A, T, cset, step, k_idx))
            end
            sort!(small_evals, by = e -> e.score)

            confirm_limit = _pac_confirm_limit(cfg, score, best_score)
            confirm_pool = [e for e in small_evals if e.score <= confirm_limit]

            if isempty(confirm_pool) && cfg.pac_confirm_best_even_if_screen_fails
                nforce = min(1, length(small_evals))
                confirm_pool = first(small_evals, nforce)
                cfg.verbose && @printf("No small-screen pass under %.8e; force-confirming the best small candidate.\n",
                                       confirm_limit)
            end

            if isempty(confirm_pool)
                cfg.verbose && @printf("No candidate passed the small screen under %.8e; stopping.\n",
                                       confirm_limit)
                break
            end
        end

        confirm_pool = first(confirm_pool, min(cfg.pac_full_confirm_top_candidates,
                                               length(confirm_pool)))

        cfg.verbose && begin
            screen_tag = (cfg.pac_replay_screen && buffer !== nothing) ?
                         "replay" : "small-MC"
            @printf("Confirming %d candidate sets with full trials (%s screen)\n",
                    length(confirm_pool), screen_tag)
            for (idx, e) in enumerate(confirm_pool)
                @printf("  confirm pool %2d  sz=%d  edges=%s  screen=%.8e  score=%.0f  kept=%.0f  lost_urgent=%.0f  lost_late=%.0f\n",
                        idx, e.cset.kicker_size, e.edge_repr, e.score, e.cset.saved_paths,
                        e.cset.saved_kept, e.cset.saved_lost_urgent, e.cset.saved_lost_late)
            end
        end

        full_evals = NamedTuple[]
        for (cidx, small_eval) in enumerate(confirm_pool)
            push!(full_evals, _evaluate_edge_set_full(cfg, A, T, small_eval, score, step, cidx))
        end
        sort!(full_evals, by = e -> e.score)

        best_full = first(full_evals)
        move_type = :reject
        if _pac_is_improvement(cfg, best_full.score, score)
            move_type = :improve
        elseif _pac_explore_allowed(cfg, best_full.score, score, best_score, explore_used)
            move_type = :explore
        end

        if move_type == :reject
            cfg.verbose && begin
                @printf("No confirmed improving move at step %d. Best full candidate was sz=%d edges=%s score %.8e.\n",
                        step, best_full.cset.kicker_size, best_full.edge_repr, best_full.score)
                @printf("Exploration limit from current %.8e, from best %.8e, explore_used %d/%d.\n",
                        _pac_explore_limit_from_current(cfg, score),
                        _pac_explore_limit_from_best(cfg, best_score),
                        explore_used, cfg.pac_max_explore_steps)
            end
            break
        end

        cset = best_full.cset
        for e in cset.edges
            T[e.i, e.j] = 1
        end
        res = best_full.res
        score = best_full.score
        res.replay !== nothing && (buffer = res.replay)
        if move_type == :explore
            explore_used += 1
        end

        if score < best_score
            best_T = copy(T)
            best_res = res
            best_score = score
            best_step = step
        end

        push!(edges_log, (
            step                 = step,
            move_type            = move_type,
            kicker_size          = cset.kicker_size,
            edges                = [(i = e.i, j = e.j) for e in cset.edges],
            saved_paths_estimate = Float64(cset.saved_paths),
            saved_kept_l0        = Float64(cset.saved_kept),
            saved_lost_l0_urgent = Float64(cset.saved_lost_urgent),
            saved_lost_l0_late   = Float64(cset.saved_lost_late),
            small_score          = best_full.score_small,
            full_score           = best_full.score,
            working_score        = score,
            best_score           = best_score,
        ))

        cfg.verbose && @printf("ACCEPTED step %d  move=%s  sz=%d  edges=%s  working -> %.8e  best -> %.8e\n",
                               step, string(move_type), cset.kicker_size,
                               best_full.edge_repr, score, best_score)
    end

    T_return = cfg.pac_return_best_seen ? best_T : T
    res_return = cfg.pac_return_best_seen ? best_res : res
    score_return = cfg.pac_return_best_seen ? best_score : score

    cfg.verbose && begin
        println("\n--- Stage 2 summary ---")
        @printf("edge-add steps       = %d\n", length(edges_log))
        @printf("working objective    = %.8e\n", score)
        @printf("best objective       = %.8e at step %d\n", best_score, best_step)
        @printf("returned objective   = %.8e\n", score_return)
        @printf("exploratory moves    = %d\n", explore_used)
        if !isempty(edges_log)
            println("edge-add history:")
            for e in edges_log
                edge_repr = join(["$(es.i)->$(es.j)" for es in e.edges], ",")
                @printf("  step %2d  move=%s  sz=%d  edges=%s  full=%.4e  working=%.4e  best=%.4e  kept=%.0f lost_urgent=%.0f late=%.0f\n",
                         e.step, string(e.move_type), e.kicker_size, edge_repr,
                         e.full_score, e.working_score, e.best_score,
                         e.saved_kept_l0, e.saved_lost_l0_urgent, e.saved_lost_l0_late)
            end
        end
    end

    return (Tmat = Int.(T_return),
            Tmat_working = Int.(T),
            depths = DPS.infer_depths(T_return),
            edges = edges_log,
            result = res_return,
            score = score_return,
            working_result = res,
            working_score = score,
            best_step = best_step,
            explore_used = explore_used)
end

# ============================================================
# Stage 3 (optional): joint refinement on (A, T) starting from the
# post-Stage-2 state.  Each iteration proposes a small set of "joint
# moves" — a single info-set swap (j_remove → i_add) optionally
# accompanied by one or more PAC edges sourced from the newly promoted
# info bit i.  Candidates are scored by a per-trial counterfactual
# (joint saved-paths) computed directly from the existing dangerous
# samples.  Small-screen + full-eval the top-k, commit the best, repeat.
#
# The motivation:
#  * Stage 1 picks A under a fixed Tgeneric.  It cannot see "promote i,
#    AND add edge (i, k)" because the edge does not exist when Stage 1
#    evaluates i's promotion.
#  * Stage 2 picks edges under a fixed A.  It cannot see "add edge (i, k),
#    AND swap j → i" because i is not in A when Stage 2 evaluates edges
#    sourced from i.
#  * A joint Stage 3 move proposes the swap and the edge(s) at once.
# ============================================================

# ──────────────────────────────────────────────────────────────────────
# Joint per-trial counterfactual score.  Returns (saved, broken, score)
# where score = saved − broken.
#
#   freeze_gain(t, j)  = path_v_scl[j, t] · |LLR_j^t|
#       (BM the wrong path would pay when v_j is forced from 1 to 0)
#   promote_cost(t, i) = max(0, −LLR_i^t)
#       (BM the wrong path saves when it can pick v_i = 1 against
#        a wrong-signed LLR instead of being forced to 0)
#   edge_gain(t, i→k)  = [LLR_i^t < 0] · (1 − 2 · B[k, t]) · LLR_k^t
#       (PM change at frozen k when the edge fires; "v_i^wrong = 1"
#        is approximated by LLR_i^t < 0 — the SCL fork would pick the
#        LLR-favored side at the freshly-promoted i.)
#
#   ΔD(t, m) = freeze_gain − promote_cost + Σ_e edge_gain
#   saved(m)  = #{ t : D(t) ≤ 0 AND D(t) + ΔD(t, m) > 0 }
#   broken(m) = #{ t : D(t) > 0 AND D(t) + ΔD(t, m) ≤ 0 }
#
# Notes:
#  * j or i may be 0 to indicate "no remove" or "no promote" — used by
#    edge-only moves on existing info bits.
#  * edge_targets must satisfy i > 0 (we only propose edges sourced from
#    the newly promoted bit in this implementation).  For edge-only moves
#    on existing info bits, the source position is passed via a separate
#    parameter `edge_source_existing`.
# ──────────────────────────────────────────────────────────────────────
function _joint_move_saved_paths(res, N::Int,
                                  j_remove::Int, i_add::Int,
                                  edge_source_existing::Int,
                                  edge_targets::Vector{Int};
                                  only_dangerous::Bool = false)
    samples = _stage1_explicit_samples(res)
    saved    = 0
    broken   = 0
    n_dang   = 0
    n_safe   = 0
    delta_acc_dang = 0.0   # for diagnostics
    @inbounds for s in samples
        if hasproperty(s, :mode) && getproperty(s, :mode) !== :explicit
            continue
        end
        D = hasproperty(s, :D) ? Float64(getproperty(s, :D)) : 0.0
        if only_dangerous && D > 0.0
            continue
        end
        path_v = _first_existing_vector(s, (:path_v_scl, :path_v, :v, :V))
        Bvec   = _first_existing_vector(s, (:B, :path_x_scl, :path_x, :x, :X))
        LLR    = _first_existing_vector(s, (:LLR, :llr, :path_LLR, :path_llr))
        (path_v === nothing || LLR === nothing) && continue

        ΔD = 0.0

        # Freeze gain at j (if any).
        if j_remove > 0 && j_remove <= length(path_v) && j_remove <= length(LLR)
            if _bit01(path_v, j_remove) == 1
                ΔD += abs(Float64(LLR[j_remove]))
            end
        end

        # Promote cost at i (if any).
        i_for_edges = 0
        if i_add > 0 && i_add <= length(LLR)
            llr_i = Float64(LLR[i_add])
            ΔD -= max(0.0, -llr_i)
            i_for_edges = i_add
        elseif edge_source_existing > 0
            # Edge-only move on existing info bit: no promote cost.
            i_for_edges = edge_source_existing
        end

        # Edge gains.  Fire iff wrong-path v_i = 1 in the *hypothesized*
        # post-move state.  For a newly promoted i this is approximated
        # by LLR_i^t < 0.  For an existing info bit i, the wrong path's
        # v_i is directly observable via path_v[i].
        if !isempty(edge_targets) && i_for_edges > 0 && Bvec !== nothing
            edge_fires = false
            if i_add > 0
                # Newly promoted: heuristic on LLR sign.
                edge_fires = i_for_edges <= length(LLR) &&
                             Float64(LLR[i_for_edges]) < 0.0
            else
                # Existing info bit: observed wrong-path value.
                edge_fires = i_for_edges <= length(path_v) &&
                             _bit01(path_v, i_for_edges) == 1
            end
            if edge_fires
                @inbounds for k in edge_targets
                    (k <= 0 || k > length(Bvec) || k > length(LLR)) && continue
                    bk = _bit01(Bvec, k)
                    ΔD += (1.0 - 2.0 * Float64(bk)) * Float64(LLR[k])
                end
            end
        end

        if D <= 0.0
            n_dang += 1
            delta_acc_dang += ΔD
            if D + ΔD > 0.0
                saved += 1
            end
        else
            n_safe += 1
            if D + ΔD <= 0.0
                broken += 1
            end
        end
    end

    return (saved = saved, broken = broken, score = saved - broken,
            n_dangerous = n_dang, n_safe = n_safe,
            mean_delta_dangerous = n_dang > 0 ? delta_acc_dang / n_dang : 0.0)
end

# ──────────────────────────────────────────────────────────────────────
# Apply a Stage 3 joint move to (A, T) → returns (A_new, T_new).
# Removes auto-drop edges sourced from the removed bit.
# ──────────────────────────────────────────────────────────────────────
function _stage3_apply_move(N::Int, A::Vector{Int}, T::AbstractMatrix,
                              move::NamedTuple)
    A_new = copy(Int.(A))
    T_new = Matrix{UInt8}(T)

    for j in move.removes
        idx = findfirst(==(j), A_new)
        idx === nothing || deleteat!(A_new, idx)
        @inbounds for k in (j + 1):N
            T_new[j, k] = 0x00
        end
    end

    for i in move.adds
        push!(A_new, i)
    end
    sort!(A_new)
    Aset_new = Set(A_new)

    for (s, t) in move.edges_removed
        T_new[s, t] = 0x00
    end

    for (s, t) in move.edges_added
        s in Aset_new ||
            throw(ArgumentError("apply_move: edge source $s not in new info set."))
        s < t ||
            throw(ArgumentError("apply_move: edge $s→$t violates s < t."))
        T_new[s, t] = 0x01
    end

    return A_new, T_new
end

# ──────────────────────────────────────────────────────────────────────
# Generate Stage 3 candidate moves.  Reuses the Stage 1 priority ranking
# for remove candidates and the add ranking (DEGA + post-σ filter +
# add_strategy) for promotion candidates.  For each (j, i) pair,
# optionally enumerate amplifier edges from the new info bit i.  Also
# (if enabled) propose edge-only candidates on existing info bits.
# ──────────────────────────────────────────────────────────────────────
function _stage3_generate_candidates(cfg::PipelineConfig,
                                       A::Vector{Int},
                                       T::AbstractMatrix,
                                       res, rel::Vector{Float64})
    stats = _stage1_priority_vectors(res, cfg)
    N = cfg.N
    A_set = Set(A)
    F = [i for i in 1:N if !(i in A_set)]

    remove_scored = [(_stage1_remove_sort_key(stats, cfg, rel, i), i) for i in A]
    sort!(remove_scored; by = x -> x[1])
    remove_pool = [x[2] for x in remove_scored[1:min(cfg.joint_max_remove_candidates,
                                                      length(remove_scored))]]

    F_full = F
    promote_thr = Inf
    promote_pool_pre = F_full
    if cfg.add_post_sig_keep_quantile < 1.0 && !isempty(F_full)
        post_sigs = [stats.post_sigma[i] for i in F_full]
        q = clamp(cfg.add_post_sig_keep_quantile, 0.0, 1.0)
        thr = quantile(post_sigs, q)
        promote_thr = thr
        kept = [i for i in F_full if stats.post_sigma[i] <= thr]
        if !isempty(kept)
            promote_pool_pre = kept
        end
    end
    promote_pool = if cfg.add_use_zhat_dang ||
                       cfg.add_strategy in (:zhat_dang, :low_zhat_dang)
        sort(promote_pool_pre; by = j -> (stats.zhat_dang[j], -Float64(rel[j]), j))
    elseif cfg.add_strategy in (:q10_llr_dang, :q10_llr, :top_q10)
        sort(promote_pool_pre; by = j -> (-Float64(stats.q10_abs_llr_dang[j]),
                                           -Float64(rel[j]), j))
    elseif cfg.add_strategy in (:low_sigma, :low_support, :low_post_sigma, :low_chosen_sigma)
        sort(promote_pool_pre; by = j -> (stats.sigma[j], -Float64(rel[j]), j))
    else
        sort(promote_pool_pre; by = j -> (-Float64(rel[j]), j))
    end
    promote_pool = promote_pool[1:min(cfg.joint_max_promote_candidates,
                                       length(promote_pool))]

    # Amplifier ranking for a candidate new info bit i.  amp_k uses the
    # mean |LLR_k| over dangerous trials where LLR_i^t < 0 (the heuristic
    # for "wrong path picks v_i = 1 once i is info").
    function amplifier_ranking(i_promote::Int, Aset_new::Set{Int};
                                use_observed_v_i::Bool = false)
        samples = _stage1_explicit_samples(res)
        amp = zeros(Float64, N)
        count_fire = 0
        @inbounds for s in samples
            if hasproperty(s, :mode) && getproperty(s, :mode) !== :explicit
                continue
            end
            if hasproperty(s, :D) && Float64(getproperty(s, :D)) > 0.0
                continue
            end
            LLR = _first_existing_vector(s, (:LLR, :llr, :path_LLR, :path_llr))
            LLR === nothing && continue
            i_promote > length(LLR) && continue
            fire = if use_observed_v_i
                path_v = _first_existing_vector(s, (:path_v_scl, :path_v, :v, :V))
                path_v !== nothing && i_promote <= length(path_v) &&
                    _bit01(path_v, i_promote) == 1
            else
                Float64(LLR[i_promote]) < 0.0
            end
            fire || continue
            count_fire += 1
            kmax = min(N, i_promote + cfg.pac_max_span)
            for k in (i_promote + 1):kmax
                (k in Aset_new) && continue
                k > length(LLR) && continue
                amp[k] += abs(Float64(LLR[k]))
            end
        end
        count_fire > 0 && (amp ./= count_fire)
        return sortperm(amp; rev = true), amp, count_fire
    end

    Tset = Set{Tuple{Int,Int}}()
    @inbounds for i in 1:N, j in (i + 1):N
        if Int(T[i, j]) != 0
            push!(Tset, (i, j))
        end
    end

    candidates = NamedTuple[]

    # (1) Swap + edge moves.
    for j in remove_pool
        edges_dead_from_j = Tuple{Int,Int}[(j, t) for t in (j + 1):N
                                             if (j, t) in Tset]
        for i in promote_pool
            A_new_set = Set(setdiff(A, [j])) ∪ Set([i])

            if cfg.joint_allow_bare_swap
                push!(candidates, (
                    label    = "swap($j→$i)",
                    removes  = [j],   adds = [i],
                    edges_added = Tuple{Int,Int}[],
                    edges_removed = edges_dead_from_j,
                    edge_source = i, j_remove = j, i_add = i,
                    edge_targets = Int[],
                    edge_amp_sum = 0.0,
                ))
            end

            cfg.joint_max_edges_per_move < 1 && continue
            ranked, amp, _ = amplifier_ranking(i, A_new_set)
            n_amp = min(cfg.joint_max_amplifiers_per_source, length(ranked))
            top_k = filter(k -> amp[k] > 0.0, ranked[1:n_amp])
            for k in top_k
                push!(candidates, (
                    label    = "swap($j→$i)+edge($i→$k)",
                    removes  = [j], adds = [i],
                    edges_added   = [(i, k)],
                    edges_removed = edges_dead_from_j,
                    edge_source = i, j_remove = j, i_add = i,
                    edge_targets = [k],
                    edge_amp_sum = Float64(amp[k]),
                ))
            end
            if cfg.joint_max_edges_per_move >= 2 && length(top_k) >= 2
                for a in 1:length(top_k), b in (a + 1):length(top_k)
                    k1, k2 = top_k[a], top_k[b]
                    push!(candidates, (
                        label    = "swap($j→$i)+edges($i→$k1,$i→$k2)",
                        removes  = [j], adds = [i],
                        edges_added   = [(i, k1), (i, k2)],
                        edges_removed = edges_dead_from_j,
                        edge_source = i, j_remove = j, i_add = i,
                        edge_targets = [k1, k2],
                        edge_amp_sum = Float64(amp[k1]) + Float64(amp[k2]),
                    ))
                end
            end
        end
    end

    # (2) Edge-only moves on existing info bits.
    if cfg.joint_allow_edge_only
        info_by_sigma = sort(A; by = i -> -stats.sigma[i])
        n_src = min(cfg.joint_max_remove_candidates, length(info_by_sigma))
        for i in info_by_sigma[1:n_src]
            existing_from_i = Set{Int}([t for (s, t) in Tset if s == i])
            ranked, amp, _ = amplifier_ranking(i, A_set; use_observed_v_i = true)
            n_amp = min(cfg.joint_max_amplifiers_per_source, length(ranked))
            top_k = filter(k -> amp[k] > 0.0 && !(k in existing_from_i),
                           ranked[1:min(n_amp, length(ranked))])
            for k in top_k
                push!(candidates, (
                    label    = "edge($i→$k)",
                    removes  = Int[], adds = Int[],
                    edges_added = [(i, k)],
                    edges_removed = Tuple{Int,Int}[],
                    edge_source = i, j_remove = 0, i_add = 0,
                    edge_targets = [k],
                    edge_amp_sum = Float64(amp[k]),
                ))
            end
        end
    end

    return candidates, stats, promote_thr
end

# ──────────────────────────────────────────────────────────────────────
# One Stage 3 iteration.
# ──────────────────────────────────────────────────────────────────────
function _try_stage3_step(cfg::PipelineConfig, iter::Int,
                            A::Vector{Int}, T::AbstractMatrix,
                            res, score::Float64, rel::Vector{Float64})
    candidates, _, post_sig_thr = _stage3_generate_candidates(cfg, A, T, res, rel)
    if isempty(candidates)
        cfg.verbose && println("\n>>> Stage 3 iter $iter: no candidates")
        return (accepted = false, A = A, T = T, res = res, score = score,
                record = (iter = iter, label = "none", accepted = false,
                          old_score = score, new_score = score, n_screened = 0))
    end

    stats = _stage1_priority_vectors(res, cfg)

    # Marginal score helpers — match exactly what Stages 1 / 2 use so the
    # add_strategy / set_risk_score / freeze_use_rho knobs propagate here.
    function remove_score_of(j::Int)
        j == 0 && return 0.0
        if _using_sigma_llr(cfg)
            return Float64(stats.sigma_llr[j])
        elseif cfg.freeze_use_rho
            return Float64(stats.priority_rho[j])
        elseif cfg.set_sigma_domain in (:post, :polarized, :after, :v, :channel)
            return Float64(stats.post_sigma[j])
        else
            return Float64(stats.sigma[j])
        end
    end
    function add_score_of(i::Int)
        i == 0 && return 0.0
        if cfg.add_use_zhat_dang || cfg.add_strategy in (:zhat_dang, :low_zhat_dang)
            return 1.0 / max(Float64(stats.zhat_dang[i]) - cfg.add_zhat_floor, 1e-12)
        elseif cfg.add_strategy in (:q10_llr_dang, :q10_llr, :top_q10)
            return Float64(stats.q10_abs_llr_dang[i])
        elseif cfg.add_strategy in (:low_sigma, :low_support, :low_post_sigma, :low_chosen_sigma)
            # Lower σ → safer add; invert.
            return -Float64(stats.sigma[i])
        else
            return Float64(rel[i])
        end
    end

    # Compute marginals + joint counterfactual (kept as diagnostic) for each candidate.
    scored = NamedTuple[]
    for c in candidates
        r_raw = remove_score_of(c.j_remove)
        a_raw = add_score_of(c.i_add > 0 ? c.i_add : c.edge_source)
        e_raw = Float64(c.edge_amp_sum)
        # Joint counterfactual — diagnostic only now.
        edge_src = c.i_add > 0 ? 0 : c.edge_source
        s = _joint_move_saved_paths(res, cfg.N,
                                    c.j_remove, c.i_add, edge_src,
                                    c.edge_targets; only_dangerous = false)
        push!(scored, merge(c, (remove_raw = r_raw,
                                 add_raw    = a_raw,
                                 edge_raw   = e_raw,
                                 saved      = s.saved,
                                 broken     = s.broken,
                                 joint_score = s.score)))
    end

    # Normalize each component to [0, 1] across the pool so the weights are
    # scale-free.  Empty pools (no nonzero in a component) collapse to 0.
    max_r = max(maximum(x -> abs(x.remove_raw), scored; init = 0.0), 1e-12)
    max_a = max(maximum(x -> abs(x.add_raw),    scored; init = 0.0), 1e-12)
    max_e = max(maximum(x -> abs(x.edge_raw),   scored; init = 0.0), 1e-12)
    composite_scored = NamedTuple[]
    for c in scored
        comp = cfg.joint_w_remove * (c.remove_raw / max_r) +
               cfg.joint_w_add    * (c.add_raw    / max_a) +
               cfg.joint_w_edge   * (c.edge_raw   / max_e)
        push!(composite_scored, merge(c, (composite_score = comp,)))
    end

    sort!(composite_scored; by = x -> (-x.composite_score,
                                        -x.edge_raw, -x.remove_raw))
    scored = composite_scored

    n_screen = min(cfg.joint_max_total_candidates, length(scored))
    pool     = scored[1:n_screen]
    cfg.verbose && begin
        println("\n>>> Stage 3 iter $iter: ", length(scored),
                " candidates generated; screening top ", n_screen,
                " by marginal composite")
        @printf("    post-σ threshold for add pool: %.4e\n", post_sig_thr)
        @printf("    composite weights: w_R=%.2f w_A=%.2f w_E=%.2f\n",
                cfg.joint_w_remove, cfg.joint_w_add, cfg.joint_w_edge)
        for (k, c) in enumerate(pool[1:min(8, length(pool))])
            @printf("    [%3d] comp=%.4f (r=%.4f, a=%.4f, e=%.4f) | joint_cf=%-6d (sav=%d, brk=%d)  %s\n",
                    k, c.composite_score,
                    c.remove_raw / max_r,
                    c.add_raw / max_a,
                    c.edge_raw / max_e,
                    c.joint_score, c.saved, c.broken, c.label)
        end
    end

    small_results = NamedTuple[]
    for (k, c) in enumerate(pool)
        A_new, T_new = _stage3_apply_move(cfg.N, A, T, c)
        res_try = spectrum_for_A(cfg, A_new, T_new;
                                  n_trials = cfg.joint_n_trials_small,
                                  seed = cfg.seed + 31_000 * iter + k,
                                  label = cfg.verbose ?
                                      "Stage3 small iter=$iter cand=$k $(c.label)" :
                                      "")
        s_small = tail_score(res_try, cfg)
        cfg.verbose && @printf("    small [%3d] %s  score=%.6e (prev %.6e)\n",
                                k, c.label, s_small, score)
        push!(small_results, merge(c, (small_score = s_small,
                                       A_new = A_new, T_new = T_new)))
    end

    sort!(small_results; by = x -> x.small_score)
    full_pool = small_results[1:min(max(1, cfg.joint_full_eval_top_k),
                                      length(small_results))]

    best = nothing
    best_score = Inf
    best_res = nothing
    for (rank, c) in enumerate(full_pool)
        res_full = spectrum_for_A(cfg, c.A_new, c.T_new;
                                   n_trials = cfg.joint_n_trials_full,
                                   seed = cfg.seed + 313_000 * iter + 7 * rank,
                                   label = cfg.verbose ?
                                       "Stage3 full iter=$iter rank=$rank $(c.label)" :
                                       "")
        s_full = tail_score(res_full, cfg)
        cfg.verbose && @printf("    full  rank %2d: %s  full=%.6e prev=%.6e\n",
                                rank, c.label, s_full, score)
        if s_full < best_score
            best = c
            best_score = s_full
            best_res = res_full
        end
    end

    if best !== nothing && best_score < score - cfg.joint_min_improvement
        cfg.verbose && @printf("    ACCEPTED stage3 iter %d: %s  %.6e → %.6e\n",
                                iter, best.label, score, best_score)
        return (accepted = true, A = best.A_new, T = best.T_new,
                res = best_res, score = best_score,
                record = (iter = iter, label = String(best.label),
                          accepted = true, old_score = score,
                          new_score = best_score, n_screened = length(scored)))
    else
        cfg.verbose && println("    no Stage 3 improvement at iter $iter")
        return (accepted = false, A = A, T = T, res = res, score = score,
                record = (iter = iter,
                          label = best === nothing ? "none" : String(best.label),
                          accepted = false, old_score = score,
                          new_score = best === nothing ? score : best_score,
                          n_screened = length(scored)))
    end
end

# ──────────────────────────────────────────────────────────────────────
# Stage 3 driver.
# ──────────────────────────────────────────────────────────────────────
function run_stage3_joint(cfg::PipelineConfig, A0::Vector{Int},
                            T0::AbstractMatrix, rel::Vector{Float64})
    A = Int.(copy(A0))
    T = Matrix{UInt8}(T0)
    res = spectrum_for_A(cfg, A, T;
                          n_trials = cfg.joint_n_trials_full,
                          seed = cfg.seed + 911,
                          label = "Stage 3 initial state")
    score = tail_score(res, cfg)
    cfg.verbose && @printf("Stage 3 starting score: %.6e\n", score)

    history = NamedTuple[]
    for iter in 1:cfg.joint_iterations
        result = _try_stage3_step(cfg, iter, A, T, res, score, rel)
        push!(history, result.record)
        if result.accepted
            A = result.A
            T = result.T
            res = result.res
            score = result.score
        else
            cfg.verbose && println("Stage 3 stopping at iter $iter.")
            break
        end
    end
    return (A = A, T = T, res = res, score = score, history = history)
end

# ============================================================
# Stage 2 evaluation: baseline vs constructed PAC on the same A
# ============================================================

function evaluate_constructed_precoder(cfg::PipelineConfig, A::Vector{Int},
                                        Tgeneric::AbstractMatrix{<:Integer},
                                        design)
    T_identity = identity_tmat(cfg.N)
    T_built    = design.Tmat

    cfg.verbose && println("\n=== Stage 2 evaluation on selected A ===")

    res_identity = spectrum_for_A(cfg, A, T_identity;
                                    n_trials = cfg.n_trials_eval,
                                    seed = cfg.eval_seed,
                                    label = "final eval: identity")

    res_generic = spectrum_for_A(cfg, A, Tgeneric;
                                   n_trials = cfg.n_trials_eval,
                                   seed = cfg.eval_seed,
                                   label = "final eval: generic PAC")

    res_built = spectrum_for_A(cfg, A, T_built;
                                 n_trials = cfg.n_trials_eval,
                                 seed = cfg.eval_seed,
                                 label = "final eval: constructed PAC")

    score_identity = tail_score(res_identity, cfg)
    score_generic  = tail_score(res_generic,  cfg)
    score_built    = tail_score(res_built,    cfg)

    cfg.verbose && begin
        println("\n--- Final comparison on selected A ---")
        @printf("identity            objective = %.8e\n", score_identity)
        @printf("generic PAC         objective = %.8e\n", score_generic)
        @printf("constructed PAC     objective = %.8e\n", score_built)
        if score_built < score_generic
            @printf("Constructed beats generic by factor %.3fx.\n",
                     score_generic / max(score_built, eps()))
        elseif score_built < score_identity
            @printf("Constructed beats identity but lags generic.\n")
        else
            @printf("Constructed does not improve over identity.\n")
        end
    end

    return (
        result_identity = res_identity,
        result_generic  = res_generic,
        result_built    = res_built,
        score_identity  = score_identity,
        score_generic   = score_generic,
        score_built     = score_built,
    )
end

# ============================================================
# Saving and top-level pipeline
# ============================================================


function _show_full_value(io::IO, x)
    show(IOContext(io, :limit => false, :compact => false), MIME("text/plain"), x)
    return nothing
end

function _write_key_value(io::IO, key::AbstractString, value)
    print(io, key, " = ")
    _show_full_value(io, value)
    println(io)
    return nothing
end

function _safe_derived_initial_A(cfg::PipelineConfig)
    try
        rel = reliability_vector(cfg)
        return initial_information_set(cfg, rel)
    catch err
        return "<could not recompute initial A: $(typeof(err)): $(err)>"
    end
end

function save_complete_parameters(path::AbstractString,
                                  cfg::PipelineConfig,
                                  set_out,
                                  design,
                                  evaluation;
                                  stage3 = nothing,
                                  A_final = nothing,
                                  T_final = nothing,
                                  stdout_log_path = nothing)
    open(path, "w") do io
        println(io, "Stage123 complete parameter snapshot")
        println(io, "generated_at = ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)

        println(io, "[runtime]")
        _write_key_value(io, "julia_VERSION", VERSION)
        _write_key_value(io, "PROGRAM_FILE", PROGRAM_FILE)
        _write_key_value(io, "ARGS", ARGS)
        _write_key_value(io, "pwd", pwd())
        _write_key_value(io, "Threads.nthreads", Threads.nthreads())
        _write_key_value(io, "Base.julia_cmd", Base.julia_cmd())
        _write_key_value(io, "stdout_stderr_log", stdout_log_path)
        for key in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_NODELIST", "SLURM_CPUS_PER_TASK",
                    "SLURM_NTASKS", "SLURM_MEM_PER_NODE", "SLURM_SUBMIT_DIR",
                    "JULIA_PROJECT", "JULIA_NUM_THREADS")
            _write_key_value(io, "ENV[\"$key\"]", get(ENV, key, ""))
        end
        println(io)

        println(io, "[PipelineConfig fields]")
        for name in fieldnames(PipelineConfig)
            _write_key_value(io, String(name), getfield(cfg, name))
        end
        println(io)

        println(io, "[derived / effective values]")
        _write_key_value(io, "initial_A_effective", _safe_derived_initial_A(cfg))
        _write_key_value(io, "selected_A_after_stage1", set_out.A)
        _write_key_value(io, "A_final_evaluated", A_final === nothing ? set_out.A : A_final)
        _write_key_value(io, "generic_taps", cfg.generic_taps)
        _write_key_value(io, "sigma", noise_sigma_from_ebn0db(cfg.N, cfg.K, cfg.EbN0_dB))
        _write_key_value(io, "set_score_train", set_out.score)
        _write_key_value(io, "n_stage1_history_rows", length(set_out.history))
        _write_key_value(io, "pac_n_edges_added", length(design.edges))
        _write_key_value(io, "pac_edges", design.edges)
        _write_key_value(io, "pac_final_objective_train", design.score)
        if hasproperty(design, :working_score)
            _write_key_value(io, "pac_working_objective_train", design.working_score)
        end
        if hasproperty(design, :best_step)
            _write_key_value(io, "pac_best_step", design.best_step)
        end
        if hasproperty(design, :explore_used)
            _write_key_value(io, "pac_explore_used", design.explore_used)
        end
        _write_key_value(io, "final_identity_objective", evaluation.score_identity)
        _write_key_value(io, "final_generic_objective", evaluation.score_generic)
        _write_key_value(io, "final_constructed_objective", evaluation.score_built)
        _write_key_value(io, "stage3_enabled", cfg.enable_joint_stage3)
        _write_key_value(io, "stage3_present", stage3 !== nothing)
        if stage3 !== nothing
            _write_key_value(io, "stage3_score", stage3.score)
            _write_key_value(io, "stage3_A", stage3.A)
            _write_key_value(io, "stage3_history_length", length(stage3.history))
        end
        println(io)

        println(io, "[saved files in this output directory]")
        println(io, "A_selected.txt")
        println(io, "generic_taps.txt")
        println(io, "set_history.csv")
        println(io, "pac_edges.csv")
        println(io, "summary.txt")
        println(io, "parameters_complete.txt")
        if stdout_log_path !== nothing
            println(io, "stage123_stdout_stderr.log")
        end
        if cfg.save_matrices
            println(io, "T_generic.csv")
            println(io, "T_constructed.csv")
            println(io, "T_working_last.csv  # if present")
            println(io, "depths_generic.txt")
            println(io, "depths_constructed.txt")
        end
    end
    return path
end

function save_pipeline_outputs(cfg::PipelineConfig, set_out, design, evaluation;
                               stage3 = nothing, A_final = nothing, T_final = nothing,
                               stdout_log_path = nothing)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    snr_str = isinteger(cfg.EbN0_dB) ? string(Int(cfg.EbN0_dB)) : string(cfg.EbN0_dB)
    run_tag = "$(cfg.N)_$(cfg.K)_L$(cfg.L)_SNR$(snr_str)_$(timestamp)"
    outdir = ensure_dir(joinpath(cfg.outdir, run_tag))

    save_vector(joinpath(outdir, "A_selected.txt"), set_out.A)
    save_vector(joinpath(outdir, "generic_taps.txt"), cfg.generic_taps)
    save_set_history_csv(joinpath(outdir, "set_history.csv"), set_out.history)
    save_edges_csv(joinpath(outdir, "pac_edges.csv"), design.edges)

    if cfg.save_matrices
        writedlm(joinpath(outdir, "T_generic.csv"), Int.(set_out.Tgeneric), ',')
        writedlm(joinpath(outdir, "T_constructed.csv"), Int.(design.Tmat), ',')
        if hasproperty(design, :Tmat_working)
            writedlm(joinpath(outdir, "T_working_last.csv"), Int.(design.Tmat_working), ',')
        end
        save_vector(joinpath(outdir, "depths_generic.txt"), set_out.depths_generic)
        save_vector(joinpath(outdir, "depths_constructed.txt"), design.depths)
    end

    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "RunGenericPACSetThenSparseTV summary")
        println(io, "timestamp = ", timestamp)
        println(io, "N = ", cfg.N)
        println(io, "K = ", cfg.K)
        println(io, "EbN0_dB = ", cfg.EbN0_dB)
        println(io, "L = ", cfg.L)
        println(io, "generic_taps = ", cfg.generic_taps)
        println(io, "set_custom_Tmat_used = ", cfg.set_custom_Tmat !== nothing)
        println(io, "set_custom_Tmat_path = ", cfg.set_custom_Tmat_path)
        println(io, "selected_A = ", set_out.A)
        println(io, "set_objective = ", cfg.set_objective)
        println(io, "set_objective_train = ", set_out.score)
        println(io, "risk_mix = ", cfg.risk_mix)
        println(io, "set_sigma_domain = ", cfg.set_sigma_domain)
        println(io, "set_sigma_pre_weight = ", cfg.set_sigma_pre_weight)
        println(io, "set_sigma_post_weight = ", cfg.set_sigma_post_weight)
        println(io, "set_candidate_generator = ", cfg.set_candidate_generator)
        println(io, "set_risk_score = ", cfg.set_risk_score)
        println(io, "kicker_sizes = ", cfg.kicker_sizes)
        println(io, "set_full_eval_top_k = ", cfg.set_full_eval_top_k)
        println(io, "set_full_eval_only_small_improvers = ", cfg.set_full_eval_only_small_improvers)
        println(io, "freeze_use_rho = ", cfg.freeze_use_rho)
        println(io, "freeze_rho_floor = ", cfg.freeze_rho_floor)
        println(io, "freeze_rho_min = ", cfg.freeze_rho_min)
        println(io, "add_use_zhat_dang = ", cfg.add_use_zhat_dang)
        println(io, "add_zhat_weight = ", cfg.add_zhat_weight)
        println(io, "add_zhat_floor = ", cfg.add_zhat_floor)
        println(io, "add_post_sig_keep_quantile = ", cfg.add_post_sig_keep_quantile)
        println(io, "add_strategy = ", cfg.add_strategy)
        println(io, "pac_initial = ", cfg.pac_initial)
        println(io, "pac_max_edges = ", cfg.pac_max_edges)
        println(io, "pac_max_span = ", cfg.pac_max_span)
        println(io, "pac_target = ", cfg.pac_target)
        println(io, "pac_edge_ranking = ", cfg.pac_edge_ranking)
        println(io, "pac_source_first_min_saved = ", cfg.pac_source_first_min_saved)
        println(io, "pac_kicker_sizes = ", cfg.pac_kicker_sizes)
        println(io, "pac_candidate_sets_per_step = ", cfg.pac_candidate_sets_per_step)
        println(io, "pac_full_confirm_top_candidates = ", cfg.pac_full_confirm_top_candidates)
        println(io, "pac_explore_when_stalled = ", cfg.pac_explore_when_stalled)
        println(io, "pac_return_best_seen = ", cfg.pac_return_best_seen)
        println(io, "pac_l0_aware_scoring = ", cfg.pac_l0_aware_scoring)
        println(io, "pac_l0_lost_window = ", cfg.pac_l0_lost_window)
        println(io, "pac_l0_kept_weight = ", cfg.pac_l0_kept_weight)
        println(io, "pac_l0_lost_weight = ", cfg.pac_l0_lost_weight)
        println(io, "pac_l0_late_penalty = ", cfg.pac_l0_late_penalty)
        println(io, "pac_design = ", cfg.pac_design)
        println(io, "pac_replay_screen = ", cfg.pac_replay_screen)
        println(io, "pac_replay_delta = ", cfg.pac_replay_delta)
        println(io, "pac_parity_max_steps = ", cfg.pac_parity_max_steps)
        println(io, "pac_parity_source_pool = ", cfg.pac_parity_source_pool)
        println(io, "pac_parity_targets_screened = ", cfg.pac_parity_targets_screened)
        println(io, "pac_parity_candidates_per_step = ", cfg.pac_parity_candidates_per_step)
        println(io, "pac_parity_full_confirm = ", cfg.pac_parity_full_confirm)
        println(io, "pac_parity_min_trial_gain = ", cfg.pac_parity_min_trial_gain)
        println(io, "pac_parity_restarts = ", cfg.pac_parity_restarts)
        println(io, "pac_edge_metric = ", cfg.pac_edge_metric)
        println(io, "pac_parity_max_col_weight = ", cfg.pac_parity_max_col_weight)
        println(io, "pac_parity_tap_penalty = ", cfg.pac_parity_tap_penalty)
        println(io, "pac_parity_threshold = ", cfg.pac_parity_threshold)
        println(io, "pac_parity_bulk_prune = ", cfg.pac_parity_bulk_prune)
        println(io, "pac_parity_bulk_prune_slack = ", cfg.pac_parity_bulk_prune_slack)
        println(io, "pac_parity_bulk_prune_rel_slack = ", cfg.pac_parity_bulk_prune_rel_slack)
        println(io, "enable_joint_stage3 = ", cfg.enable_joint_stage3)
        println(io, "joint_iterations = ", cfg.joint_iterations)
        println(io, "joint_max_edges_per_move = ", cfg.joint_max_edges_per_move)
        println(io, "joint_allow_bare_swap = ", cfg.joint_allow_bare_swap)
        println(io, "joint_allow_edge_only = ", cfg.joint_allow_edge_only)
        println(io, "pac_n_edges_added_steps = ", length(design.edges))
        println(io, "pac_final_objective_train = ", design.score)
        println(io, "pac_working_objective_train = ", design.working_score)
        println(io, "pac_best_step = ", design.best_step)
        println(io, "pac_explore_used = ", design.explore_used)
        println(io, "final_identity_objective = ", evaluation.score_identity)
        println(io, "final_generic_objective = ", evaluation.score_generic)
        println(io, "final_constructed_objective = ", evaluation.score_built)
    end

    save_complete_parameters(joinpath(outdir, "parameters_complete.txt"),
                             cfg, set_out, design, evaluation;
                             stage3 = stage3,
                             A_final = A_final === nothing ? set_out.A : A_final,
                             T_final = T_final,
                             stdout_log_path = stdout_log_path)

    if stdout_log_path !== nothing && isfile(stdout_log_path)
        try
            cp(stdout_log_path, joinpath(outdir, "stage123_stdout_stderr.log"); force = true)
        catch err
            @warn "Could not copy stdout/stderr log into final output directory" stdout_log_path outdir err
        end
    end

    return outdir
end

function run_pipeline(cfg::PipelineConfig = PipelineConfig(); stdout_log_path = nothing)
    println("============================================================")
    println("Stage 1: information-set design under generic PAC")
    println("Stage 2: iterative PAC precoder construction (path-based scoring)")
    println("============================================================")
    println("N = ", cfg.N, ", K = ", cfg.K, ", EbN0_dB = ", cfg.EbN0_dB, ", L = ", cfg.L)
    println("generic taps         = ", cfg.generic_taps)
    println("reliability          = ", cfg.reliability_method)
    println("set_objective        = ", cfg.set_objective)
    println("set_risk_score       = ", cfg.set_risk_score,
            "  (use :sigma_llr for σ * |LLR| ranking)")
    println("kicker_sizes (set)   = ", cfg.kicker_sizes)
    println("pac_design           = ", cfg.pac_design,
            cfg.pac_design == :parity ?
            "  (column redesign + replay screen)" : "  (legacy per-edge)")
    println("pac_initial          = ", cfg.pac_initial)
    if cfg.pac_design == :parity
        println("pac_replay_screen    = ", cfg.pac_replay_screen,
                "  delta = ", cfg.pac_replay_delta)
        println("pac_parity steps     = ", cfg.pac_parity_max_steps,
                "  bulk_prune = ", cfg.pac_parity_bulk_prune)
    end
    println("pac_max_edges        = ", cfg.pac_max_edges,
            "  (max_span = ", cfg.pac_max_span, ")")
    println("pac_target           = ", cfg.pac_target)
    println("pac_kicker_sizes     = ", cfg.pac_kicker_sizes)
    println("pac batch confirms   = ", cfg.pac_full_confirm_top_candidates, " of ", cfg.pac_candidate_sets_per_step)
    println("pac exploration      = ", cfg.pac_explore_when_stalled, "  return_best_seen = ", cfg.pac_return_best_seen)
    println("pac L0-aware scoring = ", cfg.pac_l0_aware_scoring, "  lost_window = ", cfg.pac_l0_lost_window)
    println("sigma                = ", noise_sigma_from_ebn0db(cfg.N, cfg.K, cfg.EbN0_dB))

    set_out    = run_generic_pac_set_design(cfg)
    design     = cfg.pac_design == :parity ?
                 run_iterative_pac_design_parity(cfg, set_out) :
                 run_iterative_pac_design(cfg, set_out)

    # Stage 3 (optional): joint refinement on (A, T).
    stage3 = nothing
    A_final = set_out.A
    T_final = design.Tmat
    if cfg.enable_joint_stage3
        println("\n============================================================")
        println("Stage 3 (joint refinement)")
        println("============================================================")
        stage3 = run_stage3_joint(cfg, set_out.A, design.Tmat, set_out.reliability)
        A_final = stage3.A
        T_final = stage3.T
        println("\nStage 3 finished:")
        println("  iterations recorded     = ", length(stage3.history))
        println("  iterations accepted     = ", count(h -> h.accepted, stage3.history))
        @printf("  starting → final score  = %.6e → %.6e\n",
                length(stage3.history) > 0 ? stage3.history[1].old_score : design.score,
                stage3.score)
        # Replace design's Tmat/A so downstream evaluation/save uses the refined code.
        design = merge(design, (Tmat = Int.(T_final),
                                Tmat_working = Int.(T_final),
                                depths = DPS.infer_depths(T_final),
                                result = stage3.res,
                                score  = stage3.score))
    end
    evaluation = evaluate_constructed_precoder(cfg, A_final, set_out.Tgeneric, design)

    outdir = save_pipeline_outputs(cfg, set_out, design, evaluation;
                                   stage3 = stage3,
                                   A_final = A_final,
                                   T_final = T_final,
                                   stdout_log_path = stdout_log_path)
    if stage3 !== nothing
        open(joinpath(outdir, "stage3_history.csv"), "w") do io
            println(io, "iter,label,accepted,old_score,new_score,n_screened")
            for h in stage3.history
                println(io, h.iter, ",\"", h.label, "\",",
                        h.accepted, ",", h.old_score, ",",
                        h.new_score, ",", h.n_screened)
            end
        end
        save_vector(joinpath(outdir, "A_after_stage3.txt"), A_final)
        if cfg.save_matrices
            writedlm(joinpath(outdir, "T_after_stage3.csv"), Int.(T_final), ',')
        end
    end
    println("\nSaved outputs to: ", outdir)

    return (config = cfg, set = set_out, design = design,
             evaluation = evaluation, stage3 = stage3, outdir = outdir)
end


function run_pipeline_with_output_log(cfg::PipelineConfig = PipelineConfig())
    logpath = _active_log_path(cfg)
    result_ref = Ref{Any}(nothing)

    # Julia 1.12 performs stdout/stderr redirection at the file-descriptor
    # level and does not accept arbitrary custom IO objects here.  Therefore
    # we redirect to the real log file directly rather than to TeeIO.  This
    # preserves all experiment logic; only the destination of stdout/stderr is
    # changed while run_pipeline executes.
    original_stdout = stdout
    original_stderr = stderr
    println(original_stdout, "Stage123 stdout/stderr log = ", logpath)
    flush(original_stdout)

    open(logpath, "w") do logfile
        redirect_stdout(logfile) do
            redirect_stderr(logfile) do
                try
                    println("Stage123 stdout/stderr log = ", logpath)
                    result_ref[] = run_pipeline(cfg; stdout_log_path = logpath)
                    if result_ref[] !== nothing && hasproperty(result_ref[], :outdir)
                        flush(stdout)
                        flush(stderr)
                        flush(logfile)
                        try
                            cp(logpath, joinpath(result_ref[].outdir, "stage123_stdout_stderr.log"); force = true)
                        catch copy_err
                            @warn "Could not finalize stdout/stderr log copy" logpath result_ref[].outdir copy_err
                        end
                    end
                catch err
                    println(stderr, "\nERROR while running Stage123. Stack trace follows.")
                    showerror(stderr, err, catch_backtrace())
                    println(stderr)
                    flush(stdout)
                    flush(stderr)
                    rethrow()
                finally
                    flush(stdout)
                    flush(stderr)
                end
            end
        end
    end

    if result_ref[] !== nothing && hasproperty(result_ref[], :outdir)
        println(original_stdout, "Stage123 stdout/stderr log copied to: ", joinpath(result_ref[].outdir, "stage123_stdout_stderr.log"))
    else
        println(original_stdout, "Stage123 stdout/stderr log saved to: ", logpath)
    end
    flush(original_stdout)

    return result_ref[]
end

# ------------------------------------------------------------
# Edit this block when running as a script.
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    N = 64; 
    K = 32;
    SNR = 3;
    L = 32;


    
    cfg = PipelineConfig(
        N = N,
        K = K,
        EbN0_dB = SNR,
        L = L,
        generic_taps = [1,0,1,1,0,1,1],
        reliability_method = :dega,

        # ── Stage 1 ──
        set_iters = 10,
        n_trials_set = 300_000,
        n_trials_candidate = 100_000,
        set_objective = :fer,
        risk_mix      = (lambda = 0.0, sigma = 1.0),
        # set_risk_score choices for Stage-1 removal ranking:
        #   :sigma      → rank by chosen-domain sigma (count of v=1 events on
        #                 dangerous trials, normalized by n_trials).  This is
        #                 the historical default.
        #   :sigma_llr  → rank by sigma * E[|LLR| | bit was 1] in the chosen
        #                 domain.  Equivalent to (1/n_trials) Σ_t (bit=1)·|LLR_t|,
        #                 i.e. the per-trial expected freeze PM penalty.  This
        #                 down-ranks high-σ-low-LLR bits whose freeze would not
        #                 inflict meaningful PM on the wrong path, and lifts
        #                 high-σ-high-LLR bits where the freeze actually bites.
        #                 The report always shows both σ and σ·|LLR| columns so
        #                 the contrast is visible per bit.
        set_risk_score = :sigma,
        # Use :source for the pre-convolution sigma, :post for the
        # post-convolution/channel-facing support statistic, or :hybrid.  The
        # chosen domain also determines which sigma_llr (source or post)
        # drives ranking when set_risk_score = :sigma_llr.
        set_sigma_domain = :source,
        set_candidate_generator = :local,
        add_strategy  = :q10_llr_dang,
        # add_strategy options:
        #   :top_reliability                — DEGA descending (default).
        #   :low_sigma                      — chosen-domain σ ascending.
        #   :high_reliability_low_sigma     — DEGA first, σ tie-break.
        #   :zhat_dang                      — Ẑ on dangerous trials, ascending.
        #   :zhat_then_reliability          — Ẑ_dang first, DEGA tie-break.
        #   :q10_llr_dang                   — 10th-percentile |LLR| on
        #                                     dangerous trials, descending.
        #                                     Robust lower-tail reliability;
        #                                     see notes for why this can be
        #                                     useful when mean |LLR| is
        #                                     inflated by outlier-huge LLRs.
        kicker_sizes  = [],
        set_full_eval_top_k = 8,
        set_full_eval_only_small_improvers = false,
        set_kicker_escalate = true,
        initial_reliability_method = :dega, 

        nremove = 4,
        nadd = 4,
        max_candidates_per_iter = 16,
        # ── rho_i freeze efficacy (off by default) ──
        # When on, removal priority becomes
        #   source_sigma_i * max(rho_i, freeze_rho_floor)
        # i.e. "freezable mass on v" times "rate that this v=1 propagates to
        # u=1 in dangerous paths".  source_sigma (not post_sigma) is the right
        # base because freezing acts on v.  freeze_rho_min hard-filters info
        # bits with rho strictly below the threshold out of the remove pool.
        freeze_use_rho = false,
        freeze_rho_floor = 0.0,
        freeze_rho_min = 0.0,

        # ── Zhat_j^dang add reliability (off by default) ──
        # When add_use_zhat_dang = true, the add pool is ranked by lowest
        # Zhat_j^dang first (Bhattacharyya-like reliability conditioned on
        # dangerous trials).  add_zhat_weight controls how strongly Zhat
        # enters the composite (remove_risk - beta * Zhat) candidate sort.
        add_use_zhat_dang = false,
        add_zhat_weight = 1.0,
        add_zhat_floor = 0.0,

        # ── post_sigma add-pool filter (within-run quantile) ──
        # add_post_sig_keep_quantile = q means "keep frozen bits whose
        # post_sigma is in the bottom q-quantile of the current frozen-bit
        # distribution."  This automatically rescales the cutoff per run, so
        # a fixed q works across SNR / code dimension / T density.  Frozen
        # bits with high post_sigma are positions where the current precoder
        # T is actively routing wrong-path mass — promoting them destroys
        # that parity check and is usually a worse add than promoting a
        # structurally-clean position.  Defaults to 0.5 (drop the top half).
        # Set to 1.0 to disable the filter and restore legacy behavior.
        add_post_sig_keep_quantile = 0.5,

        # ── Stage 2 design mode ──
        # :edges  — per-edge additive search (1-2 taps per step), now with
        #           trial-level scoring (pac_edge_metric = :trials) and
        #           replay screening.  The sparse-precoder recipe: every tap
        #           must individually survive an honest paired evaluation,
        #           and the search stops when marginal taps stop paying.
        # :parity — per-target column redesign.  Finds more FER but buys
        #           density; if used for the sparse story, constrain it with
        #           pac_parity_max_col_weight / pac_parity_tap_penalty.
        pac_design = :edges,
        pac_replay_screen = true,
        pac_replay_delta = 10.0,
        pac_parity_max_steps = 32,
        pac_parity_source_pool = 48,
        pac_parity_targets_screened = 24,
        pac_parity_candidates_per_step = 8,
        pac_parity_full_confirm = 3,
        pac_parity_min_trial_gain = 2.0,
        pac_parity_restarts = 8,
        # Sparsity: cap designed taps per column and/or charge per tap.
        pac_parity_max_col_weight = 3,
        pac_parity_tap_penalty = 1.0,
        # Two entry points, both one confirmed move per committed column:
        #   grow  : pac_initial = :identity, pac_parity_bulk_prune = false
        #   prune : pac_initial = :generic,  pac_parity_bulk_prune = true
        pac_parity_bulk_prune = true,

          # ── Stage 2 (new path-based scoring + kicker) ──
        pac_initial = :identity,
        #pac_initial = :custom, pac_custom_Tmat = run.T,
        pac_max_edges = 96,
        pac_max_span  = N,
        pac_target    = :frozen,
        # pac_edge_ranking options:
        #   :saved_paths  — global ranking of (i, j) pairs by saved-paths count
        #                   (legacy).  Can concentrate the single-edge pool on
        #                   a few high-σ sources that each have multiple decent
        #                   targets, leaving other high-σ sources unedged.
        #   :source_first — pick the best target per source, then rank pairs
        #                   by σ_source.  One edge per high-σ info bit per
        #                   iteration; better diversification.  pac_source_first_min_saved
        #                   sets the floor (a per-source pick must save at
        #                   least this many trials, otherwise the source is
        #                   skipped — safety valve against committing edges
        #                   from high-σ bits with no good amplifier in span).
        pac_edge_ranking = :saved_paths,
        pac_source_first_min_saved = 1,
        # Trial-level counterfactual scoring (:paths restores legacy).
        pac_edge_metric = :trials,
        pac_top_edges_per_step = 32,
        pac_kicker_sizes = [1],
        pac_kicker_max_candidates = 192,
        pac_kicker_single_pool = 32,
        pac_candidate_sets_per_step = 32,
        pac_full_confirm_top_candidates = 20,
        pac_small_screen_slack_events = 2.0,
        pac_confirm_best_even_if_screen_fails = true,
        pac_explore_when_stalled = true,
        pac_max_explore_steps = 0,
        pac_explore_max_rel_worse = 0.03,
        pac_explore_max_rel_above_best = 0.10,
        pac_return_best_seen = true,
        pac_n_trials_small = 200_000,
        pac_n_trials_full  = 400_000,
        pac_l0_aware_scoring = true,
        pac_l0_lost_window = 32,
        pac_l0_kept_weight = 1.0,
        pac_l0_lost_weight = 1.0,
        pac_l0_late_penalty = 0.0,


        n_trials_eval = 300_000,
        #initial_A = Misc.rm_polar(128,42,2),
        initial_A = [14, 15, 16, 22, 23, 24, 27, 28, 29, 30, 31, 32, 39, 40, 43, 44, 45, 46, 47, 48, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64],
        #initial_A = [14, 15, 16, 22, 23, 24, 27, 28, 29, 30, 31, 32, 39, 40, 43, 44, 45, 46, 47, 48, 49, 51, 53, 55, 56, 57, 58, 59, 61, 62, 63, 64],
        #initial_A = sort(sortperm(Misc.dega(N,K,SNR), rev = true)[1:K]),
        #initial_A = [12, 14, 15, 16, 23, 24, 27, 28, 29, 30, 31, 32, 39, 40, 42, 43, 44, 45, 46, 47, 48, 51, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64],
        #initial_A =[32, 48, 56, 60, 62, 63, 64, 80, 88, 92, 93, 94, 95, 96, 103, 104, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 143, 144, 151, 152, 154, 155, 156, 157, 158, 159, 160, 164, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 196, 198, 199, 200, 202, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 224, 226, 227, 228, 229, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256],
        #initial_A = [16, 24, 28, 31, 32, 40, 44, 46, 47, 48, 52, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 72, 76, 78, 79, 80, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 103, 104, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121, 123, 124, 125, 126, 127, 128],

        # ── Stage 3 (optional joint refinement) ──
        # When enabled, after Stages 1 + 2 a final pass proposes joint
        # (R_j, A_i, i→k) moves on the already-refined (A, T).  Default off.
        enable_joint_stage3            = false,
        joint_iterations               = 5,
        joint_max_remove_candidates    = 6,
        joint_max_promote_candidates   = 8,
        joint_max_amplifiers_per_source = 4,
        joint_max_edges_per_move       = 1,
        joint_allow_bare_swap          = true,
        joint_allow_edge_only          = false,
        joint_max_total_candidates     = 24,
        joint_full_eval_top_k          = 8,
        joint_n_trials_small           = 100_000,
        joint_n_trials_full            = 300_000,
        joint_min_improvement          = 0.0,
        # Marginal-composite ranking weights for Stage 3 (matches Stage 1/2
        # selection criteria — your add_strategy = :q10_llr_dang etc. propagate).
        joint_w_remove                 = 1.0,
        joint_w_add                    = 1.0,
        joint_w_edge                   = 0.0,
    )

    run_pipeline_with_output_log(cfg)
end
