# ============================================================
# DPS_Search.jl
#
# Dangerous-path-spectrum (DPS) driven construction of PAC codes for
# SCL decoding.  Two stages:
#
#   STAGE 1 — Information-set design under a fixed PAC precoder
#             (Algorithm 2 in the paper).  Each iteration runs one Monte
#             Carlo simulation of the current code, computes the
#             dangerous-path spectrum D, and proposes k-swaps:
#               * removal pool: information bits with the largest delta_i;
#               * add pool:     frozen bits whose rho_j is at or below the
#                               `add_rho_keep_quantile` of the frozen bits,
#                               ranked by q_{0.1}(|LLR_j|) on dangerous paths.
#             Candidates are screened with a reduced Monte-Carlo budget and
#             the best few are re-evaluated with the full budget.
#
#   STAGE 2 — Sparse PAC precoder construction starting from the identity
#             (Algorithm 3 in the paper).  Each candidate edge (i -> j) is
#             scored by the number of dangerous trials it would rescue
#             (eq. (17), with the l0-aware window b = `pac_l0_lost_window`).
#             The best edge sets are screened by replaying the stored error
#             and near-miss trials (margin <= `pac_replay_delta`) and the
#             winners are confirmed with a full Monte-Carlo run.
#
# Notation (matches the paper):
#   delta_j  = (1/M) sum_m sum_{dangerous paths l} 1{v_j = 1}   eq. (15)
#              (pre-precoder / information domain)
#   rho_j    = (1/M) sum_m sum_{dangerous paths l} 1{u_j = 1}   eq. (16)
#              (post-precoder / polar-input domain)
#   gamma_j  = P(u_j = 1 | v_j = 1, dangerous)   (diagnostic column only)
#   noise_sigma is the AWGN noise standard deviation (unrelated to the above).
#
# Bit indices are 1-based throughout this code (the paper uses 0-based).
#
# Files required next to this script:
#   DangerousPathSpectrum.jl, ClassicalPolarCodeFuncSCD.jl, TailAwarePM.jl
# and ../Misc.jl one directory up.
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

function _active_log_path(cfg)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    logdir = ensure_dir(joinpath(cfg.outdir, "_active_logs"))
    return joinpath(logdir, "dps_search_stdout_stderr_$(timestamp)_pid$(getpid()).log")
end

# ------------------------------------------------------------
# Configuration
#
# Defaults reproduce the settings used for the paper's constructions.
# Per-code quantities (N, K, L, EbN0_dB, kicker sizes, trial counts, initial
# set) changed between runs and should be set explicitly in the run block at
# the bottom of this file.
# ------------------------------------------------------------

Base.@kwdef mutable struct PipelineConfig
    # ── Code / channel ──
    N::Int = 64
    K::Int = 32
    EbN0_dB::Float64 = 3.0
    L::Int = 32

    # Generic (time-invariant) PAC precoder used in Stage 1 and as the
    # baseline in the final evaluation.  [1,0,1,1,0,1,1] is octal 133.
    generic_taps::Vector{Int} = [1, 0, 1, 1, 0, 1, 1]

    # Optional custom Stage-1 precoder.  Priority: matrix > CSV path > taps.
    # Must be upper-triangular with ones on the diagonal.
    set_custom_Tmat::Any = nothing
    set_custom_Tmat_path::Union{Nothing,String} = nothing

    # ── Initial information set ──
    # If `initial_A` is given (1-based) it is used directly; otherwise the K
    # most reliable positions under `initial_reliability_method` are taken.
    initial_A::Union{Nothing,Vector{Int}} = nothing
    initial_reliability_method::Symbol = :dega   # :dega or :rm
    # Reliability used to break ties in the Stage-1 removal/add rankings.
    reliability_method::Symbol = :dega           # :dega or :rm
    reliability_Q::Int = 24                      # Gauss–Hermite points for :dega

    # ── Stage 1: information-set design (Algorithm 2) ──
    set_iters::Int = 10                  # maximum number of accepted swaps
    n_trials_set::Int = 300_000          # FullMC budget
    n_trials_candidate::Int = 100_000    # MiniMC (screening) budget
    nremove::Int = 4                     # |R|: removal pool size
    nadd::Int = 4                        # |P|: addition pool size
    max_candidates_per_iter::Int = 16    # candidates kept per swap size
    max_kicker_candidates::Int = 128
    kicker_sizes::Vector{Int} = [1]      # swap sizes S
    # true:  try swap sizes in order, escalating only when a size fails.
    # false: pool all swap sizes and screen them together.
    set_kicker_escalate::Bool = true
    set_full_eval_top_k::Int = 8         # M: candidates re-evaluated with FullMC
    set_full_eval_only_small_improvers::Bool = false
    set_objective::Symbol = :fer
    min_set_improvement::Float64 = 0.0

    # Add-pool filter: keep only frozen bits whose rho_j is at or below this
    # quantile of rho over the current frozen bits (1.0 disables the filter).
    add_rho_keep_quantile::Float64 = 0.5
    # Add-pool ranking:
    #   :q10_llr_dang    — highest 10th-percentile |LLR_j| on dangerous paths
    #                      (ties broken by reliability).  Used in the paper.
    #   :top_reliability — reliability only.
    add_strategy::Symbol = :q10_llr_dang

    # ── Stage 2: sparse precoder construction (Algorithm 3) ──
    pac_initial::Symbol = :identity       # :identity, :generic, or :custom
    pac_custom_Tmat::Any = nothing
    pac_custom_Tmat_path::Union{Nothing,String} = nothing
    pac_max_edges::Int = 96               # maximum number of edge-add steps
    pac_max_span::Union{Nothing,Int} = nothing   # max j - i; nothing = no limit
    pac_target::Symbol = :frozen          # :frozen (paper) or :all
    pac_n_trials_small::Int = 200_000     # MiniMC budget (only if replay is off)
    pac_n_trials_full::Int = 400_000      # FullMC budget
    pac_min_events_for_edges::Int = 20
    pac_min_improvement::Float64 = 0.0
    pac_only_dangerous::Bool = true

    # Candidate generation: rank single edges by rescued trials, then form
    # edge sets of each size in `pac_kicker_sizes` (k in the paper) from the
    # top `pac_kicker_single_pool` single edges.
    pac_kicker_sizes::Vector{Int} = [1, 2]
    pac_top_edges_per_step::Int = 32
    pac_kicker_single_pool::Int = 32
    pac_kicker_max_candidates::Int = 192
    pac_candidate_sets_per_step::Int = 32         # sets passed to screening
    pac_full_confirm_top_candidates::Int = 20     # M: sets re-evaluated with FullMC
    pac_small_screen_slack_events::Float64 = 2.0
    pac_confirm_best_even_if_screen_fails::Bool = true

    # l0-aware scoring.  If the correct path l0 survived to the final list, an
    # edge can target any frozen bit after i.  If l0 was pruned, the edge only
    # counts when its target j <= first_error + pac_l0_lost_window (b).
    pac_l0_aware_scoring::Bool = true
    pac_l0_lost_window::Int = 32
    pac_l0_kept_weight::Float64 = 1.0
    pac_l0_lost_weight::Float64 = 1.0
    pac_l0_late_penalty::Float64 = 0.0

    # Replay screening: store every trial with final-list margin <= delta
    # (all errors and near misses) and re-decode only those under each
    # candidate precoder.  When false, MiniMC is used for screening.
    pac_replay_screen::Bool = true
    pac_replay_delta::Float64 = 10.0     # tau in the paper

    # Optional near-neutral moves when the search stalls.  The best precoder
    # seen over the walk is returned when pac_return_best_seen = true.
    pac_explore_when_stalled::Bool = true
    pac_max_explore_steps::Int = 0
    pac_explore_max_rel_worse::Float64 = 0.03
    pac_explore_max_abs_worse::Float64 = 0.0
    pac_explore_max_rel_above_best::Float64 = 0.10
    pac_explore_max_abs_above_best::Float64 = 0.0
    pac_return_best_seen::Bool = true

    # ── Final evaluation (identity vs generic vs constructed precoder) ──
    n_trials_eval::Int = 300_000

    # ── Dangerous-path spectrum options ──
    modes::Tuple = (:explicit,)
    skip_all_positive::Bool = true
    eta_grid = range(0.0, 5.0, length = 51)
    alphas::Tuple = (0.01, 0.05, 0.10)
    seed::Int = 7
    eval_seed::Int = 11

    # ── Output ──
    outdir::String = joinpath(SCRIPT_DIR, "pac_sparse_pipeline_out")
    save_matrices::Bool = true
    verbose::Bool = true
end

_pac_span(cfg::PipelineConfig) = cfg.pac_max_span === nothing ? cfg.N : cfg.pac_max_span

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
    noise_sigma = noise_sigma_from_ebn0db(N, K, EbN0_dB)
    return 2.0 / (noise_sigma * noise_sigma)
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
    noise_sigma = noise_sigma_from_ebn0db(cfg.N, cfg.K, cfg.EbN0_dB)
    depths = DPS.infer_depths(Tmat)
    cfg.verbose && !isempty(label) && println("\n--- Spectrum: ", label, " ---")
    cfg.verbose && @printf("A length = %d, trials = %d, noise_sigma = %.6g\n",
                            length(A), n_trials, noise_sigma)

    return DPS.dangerous_path_spectrum(
        cfg.N, cfg.K, sort(Int.(A)), noise_sigma, n_trials, cfg.L, Tmat;
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


# ============================================================
# Stage 1: dangerous-path statistics (delta, rho, gamma, LLR tail stats)
# ============================================================

# Explicit dangerous-path samples: one PathSample per path l in L*_m that beat
# l0 (D_{l,m} <= 0), replayed through the forced SC decoder so that the
# per-position LLRs and post-precoder bits are available.
_dangerous_samples(res) =
    [s for s in res.samples_by_mode[:explicit] if s.mode === :explicit && Float64(s.D) <= 0.0]

"""
    _stage1_support_vectors(res, N)

Per-position occupancy of the dangerous paths (all paths with D <= 0, summed
over paths and normalized by the number of Monte-Carlo trials M):

    delta_j = (1/M) Σ_m Σ_l 1{v_j = 1}     (eq. 15, pre-precoder)
    rho_j   = (1/M) Σ_m Σ_l 1{u_j = 1}     (eq. 16, post-precoder)
    gamma_j = #(v_j = 1 and u_j = 1) / #(v_j = 1)   (diagnostic)
"""
function _stage1_support_vectors(res, N::Int)
    M = max(1, Int(res.n_trials))
    v_count = zeros(Int, N)
    u_count = zeros(Int, N)
    both_count = zeros(Int, N)
    n_paths = 0

    for s in _dangerous_samples(res)
        n_paths += 1
        v = s.path_v_scl     # information-domain path bits
        u = s.B              # post-precoder path bits
        @inbounds for j in 1:N
            vj = v[j] != 0
            uj = u[j] != 0
            v_count[j] += vj
            u_count[j] += uj
            both_count[j] += (vj && uj)
        end
    end

    delta = v_count ./ M
    rho   = u_count ./ M
    gamma = [v_count[j] > 0 ? both_count[j] / v_count[j] : 0.0 for j in 1:N]

    return (delta = delta, rho = rho, gamma = gamma,
            v_count = v_count, u_count = u_count, both_count = both_count,
            n_paths = n_paths, n_trials = M)
end

"""
    _stage1_llr_dang_stats(res, N)

Statistics of |LLR_j| along the dangerous paths.  `q10_abs_llr[j]` is
q_{0.1}(|LLR_j|), the add-pool score used in the paper.
"""
function _stage1_llr_dang_stats(res, N::Int)
    buckets = [Float64[] for _ in 1:N]
    sign_err = zeros(Int, N)
    n_paths = 0

    for s in _dangerous_samples(res)
        n_paths += 1
        @inbounds for j in 1:min(N, length(s.LLR))
            l = Float64(s.LLR[j])
            push!(buckets[j], abs(l))
            l < 0.0 && (sign_err[j] += 1)
        end
    end

    mean_abs_llr = [isempty(b) ? 0.0 : mean(b) for b in buckets]
    q10_abs_llr  = [isempty(b) ? 0.0 : quantile(b, 0.1) for b in buckets]
    sign_err_rate = sign_err ./ max(1, n_paths)

    return (mean_abs_llr = mean_abs_llr, q10_abs_llr = q10_abs_llr,
            sign_err_rate = sign_err_rate, n_paths = n_paths)
end

# Number of dangerous paths whose first error is at each position (lambda).
function _stage1_first_error_counts(res, N::Int)
    counts = zeros(Int, N)
    for ev in res.events
        1 <= ev.first_error <= N && (counts[ev.first_error] += 1)
    end
    return (count = counts, lambda = counts ./ max(1, Int(res.n_trials)))
end

function _stage1_stats(res, cfg::PipelineConfig)
    sup = _stage1_support_vectors(res, cfg.N)
    llr = _stage1_llr_dang_stats(res, cfg.N)
    fe  = _stage1_first_error_counts(res, cfg.N)
    return (delta = sup.delta, rho = sup.rho, gamma = sup.gamma,
            v_count = sup.v_count, u_count = sup.u_count, both_count = sup.both_count,
            count = fe.count, lambda = fe.lambda,
            mean_abs_llr_dang = llr.mean_abs_llr,
            q10_abs_llr_dang = llr.q10_abs_llr,
            sign_err_rate_dang = llr.sign_err_rate,
            n_paths = sup.n_paths, n_paths_llr = llr.n_paths,
            n_trials = sup.n_trials)
end

# Removal ranking: largest delta first, ties broken by lower reliability.
_stage1_remove_sort_key(stats, rel::AbstractVector{<:Real}, i::Int) =
    (-stats.delta[i], -Float64(rel[i]), i)

# Add pool: rho filter, then the configured ranking.
function _stage1_add_pool(stats, cfg::PipelineConfig, A::AbstractVector{<:Integer},
                          rel::AbstractVector{<:Real})
    Aset = Set(Int.(A))
    F_full = [j for j in 1:cfg.N if !(j in Aset)]

    F, threshold, filtered = if cfg.add_rho_keep_quantile >= 1.0 || isempty(F_full)
        F_full, Inf, false
    else
        q = clamp(cfg.add_rho_keep_quantile, 0.0, 1.0)
        thr = quantile([stats.rho[j] for j in F_full], q)
        kept = [j for j in F_full if stats.rho[j] <= thr]
        # If the filter would empty the pool, fall back to all frozen bits.
        isempty(kept) ? (F_full, thr, false) : (kept, thr, true)
    end

    key = if cfg.add_strategy == :q10_llr_dang
        j -> (-Float64(stats.q10_abs_llr_dang[j]), -Float64(rel[j]), j)
    elseif cfg.add_strategy == :top_reliability
        j -> (-Float64(rel[j]), j)
    else
        throw(ArgumentError("add_strategy must be :q10_llr_dang or :top_reliability."))
    end

    return (pool = sort(F; by = key), full = F_full,
            threshold = threshold, filtered = filtered)
end

function print_stage1_priority_report(res, cfg::PipelineConfig,
                                      rel::AbstractVector{<:Real}; max_rows::Int = 32)
    stats = _stage1_stats(res, cfg)
    active = findall(i -> stats.count[i] > 0 || stats.delta[i] > 0.0 || stats.rho[i] > 0.0,
                     1:cfg.N)
    if isempty(active)
        println("No dangerous-path events to summarize.")
        return
    end
    sort!(active; by = i -> _stage1_remove_sort_key(stats, rel, i))

    println("\nStage-1 dangerous-path report (sorted by delta, i.e. removal priority)")
    @printf("  dangerous paths = %d, normalized by trials M = %d\n",
            stats.n_paths, stats.n_trials)
    @printf("%6s %8s %8s %8s %8s %10s %12s %12s %12s %12s %12s\n",
            "i", "fe_cnt", "v_cnt", "u_cnt", "both", "gamma",
            "lambda", "delta", "rho", "|LLR|_d", "q10|LLR|")
    for i in first(active, min(max_rows, length(active)))
        @printf("%6d %8d %8d %8d %8d %10.4f %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                i, stats.count[i], stats.v_count[i], stats.u_count[i], stats.both_count[i],
                stats.gamma[i], stats.lambda[i], stats.delta[i], stats.rho[i],
                stats.mean_abs_llr_dang[i], stats.q10_abs_llr_dang[i])
    end
end

function print_stage1_add_candidate_report(res, cfg::PipelineConfig,
                                           A::AbstractVector{<:Integer},
                                           rel::AbstractVector{<:Real};
                                           max_rows::Int = 32)
    stats = _stage1_stats(res, cfg)
    add = _stage1_add_pool(stats, cfg, A, rel)
    if isempty(add.full)
        println("\nStage-1 add-candidate report: no frozen bits to summarize.")
        return
    end

    println("\nStage-1 add-candidate report (frozen bits, in add-pool order)")
    println("  add_strategy = ", cfg.add_strategy)
    if cfg.add_rho_keep_quantile >= 1.0
        println("  rho filter: OFF (add_rho_keep_quantile = 1.0)")
    elseif add.filtered
        @printf("  rho filter: keep_quantile = %.3g -> rho <= %.4e, kept %d / %d frozen bits\n",
                cfg.add_rho_keep_quantile, add.threshold, length(add.pool), length(add.full))
    else
        @printf("  rho filter: keep_quantile = %.3g -> threshold %.4e empties the pool; using all %d frozen bits\n",
                cfg.add_rho_keep_quantile, add.threshold, length(add.full))
    end

    row(j) = @printf("%6d %12.4e %12.4e %12.4e %12.4e %12.4e\n",
                     j, Float64(rel[j]), stats.mean_abs_llr_dang[j],
                     stats.q10_abs_llr_dang[j], stats.sign_err_rate_dang[j], stats.rho[j])
    @printf("%6s %12s %12s %12s %12s %12s\n",
            "j", "reliability", "|LLR|_d", "q10|LLR|_d", "sign_err_d", "rho")
    foreach(row, first(add.pool, min(max_rows, length(add.pool))))

    if add.filtered
        kept = Set(add.pool)
        excluded = sort([j for j in add.full if !(j in kept)]; by = j -> (-stats.rho[j], j))
        if !isempty(excluded)
            println("  ... excluded by the rho filter (largest rho first):")
            foreach(row, first(excluded, min(8, length(excluded))))
        end
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

"""
    _stage1_candidate_sets(A, N, res, cfg, rel; swap_size)

Form all `swap_size`-swaps between the removal pool R (largest delta) and the
add pool P (rho filter + q10|LLR| ranking), sorted by total delta removed
(descending), then total reliability added (descending).
"""
function _stage1_candidate_sets(A::Vector{Int}, N::Int, res, cfg::PipelineConfig,
                                rel::Vector{Float64}; swap_size::Int)
    stats = _stage1_stats(res, cfg)

    remove_pool = sort(Int.(A); by = i -> _stage1_remove_sort_key(stats, rel, i))
    remove_pool = first(remove_pool, min(max(swap_size, cfg.nremove), length(remove_pool)))

    add_pool = _stage1_add_pool(stats, cfg, A, rel).pool
    add_pool = first(add_pool, min(max(swap_size, cfg.nadd), length(add_pool)))

    cands = NamedTuple[]
    for rem in _combos(remove_pool, swap_size)
        remset = Set(rem)
        base = [x for x in A if !(x in remset)]
        remove_delta = sum(stats.delta[i] for i in rem)
        for add in _combos(add_pool, swap_size)
            Anew = sort(vcat(base, add))
            length(Anew) == length(A) || continue
            push!(cands, (
                A = Anew,
                removes = copy(rem),
                adds = copy(add),
                remove_delta = remove_delta,
                add_reliability = sum(Float64(rel[j]) for j in add),
                add_rho = sum(stats.rho[j] for j in add),
            ))
        end
    end

    sort!(cands; by = c -> (-Float64(c.remove_delta), -Float64(c.add_reliability),
                            string(c.removes), string(c.adds)))
    return first(cands, min(cfg.max_kicker_candidates, length(cands)))
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

    # Batch mode (set_kicker_escalate = false): pool candidates from all kicker sizes,
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
        print_stage1_priority_report(res, cfg, rel)
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
                print_stage1_priority_report(res, cfg, rel)
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
Trial-level rescue score S(i, j) for every candidate single edge (i, j).

A trial is credited only when every dangerous path recorded for it clears the
threshold after the edge's Delta (eq. 17); lifting only some of the dangerous
paths would not change the decoding outcome.  With l0-aware scoring, a trial
in which l0 was pruned only counts targets j <= first_error + lost_window.
`n_paths_considered` counts trials.
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

"""
    _edge_set_saved_trials(res, edges; ...)

Trial-level rescue score S(E) for a set of candidate edges E.

For every dangerous path l of trial m, adding the edges flips the
post-precoder bit u_j of each target j whose number of active sources is
odd, i.e. when XOR_{(i,j) in E} v_i(l) = 1.  Each such flip changes BM_j by
exactly (1 - 2 u_j) * LLR_j (eq. 17).  Two edges into the same target on a
path carrying both sources therefore cancel, and contribute nothing.

A trial is counted as saved only if D_{l,m} + Delta_{l,m}(E) > threshold for
every dangerous path l in that trial.  When l0 was pruned from the list
(path-loss error), a target j only contributes if j <= e_{l,m} + lost_window,
where e_{l,m} is the first error of path l.
"""
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

    # Group the edges by target once: target j => its source bits.
    targets = unique(Int(e.j) for e in edges)
    sources = Dict(j => [Int(e.i) for e in edges if e.j == j] for j in targets)

    @inbounds for rg in ranges
        l0_survived = _sample_l0_survived(sel[first(rg)])
        saved_gated = true
        saved_ungated = true
        for p in rg
            s = sel[p]
            Δg = 0.0
            Δa = 0.0
            for j in targets
                # Parity of the active sources feeding target j on this path.
                flips = false
                for i in sources[j]
                    s.path_v_scl[i] != 0 && (flips = !flips)
                end
                flips || continue
                Δj = (1.0 - 2.0 * Float64(s.B[j])) * Float64(s.LLR[j])
                Δa += Δj
                elig = !l0_aware || l0_survived || j <= s.first_error + lost_window
                elig && (Δg += Δj)
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
    _score_k_edge_sets(res, top_single; kicker_size, ...)

Form all `kicker_size`-subsets of the top single edges and score each by its
joint trial-level rescue count S(E).  (The field is still named `saved_paths`
for compatibility with the output CSV; it counts saved trials.)
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
                             N::Int = 0)
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
        stats = _edge_set_saved_trials(res, edges;
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

function _replay_screen(cfg::PipelineConfig, A::AbstractVector{Int},
                        T_try::AbstractMatrix{<:Integer}, buffer)
    cfg_scl = DPS.make_config(cfg.N, sort(Int.(A)), cfg.L, T_try;
                              depths = DPS.infer_depths(T_try))
    out = DPS.replay_decode(cfg_scl, buffer)
    return (n_errors = out.n_errors,
            fer_est = out.n_errors / max(buffer.n_source_trials, 1))
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
        println("pac_max_span                      = ", _pac_span(cfg))
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

        # Trial-level rescue score S(i, j) for every candidate single edge.
        single = _score_single_edges_trialwise(res, T, N, A;
                             max_span = _pac_span(cfg),
                             target = cfg.pac_target,
                             only_dangerous = cfg.pac_only_dangerous,
                             l0_aware = cfg.pac_l0_aware_scoring,
                             lost_window = cfg.pac_l0_lost_window,
                             kept_weight = cfg.pac_l0_kept_weight,
                             lost_weight = cfg.pac_l0_lost_weight,
                             late_penalty = cfg.pac_l0_late_penalty)
        saved = single.score

        # delta_i of each source under the current T (diagnostic column only).
        delta = _stage1_support_vectors(res, N).delta

        single_cands = NamedTuple[]
        Aset = Set(Int.(A))
        for i in A
            j_max = min(N, i + _pac_span(cfg))
            for j in (i + 1):j_max
                if cfg.pac_target == :frozen
                    in(j, Aset) && continue
                elseif cfg.pac_target != :all
                    throw(ArgumentError("pac_target must be :frozen or :all."))
                end
                T[i, j] != 0 && continue
                saved[i, j] > 0 || continue
                push!(single_cands, (i = Int(i), j = Int(j),
                                     delta = Float64(delta[i]),
                                     saved_paths = Float64(saved[i, j]),
                                     saved_kept = Float64(single.saved_kept[i, j]),
                                     saved_lost_urgent = Float64(single.saved_lost_urgent[i, j]),
                                     saved_lost_late = Float64(single.saved_lost_late[i, j])))
            end
        end

        if isempty(single_cands)
            cfg.verbose && println("No single-edge candidates rescue any trial; stopping.")
            break
        end

        sort!(single_cands, by = c -> (-c.saved_paths, -c.saved_kept,
                                        -c.saved_lost_urgent, c.saved_lost_late))
        top_single_print = first(single_cands, min(cfg.pac_top_edges_per_step, length(single_cands)))
        top_single_pool  = first(single_cands, min(max(cfg.pac_top_edges_per_step,
                                                       cfg.pac_kicker_single_pool),
                                                   length(single_cands)))

        cfg.verbose && begin
            println("Top single-edge candidates (ranked by rescued trials S(i, j))")
            println("  L0-aware over $(single.n_paths_considered) dangerous trials; kept=$(single.n_l0_kept), lost=$(single.n_l0_lost)")
            for c in top_single_print
                @printf("  (%3d -> %3d)  delta_i=%.4f  S=%.0f  kept=%.0f  lost_urgent=%.0f  lost_late=%.0f\n",
                        c.i, c.j, c.delta, c.saved_paths,
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
                                        N = N)
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
                                  stdout_log_path = nothing)
    open(path, "w") do io
        println(io, "DPS_Search complete parameter snapshot")
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
        _write_key_value(io, "generic_taps", cfg.generic_taps)
        _write_key_value(io, "noise_sigma", noise_sigma_from_ebn0db(cfg.N, cfg.K, cfg.EbN0_dB))
        _write_key_value(io, "set_score_train", set_out.score)
        _write_key_value(io, "n_stage1_history_rows", length(set_out.history))
        _write_key_value(io, "pac_n_edges_added", length(design.edges))
        _write_key_value(io, "pac_edges", design.edges)
        _write_key_value(io, "pac_final_objective_train", design.score)
        _write_key_value(io, "pac_working_objective_train", design.working_score)
        _write_key_value(io, "pac_best_step", design.best_step)
        _write_key_value(io, "pac_explore_used", design.explore_used)
        _write_key_value(io, "final_identity_objective", evaluation.score_identity)
        _write_key_value(io, "final_generic_objective", evaluation.score_generic)
        _write_key_value(io, "final_constructed_objective", evaluation.score_built)
        println(io)

        println(io, "[saved files in this output directory]")
        println(io, "A_selected.txt")
        println(io, "generic_taps.txt")
        println(io, "set_history.csv")
        println(io, "pac_edges.csv")
        println(io, "summary.txt")
        println(io, "parameters_complete.txt")
        if stdout_log_path !== nothing
            println(io, "dps_search_stdout_stderr.log")
        end
        if cfg.save_matrices
            println(io, "T_generic.csv")
            println(io, "T_constructed.csv")
            println(io, "T_working_last.csv")
            println(io, "depths_generic.txt")
            println(io, "depths_constructed.txt")
        end
    end
    return path
end

function save_pipeline_outputs(cfg::PipelineConfig, set_out, design, evaluation;
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
        writedlm(joinpath(outdir, "T_working_last.csv"), Int.(design.Tmat_working), ',')
        save_vector(joinpath(outdir, "depths_generic.txt"), set_out.depths_generic)
        save_vector(joinpath(outdir, "depths_constructed.txt"), design.depths)
    end

    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "DPS_Search summary")
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
        println(io, "n_trials_set = ", cfg.n_trials_set)
        println(io, "n_trials_candidate = ", cfg.n_trials_candidate)
        println(io, "nremove = ", cfg.nremove)
        println(io, "nadd = ", cfg.nadd)
        println(io, "kicker_sizes = ", cfg.kicker_sizes)
        println(io, "set_kicker_escalate = ", cfg.set_kicker_escalate)
        println(io, "set_full_eval_top_k = ", cfg.set_full_eval_top_k)
        println(io, "set_full_eval_only_small_improvers = ", cfg.set_full_eval_only_small_improvers)
        println(io, "add_rho_keep_quantile = ", cfg.add_rho_keep_quantile)
        println(io, "add_strategy = ", cfg.add_strategy)
        println(io, "pac_initial = ", cfg.pac_initial)
        println(io, "pac_max_edges = ", cfg.pac_max_edges)
        println(io, "pac_max_span = ", _pac_span(cfg))
        println(io, "pac_target = ", cfg.pac_target)
        println(io, "pac_n_trials_small = ", cfg.pac_n_trials_small)
        println(io, "pac_n_trials_full = ", cfg.pac_n_trials_full)
        println(io, "pac_kicker_sizes = ", cfg.pac_kicker_sizes)
        println(io, "pac_candidate_sets_per_step = ", cfg.pac_candidate_sets_per_step)
        println(io, "pac_full_confirm_top_candidates = ", cfg.pac_full_confirm_top_candidates)
        println(io, "pac_explore_when_stalled = ", cfg.pac_explore_when_stalled)
        println(io, "pac_max_explore_steps = ", cfg.pac_max_explore_steps)
        println(io, "pac_return_best_seen = ", cfg.pac_return_best_seen)
        println(io, "pac_l0_aware_scoring = ", cfg.pac_l0_aware_scoring)
        println(io, "pac_l0_lost_window = ", cfg.pac_l0_lost_window)
        println(io, "pac_l0_kept_weight = ", cfg.pac_l0_kept_weight)
        println(io, "pac_l0_lost_weight = ", cfg.pac_l0_lost_weight)
        println(io, "pac_l0_late_penalty = ", cfg.pac_l0_late_penalty)
        println(io, "pac_replay_screen = ", cfg.pac_replay_screen)
        println(io, "pac_replay_delta = ", cfg.pac_replay_delta)
        println(io, "n_trials_eval = ", cfg.n_trials_eval)
        println(io, "seed = ", cfg.seed)
        println(io, "eval_seed = ", cfg.eval_seed)
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
                             stdout_log_path = stdout_log_path)

    if stdout_log_path !== nothing && isfile(stdout_log_path)
        try
            cp(stdout_log_path, joinpath(outdir, "dps_search_stdout_stderr.log"); force = true)
        catch err
            @warn "Could not copy stdout/stderr log into final output directory" stdout_log_path outdir err
        end
    end

    return outdir
end

function run_pipeline(cfg::PipelineConfig = PipelineConfig(); stdout_log_path = nothing)
    println("============================================================")
    println("Stage 1: information-set design under generic PAC (Algorithm 2)")
    println("Stage 2: sparse PAC precoder construction (Algorithm 3)")
    println("============================================================")
    println("N = ", cfg.N, ", K = ", cfg.K, ", EbN0_dB = ", cfg.EbN0_dB, ", L = ", cfg.L)
    println("generic taps         = ", cfg.generic_taps)
    println("reliability          = ", cfg.reliability_method)
    println("set_objective        = ", cfg.set_objective)
    println("kicker_sizes (set)   = ", cfg.kicker_sizes, "  escalate = ", cfg.set_kicker_escalate)
    println("add pool             = ", cfg.add_strategy, "  rho keep quantile = ", cfg.add_rho_keep_quantile)
    println("pac_initial          = ", cfg.pac_initial)
    println("pac_max_edges        = ", cfg.pac_max_edges, "  (max_span = ", _pac_span(cfg), ")")
    println("pac_target           = ", cfg.pac_target)
    println("pac_kicker_sizes     = ", cfg.pac_kicker_sizes)
    println("pac batch confirms   = ", cfg.pac_full_confirm_top_candidates, " of ", cfg.pac_candidate_sets_per_step)
    println("pac replay screen    = ", cfg.pac_replay_screen, "  delta = ", cfg.pac_replay_delta)
    println("pac exploration      = ", cfg.pac_explore_when_stalled, "  return_best_seen = ", cfg.pac_return_best_seen)
    println("pac L0-aware scoring = ", cfg.pac_l0_aware_scoring, "  lost_window = ", cfg.pac_l0_lost_window)
    println("noise_sigma          = ", noise_sigma_from_ebn0db(cfg.N, cfg.K, cfg.EbN0_dB))

    set_out    = run_generic_pac_set_design(cfg)
    design     = run_iterative_pac_design(cfg, set_out)
    evaluation = evaluate_constructed_precoder(cfg, set_out.A, set_out.Tgeneric, design)

    outdir = save_pipeline_outputs(cfg, set_out, design, evaluation;
                                   stdout_log_path = stdout_log_path)
    println("\nSaved outputs to: ", outdir)

    return (config = cfg, set = set_out, design = design,
            evaluation = evaluation, outdir = outdir)
end


function run_pipeline_with_output_log(cfg::PipelineConfig = PipelineConfig())
    logpath = _active_log_path(cfg)
    result_ref = Ref{Any}(nothing)

    # Julia 1.12 performs stdout/stderr redirection at the file-descriptor
    # level and does not accept arbitrary custom IO objects here.  Therefore
    # we redirect to the real log file directly rather than to a tee.  This
    # preserves all experiment logic; only the destination of stdout/stderr is
    # changed while run_pipeline executes.
    original_stdout = stdout
    original_stderr = stderr
    println(original_stdout, "DPS_Search stdout/stderr log = ", logpath)
    flush(original_stdout)

    open(logpath, "w") do logfile
        redirect_stdout(logfile) do
            redirect_stderr(logfile) do
                try
                    println("DPS_Search stdout/stderr log = ", logpath)
                    result_ref[] = run_pipeline(cfg; stdout_log_path = logpath)
                    if result_ref[] !== nothing && hasproperty(result_ref[], :outdir)
                        flush(stdout)
                        flush(stderr)
                        flush(logfile)
                        try
                            cp(logpath, joinpath(result_ref[].outdir, "dps_search_stdout_stderr.log"); force = true)
                        catch copy_err
                            @warn "Could not finalize stdout/stderr log copy" logpath result_ref[].outdir copy_err
                        end
                    end
                catch err
                    println(stderr, "\nERROR while running DPS_Search. Stack trace follows.")
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
        println(original_stdout, "DPS_Search stdout/stderr log copied to: ", joinpath(result_ref[].outdir, "dps_search_stdout_stderr.log"))
    else
        println(original_stdout, "DPS_Search stdout/stderr log saved to: ", logpath)
    end
    flush(original_stdout)

    return result_ref[]
end

# ------------------------------------------------------------
# Run block.  Settings that changed between the paper's constructions
# (code parameters, initial set, swap/edge-set sizes, trial counts) are set
# here explicitly; everything else uses the PipelineConfig defaults.
#
# Example: the [64, 32, L = 32] code at 3.5 dB.  Information sets are 1-based.
# ------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    N   = 64
    K   = 32
    L   = 32
    SNR = 3.5

    cfg = PipelineConfig(
        N = N, K = K, L = L, EbN0_dB = SNR,

        # Initial set: K most reliable DEGA positions (default), or pass one
        # explicitly, e.g. initial_A = Misc.rm_polar(N, K, SNR).
        initial_reliability_method = :dega,

        # Stage 1
        kicker_sizes       = [1],
        n_trials_set       = 300_000,
        n_trials_candidate = 100_000,

        # Stage 2
        pac_kicker_sizes   = [1, 2],
        pac_n_trials_small = 200_000,
        pac_n_trials_full  = 400_000,

        # Final evaluation
        n_trials_eval      = 300_000,
    )

    run_pipeline_with_output_log(cfg)
end
