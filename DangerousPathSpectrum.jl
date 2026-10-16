module DangerousPathSpectrumClean

# ============================================================
# DangerousPathSpectrumClean.jl
#
# Decoder-dependent tail-event / PM-gap spectrum tools for
# polar/PAC SCL experiments.
#
# This file is designed to sit next to your existing
# ClassicalPolarCodeFuncSCD.jl decoder.
#
# Includes:
#   * lambda spectrum  (per-bit "where errors start")
#   * sigma spectrum   (per-bit "which bits dangerous paths use")
#   * suggest_tail_swaps_v2     single-swap nominator (legacy compatible)
#   * suggest_tail_swaps_kicker k-swap nominator for kicker escalation
# ============================================================

using Random
using Statistics
using Printf
using Base.Threads

include("ClassicalPolarCodeFuncSCD.jl")
const PC = ClassicalPolarCodeFuncSCD

const UIntB = UInt8

export DangerousEvent, PathSample, SpectrumResult,
       infer_depths, generate_allzero_awgn_llrs,
       collect_dangerous_events, forced_scd_all!, greedy_scd_with_extra_frozen!,
       run_path_samples, dangerous_path_spectrum,
       aggregate_samples, lambda_spectrum,
       dq_summary, tilted_weights, tilted_stats,
       risk_by_bit, tail_risk_objective, suggest_tail_swaps,
       influence_matrix, print_spectrum_report,
       # sigma additions
       support_frequency, support_frequency_per_event,
       support_risk_by_bit, combined_risk_by_bit,
       suggest_tail_swaps_v2,
       # kicker
       suggest_tail_swaps_kicker,
       # replay buffer (paired re-decoding of the failure/near-miss population)
       ReplayBuffer, build_replay_buffer, replay_decode,
       replay_error_count, replay_fer

# ------------------------------------------------------------
# Data containers
# ------------------------------------------------------------

struct DangerousEvent
    trial::Int
    rank::Int
    first_error::Int
    path_v::Vector{UIntB}
    path_pm_scl::Float64
    correct_pm_scl::Float64
    scl_gap::Float64
    correct_in_list::Bool
end

struct PathSample{T<:AbstractFloat}
    trial::Int
    rank::Int
    first_error::Int
    mode::Symbol
    path_v_scl::Vector{UIntB}
    decoded_v::Vector{UIntB}
    B::Vector{UIntB}
    LLR::Vector{T}
    BM::Vector{T}
    PM_path::Float64
    PM_correct::Float64
    D::Float64
    scl_gap::Float64
end

struct SpectrumResult{T<:AbstractFloat}
    N::Int
    K::Int
    A::Vector{Int}
    sigma::Float64
    L::Int
    n_trials::Int
    n_dismissed::Int
    n_kept::Int
    n_trials_with_events::Int
    events::Vector{DangerousEvent}
    samples_by_mode::Dict{Symbol, Vector{PathSample{T}}}
    summaries::Dict{Symbol, Any}
    lambda::Any
    # Optional ReplayBuffer built from the same trials (nothing unless the
    # caller asked for it via collect_replay = true).
    replay::Any
end

# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

@inline function safe_log1pexp(x::Real)
    xf = Float64(x)
    return xf > 0.0 ? xf + log1p(exp(-xf)) : log1p(exp(xf))
end

function require_power_of_two(N::Int)
    n = round(Int, log2(N))
    if (1 << n) != N
        throw(ArgumentError("N must be a power of two."))
    end
    return n
end

function logmeanexp(x::AbstractVector{<:Real})
    isempty(x) && return NaN
    m = maximum(x)
    if !isfinite(m)
        return m
    end
    return m + log(mean(exp.(Float64.(x) .- m)))
end

function lower_cvar(x::AbstractVector{<:Real}, alpha::Real)
    isempty(x) && return NaN
    q = quantile(Float64.(x), alpha)
    vals = Float64[v for v in x if v <= q]
    return isempty(vals) ? q : mean(vals)
end

function vector_mean_var(xs::Vector{Vector{T}}, N::Int) where {T<:AbstractFloat}
    if isempty(xs)
        return (mean = zeros(Float64, N), var = zeros(Float64, N), count = 0)
    end
    M = reduce(hcat, xs)
    m = vec(mean(M, dims = 2))
    v = vec(var(M, dims = 2; corrected = false))
    return (mean = Float64.(m), var = Float64.(v), count = length(xs))
end

function bitvector_mean_var(xs::Vector{Vector{UIntB}}, N::Int)
    if isempty(xs)
        return (mean = zeros(Float64, N), var = zeros(Float64, N), count = 0)
    end
    M = reduce(hcat, [Float64.(x) for x in xs])
    m = vec(mean(M, dims = 2))
    v = vec(var(M, dims = 2; corrected = false))
    return (mean = Float64.(m), var = Float64.(v), count = length(xs))
end

# Generate all k-element combinations from xs (without replacement, sorted).
function _combinations(xs::AbstractVector{T}, k::Int) where {T}
    n = length(xs)
    k > n && return Vector{Vector{T}}()
    k == 0 && return [T[]]
    result = Vector{Vector{T}}()
    buf = T[]
    _combinations_helper!(result, buf, xs, 1, k)
    return result
end

function _combinations_helper!(result::Vector{Vector{T}}, current::Vector{T},
                                 xs::AbstractVector{T}, start::Int, k::Int) where {T}
    if length(current) == k
        push!(result, copy(current))
        return
    end
    n = length(xs)
    remaining = k - length(current)
    @inbounds for i in start:(n - remaining + 1)
        push!(current, xs[i])
        _combinations_helper!(result, current, xs, i + 1, k)
        pop!(current)
    end
    return
end

# ------------------------------------------------------------
# T-matrix depth helper and all-zero AWGN LLR generation
# ------------------------------------------------------------

function infer_depths(Tmat::AbstractMatrix{<:Integer})
    N = size(Tmat, 1)
    size(Tmat, 2) == N || throw(ArgumentError("Tmat must be square."))
    depths = zeros(Int, N)
    @inbounds for j in 1:N
        first_nz = j
        found = false
        for r in 1:j
            if isodd(Int(Tmat[r, j]))
                first_nz = r
                found = true
                break
            end
        end
        depths[j] = found ? (j - first_nz) : 0
    end
    return depths
end

function generate_allzero_awgn_llrs(
    sigma::Real,
    N::Int,
    n_trials::Int;
    T::Type{<:AbstractFloat} = Float64,
    seed::Int = 7,
)
    sigma = Float64(sigma)
    llrs = Vector{Vector{T}}(undef, n_trials)
    rngs = [MersenneTwister(seed + t) for t in 1:n_trials]

    @threads for t in 1:n_trials
        rng = rngs[t]
        y = 1.0 .+ sigma .* randn(rng, N)
        llrs[t] = T.(2.0 .* y ./ (sigma * sigma))
    end

    return llrs
end

function make_config(
    N::Int,
    A::AbstractVector{<:Integer},
    L::Int,
    Tmat::AbstractMatrix{<:Integer};
    depths::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    T::Type{<:AbstractFloat} = Float64,
)
    n = require_power_of_two(N)
    size(Tmat, 1) == N && size(Tmat, 2) == N ||
        throw(ArgumentError("Tmat must be N by N."))

    F = sort(setdiff(collect(1:N), Int.(A)))
    FB = falses(N); FB[F] .= true
    d = depths === nothing ? infer_depths(Tmat) : Int.(collect(depths))
    length(d) == N || throw(ArgumentError("depths must have length N."))

    return PC.Config{T}(n, N, n + 1, L,
                        collect(Bool.(FB)),
                        d,
                        Matrix{Int}(Tmat),
                        Bool(FB[N]),
                        T)
end

# ------------------------------------------------------------
# Existing-decoder SCD rerun helpers
# ------------------------------------------------------------

function leaf_B_from_v(cfg::PC.Config, v::AbstractVector{UIntB})
    N = cfg.N
    length(v) == N || throw(ArgumentError("v must have length N."))
    B = zeros(UIntB, N)
    @inbounds for j in 1:N
        depth = cfg.depths[j]
        bj = UIntB(v[j] & 0x01)
        if depth > 0
            for r in (j - depth):(j - 1)
                if isodd(cfg.Tmat[r, j])
                    bj = bj ⊻ UIntB(v[r] & 0x01)
                end
            end
        end
        B[j] = bj
    end
    return B
end

function bm_from_B_LLR(B::AbstractVector{UIntB}, LLR::AbstractVector{T}) where {T<:AbstractFloat}
    N = length(B)
    length(LLR) == N || throw(ArgumentError("B and LLR must have the same length."))
    BM = Vector{T}(undef, N)
    @inbounds for j in 1:N
        u = Int(B[j])
        metric_arg = -(1 - 2u) * Float64(LLR[j])
        BM[j] = T(safe_log1pexp(metric_arg))
    end
    return BM
end

function collect_decode_output(cfg::PC.Config{T}, st::PC.State{T}) where {T<:AbstractFloat}
    N = cfg.N
    v = Vector{UIntB}(st.vx[1:N, 1])
    LLR = Vector{T}(st.temp_LLRs[1:N])
    B = leaf_B_from_v(cfg, v)
    BM = bm_from_B_LLR(B, LLR)
    PM = sum(Float64.(BM))
    return (v = v, B = B, LLR = LLR, BM = BM, PM = PM)
end

function forced_scd_all!(
    cfg::PC.Config{T},
    st::PC.State{T},
    llr::AbstractVector{T},
    syndrome::AbstractVector{UIntB},
) where {T<:AbstractFloat}
    N = cfg.N
    length(syndrome) == N || throw(ArgumentError("syndrome must have length N."))
    PC.reset!(cfg, st)
    PC.set_llr!(cfg, st, Vector{T}(llr))
    PC.set_force_decode!(cfg, st, syndrome)
    PC.decode_channel_all!(cfg, st)
    return collect_decode_output(cfg, st)
end

function greedy_scd_with_extra_frozen!(
    cfg::PC.Config{T},
    st::PC.State{T},
    llr::AbstractVector{T},
    syndrome::AbstractVector{UIntB},
) where {T<:AbstractFloat}
    N = cfg.N
    length(syndrome) == N || throw(ArgumentError("syndrome must have length N."))
    PC.reset!(cfg, st)
    PC.set_llr!(cfg, st, Vector{T}(llr))
    PC.set_syndrome!(cfg, st, syndrome)
    PC.decode_channel_all!(cfg, st)
    return collect_decode_output(cfg, st)
end

function prefix_error_config_cache(
    cfg_base::PC.Config{T},
    Tmat::AbstractMatrix{<:Integer};
    depths::Union{Nothing, AbstractVector{<:Integer}} = nothing,
) where {T<:AbstractFloat}
    N = cfg_base.N
    cache = Dict{Int, PC.Config{T}}()
    for i in 1:N
        Ftmp = copy(cfg_base.F)
        Ftmp[1:i] .= true
        d = depths === nothing ? cfg_base.depths : Int.(collect(depths))
        cache[i] = PC.Config{T}(cfg_base.n, N, cfg_base.d, 1,
                                collect(Bool.(Ftmp)), collect(d),
                                Matrix{Int}(Tmat), Bool(Ftmp[N]), T)
    end
    return cache
end

function syndrome_for_mode(ev::DangerousEvent, N::Int, mode::Symbol)
    vals = zeros(UIntB, N)
    if mode == :explicit || mode == :explicit_path
        vals .= ev.path_v
    elseif mode == :single_error_forced || mode == :single_error_full
        vals[ev.first_error] = UIntB(1)
    elseif mode == :prefix_error || mode == :first_error_greedy || mode == :prefix_error_greedy
        vals[ev.first_error] = UIntB(1)
    else
        throw(ArgumentError("Unknown path mode $mode."))
    end
    return vals
end

mode_is_greedy_prefix(mode::Symbol) =
    mode in (:prefix_error, :first_error_greedy, :prefix_error_greedy)

# ------------------------------------------------------------
# SCL pass: identify dangerous paths
# ------------------------------------------------------------

function collect_dangerous_events(
    cfg_scl::PC.Config{T},
    llrs::AbstractVector{<:AbstractVector{T}};
    skip_all_positive::Bool = true,
) where {T<:AbstractFloat}
    n_trials = length(llrs)
    nt = nthreads()

    # Use Vector{Bool}, not falses(...).  `falses` creates a packed BitVector,
    # whose different logical elements can share the same machine word;
    # concurrent writes to separate trials would then race.
    dismissed = fill(false, n_trials)
    has_events = fill(false, n_trials)

    # Per-trial final-list margin: min over wrong final-list paths of
    # (PM_wrong - PM_correct).  Conventions:
    #   +Inf  -> dismissed trial (all-positive LLRs; provably decoded correct)
    #   -Inf  -> correct path absent from the final list (path-loss error)
    #   <= 0  -> selection error (some wrong path at or below the correct PM)
    #   >  0  -> correct decode; the value is the distance to the nearest
    #            wrong survivor, which is what the replay buffer thresholds on.
    # Plain Vector{Float64}: distinct trials write distinct machine words,
    # unlike a packed BitVector (see the comment above for dismissed).
    margins = fill(Inf, n_trials)

    # Do not index mutable decoder state by threadid().  Julia tasks may migrate
    # between worker threads under the default scheduler, and Slurm CPU ids are
    # not required to match Julia's logical thread ids.  Instead, each spawned
    # task owns exactly one decoder state and one event buffer.
    tasks = Vector{Task}(undef, nt)
    for worker_id in 1:nt
        tasks[worker_id] = Base.Threads.@spawn begin
            local_events = DangerousEvent[]
            st = PC.init_state(cfg_scl)

            @inbounds for trial in worker_id:nt:n_trials
                llr = llrs[trial]

                PC.reset!(cfg_scl, st)

                if skip_all_positive && minimum(llr) > zero(T)
                    dismissed[trial] = true
                    continue
                end

                PC.set_llr!(cfg_scl, st, Vector{T}(llr))
                PC.decode_channel_stats!(cfg_scl, st)

                correct_rank = 0
                for r in 1:st.l
                    if all(iszero, @view st.vx[1:cfg_scl.N, r])
                        correct_rank = r
                        break
                    end
                end

                n_dangerous = correct_rank == 0 ? st.l : correct_rank - 1
                if n_dangerous > 0
                    has_events[trial] = true
                end

                correct_pm = correct_rank == 0 ? NaN : Float64(st.PMs[correct_rank])
                correct_in_list = correct_rank != 0

                if correct_rank == 0
                    margins[trial] = -Inf
                else
                    best_gap = Inf
                    for r in 1:st.l
                        r == correct_rank && continue
                        g = Float64(st.PMs[r]) - correct_pm
                        g < best_gap && (best_gap = g)
                    end
                    margins[trial] = best_gap
                end

                for r in 1:n_dangerous
                    v_path = Vector{UIntB}(st.vx[1:cfg_scl.N, r])
                    fe = findfirst(!iszero, v_path)
                    fe === nothing && continue

                    path_pm = Float64(st.PMs[r])
                    gap = correct_in_list ? (path_pm - correct_pm) : NaN

                    push!(local_events, DangerousEvent(
                        trial, r, Int(fe), v_path, path_pm, correct_pm, gap, correct_in_list,
                    ))
                end
            end

            local_events
        end
    end

    events = DangerousEvent[]
    for task in tasks
        append!(events, fetch(task))
    end
    sort!(events, by = ev -> (ev.trial, ev.rank))

    meta = (
        n_trials = n_trials,
        n_dismissed = count(dismissed),
        n_kept = n_trials - count(dismissed),
        n_trials_with_events = count(has_events),
        dismissed = dismissed,
        has_events = has_events,
        margins = margins,
    )

    return events, meta
end

# ------------------------------------------------------------
# Replay buffer
#
# The FER delta of a precoder change is carried almost entirely by the
# trials that are currently errors (margin <= 0) plus the near-misses
# (0 < margin <= delta): those are the only trials a modified T can
# plausibly flip in either direction at moderate |Delta PM|.  Storing
# their channel LLR vectors lets any candidate T be evaluated by an
# exact SCL re-decode of just this sub-population:
#
#   * exact under the new T — the list re-optimizes, so wrong-path
#     evasion / adversary adaptation is priced in, unlike the frozen-
#     sample counterfactual Delta(i, j);
#   * paired — every candidate sees the same noise realizations, so
#     candidate-vs-candidate comparisons have no fresh MC variance;
#   * two-sided — it is the only cheap screen that can detect a move
#     BREAKING a currently-safe near-miss trial (e.g., a tap removal
#     that deletes a penalty some wrong path is currently paying).
#
# Trials with margin > delta are assumed unchanged; the induced FER
# estimate is n_errors(buffer) / n_source_trials.  Choose delta around
# the largest plausible |Delta PM| of a single move (a conditional
# |LLR| scale), e.g. 10-20 at these operating points.
# ------------------------------------------------------------

struct ReplayBuffer{T<:AbstractFloat}
    trial_ids::Vector{Int}
    llrs::Vector{Vector{T}}
    margins::Vector{Float64}   # margins under the T the buffer was built with
    n_source_trials::Int
    delta::Float64
end

function build_replay_buffer(
    llrs::AbstractVector{<:AbstractVector{T}},
    margins::AbstractVector{<:Real},
    delta::Real,
) where {T<:AbstractFloat}
    length(llrs) == length(margins) ||
        throw(ArgumentError("llrs and margins must have the same length."))
    d = Float64(delta)
    ids = [t for t in eachindex(margins) if Float64(margins[t]) <= d]
    return ReplayBuffer{T}(
        ids,
        Vector{T}[Vector{T}(llrs[t]) for t in ids],
        Float64[Float64(margins[t]) for t in ids],
        length(margins),
        d,
    )
end

"""
    replay_decode(cfg_scl, buffer)

Re-run the SCL decoder on every trial in the buffer under the (possibly
modified) configuration `cfg_scl`.  Returns

    (n_errors, errors, margins)

where `errors[k]` is true when the all-zero path is not the final PM
winner for buffer trial k, and `margins[k]` is the new final-list margin
under `cfg_scl` (same conventions as `collect_dangerous_events`).
"""
function replay_decode(cfg_scl::PC.Config{T}, buffer::ReplayBuffer{T}) where {T<:AbstractFloat}
    nb = length(buffer.trial_ids)
    errors = fill(false, nb)
    margins = fill(Inf, nb)
    nt = nthreads()

    tasks = Vector{Task}(undef, nt)
    for worker_id in 1:nt
        tasks[worker_id] = Base.Threads.@spawn begin
            st = PC.init_state(cfg_scl)
            @inbounds for k in worker_id:nt:nb
                llr = buffer.llrs[k]
                PC.reset!(cfg_scl, st)
                PC.set_llr!(cfg_scl, st, Vector{T}(llr))
                PC.decode_channel_stats!(cfg_scl, st)

                correct_rank = 0
                for r in 1:st.l
                    if all(iszero, @view st.vx[1:cfg_scl.N, r])
                        correct_rank = r
                        break
                    end
                end

                if correct_rank == 0
                    errors[k] = true
                    margins[k] = -Inf
                else
                    errors[k] = correct_rank != 1
                    correct_pm = Float64(st.PMs[correct_rank])
                    best_gap = Inf
                    for r in 1:st.l
                        r == correct_rank && continue
                        g = Float64(st.PMs[r]) - correct_pm
                        g < best_gap && (best_gap = g)
                    end
                    margins[k] = best_gap
                end
            end
            nothing
        end
    end
    foreach(fetch, tasks)

    return (n_errors = count(errors), errors = errors, margins = margins)
end

replay_error_count(cfg_scl::PC.Config, buffer::ReplayBuffer) =
    replay_decode(cfg_scl, buffer).n_errors

"""
    replay_fer(cfg_scl, buffer)

FER estimate induced by the buffer re-decode, assuming every trial
outside the buffer keeps its current (correct) outcome.
"""
replay_fer(cfg_scl::PC.Config, buffer::ReplayBuffer) =
    replay_error_count(cfg_scl, buffer) / max(buffer.n_source_trials, 1)

# ------------------------------------------------------------
# SCD reruns for dangerous events
# ------------------------------------------------------------

function correct_pm_for_trial!(
    cache::Dict{Int, Float64},
    cfg_scd::PC.Config{T},
    st::PC.State{T},
    llr::AbstractVector{T},
    trial::Int,
) where {T<:AbstractFloat}
    if haskey(cache, trial)
        return cache[trial]
    end
    N = cfg_scd.N
    syndrome0 = zeros(UIntB, N)
    out = forced_scd_all!(cfg_scd, st, llr, syndrome0)
    cache[trial] = out.PM
    return out.PM
end

function run_path_samples(
    cfg_scd::PC.Config{T},
    llrs::AbstractVector{<:AbstractVector{T}},
    events::Vector{DangerousEvent};
    modes::Tuple = (:explicit, :prefix_error),
    Tmat::AbstractMatrix{<:Integer},
    depths::Union{Nothing, AbstractVector{<:Integer}} = nothing,
) where {T<:AbstractFloat}
    cfg_scd.L == 1 || throw(ArgumentError("cfg_scd must have L = 1 for SCD reruns."))

    mode_vec = Symbol.(collect(modes))
    prefix_cfgs = any(mode_is_greedy_prefix, mode_vec) ?
                  prefix_error_config_cache(cfg_scd, Tmat; depths = depths) :
                  Dict{Int, PC.Config{T}}()

    nt = nthreads()

    # As in collect_dangerous_events, keep all mutable decoder states and output
    # buffers task-local rather than threadid-local.  This avoids races if Julia
    # migrates tasks between threads.
    tasks = Vector{Task}(undef, nt)
    for worker_id in 1:nt
        tasks[worker_id] = Base.Threads.@spawn begin
            local_samples = Dict{Symbol, Vector{PathSample{T}}}(
                m => PathSample{T}[] for m in mode_vec
            )
            correct_cache = Dict{Int, Float64}()
            st_forced = PC.init_state(cfg_scd)
            prefix_states = Dict{Int, PC.State{T}}()

            @inbounds for idx in worker_id:nt:length(events)
                ev = events[idx]
                llr = Vector{T}(llrs[ev.trial])

                pm_correct = correct_pm_for_trial!(correct_cache, cfg_scd, st_forced, llr, ev.trial)

                for mode in mode_vec
                    syndrome = syndrome_for_mode(ev, cfg_scd.N, mode)

                    out = if mode_is_greedy_prefix(mode)
                        cfgp = prefix_cfgs[ev.first_error]
                        if !haskey(prefix_states, ev.first_error)
                            prefix_states[ev.first_error] = PC.init_state(cfgp)
                        end
                        stp = prefix_states[ev.first_error]
                        greedy_scd_with_extra_frozen!(cfgp, stp, llr, syndrome)
                    else
                        forced_scd_all!(cfg_scd, st_forced, llr, syndrome)
                    end

                    D = out.PM - pm_correct

                    push!(local_samples[mode], PathSample{T}(
                        ev.trial, ev.rank, ev.first_error, mode,
                        copy(ev.path_v), copy(out.v), copy(out.B), copy(out.LLR), copy(out.BM),
                        out.PM, pm_correct, D, ev.scl_gap,
                    ))
                end
            end

            local_samples
        end
    end

    samples = Dict{Symbol, Vector{PathSample{T}}}(m => PathSample{T}[] for m in mode_vec)
    for task in tasks
        local_samples = fetch(task)
        for m in mode_vec
            append!(samples[m], local_samples[m])
        end
    end
    for m in mode_vec
        sort!(samples[m], by = s -> (s.trial, s.rank))
    end

    return samples
end

# ------------------------------------------------------------
# Lambda and dQ summaries
# ------------------------------------------------------------

function lambda_spectrum(
    events::Vector{DangerousEvent},
    A::AbstractVector{<:Integer},
    n_trials::Int,
    n_kept::Int,
    n_trials_with_events::Int,
    L::Int,
)
    counts = Dict{Int, Int}(Int(i) => 0 for i in A)
    for ev in events
        counts[ev.first_error] = get(counts, ev.first_error, 0) + 1
    end

    lambda_true = Dict{Int, Float64}(i => counts[i] / max(n_trials, 1) for i in keys(counts))
    lambda_kept = Dict{Int, Float64}(i => counts[i] / max(n_kept, 1) for i in keys(counts))
    lambda_total_true = length(events) / max(n_trials, 1)
    lambda_total_kept = length(events) / max(n_kept, 1)
    fer_true = n_trials_with_events / max(n_trials, 1)
    fer_kept = n_trials_with_events / max(n_kept, 1)

    return (
        counts = counts,
        lambda_true = lambda_true,
        lambda_kept = lambda_kept,
        lambda_total_true = lambda_total_true,
        lambda_total_kept = lambda_total_kept,
        fer_true = fer_true,
        fer_kept = fer_kept,
        fer_bounds_true = (lower = lambda_total_true / max(L, 1), upper = lambda_total_true),
        fer_bounds_kept = (lower = lambda_total_kept / max(L, 1), upper = lambda_total_kept),
    )
end

function tilted_weights(D::AbstractVector{<:Real}, eta::Real)
    isempty(D) && return Float64[]
    x = -Float64(eta) .* Float64.(D)
    m = maximum(x)
    w = exp.(x .- m)
    s = sum(w)
    return w ./ s
end

function tilted_stats(D::AbstractVector{<:Real}, eta::Real)
    if isempty(D)
        return (mean = NaN, var = NaN, ess = 0.0)
    end
    d = Float64.(D)
    w = tilted_weights(d, eta)
    mu = sum(w .* d)
    va = sum(w .* (d .- mu).^2)
    ess = 1.0 / sum(w.^2)
    return (mean = mu, var = va, ess = ess)
end

function dq_summary(D::AbstractVector{<:Real}, eta_grid)
    d = Float64.(D)
    etas = Float64.(collect(eta_grid))

    if isempty(d)
        return (
            eta_chernoff = NaN, logM_chernoff = NaN, chernoff_bound = NaN,
            eta_boundary = NaN, tilted_mean = NaN, tilted_var = NaN, ess = 0.0,
            logM_by_eta = Float64[], tilted_mean_by_eta = Float64[], eta_grid = etas,
        )
    end

    logM = [logmeanexp(-eta .* d) for eta in etas]
    tmean = [tilted_stats(d, eta).mean for eta in etas]

    idx_ch = argmin(logM)
    idx_bd = argmin(abs.(tmean))
    ts = tilted_stats(d, etas[idx_ch])

    return (
        eta_chernoff = etas[idx_ch],
        logM_chernoff = logM[idx_ch],
        chernoff_bound = exp(logM[idx_ch]),
        eta_boundary = etas[idx_bd],
        tilted_mean = ts.mean,
        tilted_var = ts.var,
        ess = ts.ess,
        logM_by_eta = logM,
        tilted_mean_by_eta = tmean,
        eta_grid = etas,
    )
end

# ------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------

function aggregate_samples(
    samples::Vector{PathSample{T}},
    N::Int;
    eta_grid = range(0.0, 5.0, length = 51),
    alphas = (0.01, 0.05, 0.10),
    total_trials::Int = 1,
    kept_trials::Int = 1,
) where {T<:AbstractFloat}
    out = Dict{Int, Any}()
    first_errors = sort(unique(s.first_error for s in samples))

    for i in first_errors
        ss = [s for s in samples if s.first_error == i]
        D = Float64[s.D for s in ss]
        PM_path = Float64[s.PM_path for s in ss]
        PM_correct = Float64[s.PM_correct for s in ss]

        q = Dict{Float64, Float64}()
        cvar = Dict{Float64, Float64}()
        for a in alphas
            aa = Float64(a)
            q[aa] = isempty(D) ? NaN : quantile(D, aa)
            cvar[aa] = isempty(D) ? NaN : lower_cvar(D, aa)
        end

        llr_stats = vector_mean_var([s.LLR for s in ss], N)
        bm_stats = vector_mean_var([s.BM for s in ss], N)
        b_stats = bitvector_mean_var([s.B for s in ss], N)
        v_stats = bitvector_mean_var([s.decoded_v for s in ss], N)

        out[i] = (
            first_error = i,
            count = length(ss),
            lambda_true = length(ss) / max(total_trials, 1),
            lambda_kept = length(ss) / max(kept_trials, 1),
            D_mean = isempty(D) ? NaN : mean(D),
            D_var = length(D) <= 1 ? 0.0 : var(D; corrected = false),
            D_std = length(D) <= 1 ? 0.0 : std(D; corrected = false),
            P_D_le_0 = isempty(D) ? NaN : mean(D .<= 0.0),
            quantile = q,
            lower_cvar = cvar,
            PM_path_mean = isempty(PM_path) ? NaN : mean(PM_path),
            PM_path_var = length(PM_path) <= 1 ? 0.0 : var(PM_path; corrected = false),
            PM_correct_mean = isempty(PM_correct) ? NaN : mean(PM_correct),
            PM_correct_var = length(PM_correct) <= 1 ? 0.0 : var(PM_correct; corrected = false),
            LLR_stats = llr_stats,
            BM_stats = bm_stats,
            B_stats = b_stats,
            v_stats = v_stats,
            dq = dq_summary(D, eta_grid),
        )
    end

    return out
end

# ------------------------------------------------------------
# Top-level driver
# ------------------------------------------------------------

function dangerous_path_spectrum(
    N::Int,
    K::Int,
    A::AbstractVector{<:Integer},
    sigma::Real,
    n_trials::Int,
    L::Int,
    Tmat::AbstractMatrix{<:Integer};
    T::Type{<:AbstractFloat} = Float64,
    seed::Int = 7,
    depths::Union{Nothing, AbstractVector{<:Integer}} = nothing,
    modes::Tuple = (:explicit, :prefix_error),
    skip_all_positive::Bool = true,
    eta_grid = range(0.0, 5.0, length = 51),
    alphas = (0.01, 0.05, 0.10),
    collect_replay::Bool = false,
    replay_delta::Real = 10.0,
)
    Aint = sort(Int.(collect(A)))
    length(Aint) == K || @warn "length(A) != K" length_A=length(Aint) K=K

    cfg_scl = make_config(N, Aint, L, Tmat; depths = depths, T = T)
    cfg_scd = make_config(N, Aint, 1, Tmat; depths = depths, T = T)

    llrs = generate_allzero_awgn_llrs(sigma, N, n_trials; T = T, seed = seed)
    events, meta = collect_dangerous_events(cfg_scl, llrs;
                                            skip_all_positive = skip_all_positive)

    # Build the replay buffer from the same trials before llrs goes out of
    # scope.  Only the selected (error + near-miss) LLR vectors stay alive.
    replay = collect_replay ?
             build_replay_buffer(llrs, meta.margins, replay_delta) : nothing

    samples_by_mode = run_path_samples(cfg_scd, llrs, events; modes = modes, Tmat = Tmat, depths = depths)

    summaries = Dict{Symbol, Any}()
    for m in keys(samples_by_mode)
        summaries[m] = aggregate_samples(samples_by_mode[m], N;
                                         eta_grid = eta_grid,
                                         alphas = alphas,
                                         total_trials = meta.n_trials,
                                         kept_trials = meta.n_kept)
    end

    lam = lambda_spectrum(events, Aint, meta.n_trials, meta.n_kept,
                          meta.n_trials_with_events, L)

    return SpectrumResult{T}(
        N, K, Aint, Float64(sigma), L,
        meta.n_trials, meta.n_dismissed, meta.n_kept, meta.n_trials_with_events,
        events, samples_by_mode, summaries, lam, replay,
    )
end

# ============================================================
# Per-bit risk views (lambda)
# ============================================================

function risk_by_bit(result::SpectrumResult; mode::Symbol = :explicit, score::Symbol = :lambda_true)
    summaries = result.summaries[mode]
    out = Dict{Int, Float64}()

    for (i, s) in summaries
        if score == :lambda_true
            out[i] = s.lambda_true
        elseif score == :lambda_kept
            out[i] = s.lambda_kept
        elseif score == :count
            out[i] = Float64(s.count)
        elseif score == :D_cvar_05
            out[i] = get(s.lower_cvar, 0.05, NaN)
        elseif score == :D_quantile_05
            out[i] = get(s.quantile, 0.05, NaN)
        elseif score == :hybrid
            c = get(s.lower_cvar, 0.05, NaN)
            out[i] = s.lambda_true * max(0.0, -c)
        else
            throw(ArgumentError("Unknown score $score."))
        end
    end

    return out
end

function tail_risk_objective(result::SpectrumResult; mode::Symbol = :explicit, combine::Symbol = :fer)
    summaries = result.summaries[mode]

    if combine == :fer
        return result.lambda.fer_true
    elseif combine == :sum_lambda
        isempty(summaries) && return 0.0
        return sum(s.lambda_true for s in values(summaries))
    elseif combine == :max_lambda
        isempty(summaries) && return 0.0
        return maximum(s.lambda_true for s in values(summaries))
    elseif combine == :hybrid
        isempty(summaries) && return 0.0
        vals = Float64[]
        for s in values(summaries)
            c = get(s.lower_cvar, 0.05, NaN)
            push!(vals, s.lambda_true * max(0.0, -c))
        end
        return sum(vals)
    elseif combine == :max_hybrid
        isempty(summaries) && return 0.0
        vals = Float64[]
        for s in values(summaries)
            c = get(s.lower_cvar, 0.05, NaN)
            push!(vals, s.lambda_true * max(0.0, -c))
        end
        return maximum(vals)
    else
        throw(ArgumentError("Unknown combine mode $combine."))
    end
end

"""
Legacy lambda-only single-swap nominator.  Kept for backward compatibility.
Prefer `suggest_tail_swaps_v2` (sigma-aware single-swap) or
`suggest_tail_swaps_kicker` (k-swap kicker).
"""
function suggest_tail_swaps(
    A::AbstractVector{<:Integer},
    N::Int,
    result::SpectrumResult;
    reliability::AbstractVector{<:Real},
    mode::Symbol = :explicit,
    score::Symbol = :hybrid,
    nremove::Int = 4,
    nadd::Int = 8,
)
    length(reliability) == N || throw(ArgumentError("reliability must have length N."))

    Aset = Set(Int.(A))
    F = sort(setdiff(collect(1:N), collect(Aset)))
    risk = risk_by_bit(result; mode = mode, score = score)

    descending = !(score in (:D_cvar_05, :D_quantile_05))
    risky = [i for i in Int.(A) if haskey(risk, i)]
    sort!(risky, by = i -> risk[i], rev = descending)
    remove_list = first(risky, min(nremove, length(risky)))

    add_list = sort(F, by = j -> Float64(reliability[j]), rev = true)
    add_list = first(add_list, min(nadd, length(add_list)))

    candidates = NamedTuple[]
    for r in remove_list
        for a in add_list
            Anew = sort(vcat(setdiff(Int.(A), [r]), [a]))
            push!(candidates, (
                remove = r,
                add = a,
                remove_risk = risk[r],
                add_reliability = Float64(reliability[a]),
                A = Anew,
            ))
        end
    end

    return candidates
end

# ============================================================
# Sigma additions
# ============================================================

function support_frequency(events::AbstractVector{DangerousEvent},
                            N::Integer,
                            n_trials::Integer)
    sigma = zeros(Float64, N)
    @inbounds for ev in events
        for j in 1:N
            if ev.path_v[j] != 0
                sigma[j] += 1.0
            end
        end
    end
    return sigma ./ max(Int(n_trials), 1)
end

function support_frequency_per_event(events::AbstractVector{DangerousEvent},
                                      N::Integer)
    sigma = zeros(Float64, N)
    isempty(events) && return sigma
    @inbounds for ev in events
        for j in 1:N
            if ev.path_v[j] != 0
                sigma[j] += 1.0
            end
        end
    end
    return sigma ./ length(events)
end

function support_risk_by_bit(result::SpectrumResult;
                              mode::Symbol = :explicit,
                              per_trial::Bool = true)
    sigma = per_trial ?
        support_frequency(result.events, result.N, result.n_trials) :
        support_frequency_per_event(result.events, result.N)
    return Dict{Int, Float64}(j => sigma[j] for j in result.A)
end

function combined_risk_by_bit(result::SpectrumResult;
                                mode::Symbol = :explicit,
                                lambda_weight::Real = 0.0,
                                sigma_weight::Real = 1.0,
                                score::Symbol = :lambda_true,
                                per_trial::Bool = true)
    lam = risk_by_bit(result; mode = mode, score = score)
    sig = support_risk_by_bit(result; mode = mode, per_trial = per_trial)

    out = Dict{Int, Float64}()
    for j in result.A
        l = get(lam, j, 0.0)
        s = get(sig, j, 0.0)
        out[j] = lambda_weight * Float64(l) + sigma_weight * Float64(s)
    end
    return out
end

"""
    suggest_tail_swaps_v2(A, N, result; reliability, ...)

Sigma-aware single-swap nominator.  Each candidate carries scalar
`remove`/`add` indices.  Prefer `suggest_tail_swaps_kicker` for the
configurable k-swap version (where `swap_size = 1` is equivalent).
"""
function suggest_tail_swaps_v2(A::AbstractVector{<:Integer},
                                N::Integer,
                                result::SpectrumResult;
                                reliability::AbstractVector{<:Real},
                                mode::Symbol = :explicit,
                                risk_mix::NamedTuple = (lambda = 0.0, sigma = 1.0),
                                score::Symbol = :lambda_true,
                                add_strategy::Symbol = :top_reliability,
                                nremove::Integer = 4,
                                nadd::Integer = 8)
    length(reliability) == N || throw(ArgumentError("reliability must have length N."))
    add_strategy in (:top_reliability, :all_frozen) ||
        throw(ArgumentError("add_strategy must be :top_reliability or :all_frozen."))

    Aset = Set(Int.(A))
    F = sort(setdiff(collect(1:N), collect(Aset)))

    lam = risk_by_bit(result; mode = mode, score = score)
    sig = support_risk_by_bit(result; mode = mode, per_trial = true)

    lw = Float64(risk_mix.lambda)
    sw = Float64(risk_mix.sigma)

    risk = Dict{Int, Float64}()
    for j in collect(Aset)
        risk[j] = lw * get(lam, j, 0.0) + sw * get(sig, j, 0.0)
    end

    risky = sort(collect(Aset), by = j -> get(risk, j, 0.0), rev = true)
    remove_list = first(risky, min(Int(nremove), length(risky)))

    add_list = if add_strategy == :top_reliability
        sorted = sort(F, by = j -> Float64(reliability[j]), rev = true)
        first(sorted, min(Int(nadd), length(sorted)))
    else
        sort(F, by = j -> Float64(reliability[j]), rev = true)
    end

    cands = NamedTuple[]
    for r in remove_list
        for a in add_list
            Anew = sort(vcat(setdiff(Int.(A), [r]), [a]))
            push!(cands, (
                remove = r,
                add = a,
                remove_risk = get(risk, r, 0.0),
                remove_lambda = get(lam, r, 0.0),
                remove_sigma = get(sig, r, 0.0),
                add_reliability = Float64(reliability[a]),
                A = Anew,
            ))
        end
    end
    return cands
end

# ============================================================
# Kicker: k-swap nominator
# ============================================================

"""
    suggest_tail_swaps_kicker(A, N, result; swap_size, ...)

Generalized k-swap nominator (1-swap, 2-swap, 3-swap, ...).

For `swap_size = 1` this is the same as `suggest_tail_swaps_v2` but with
plural `removes`/`adds` fields in each candidate.  For `swap_size = 2` or
higher, this generates candidates that simultaneously remove `swap_size`
information bits and add `swap_size` frozen bits.

Useful as a "kicker" when single-swap search has converged to a local
optimum: a 2-swap or 3-swap can escape it.

Combinatorial explosion is bounded by `max_candidates`.  Candidates are
sorted by (total remove-risk descending, total add-reliability descending)
and truncated.

Each candidate is a NamedTuple with fields:
- `removes :: Vector{Int}` — sorted, length `swap_size`.
- `adds    :: Vector{Int}` — sorted, length `swap_size`.
- `remove_risk`            — sum of per-bit risks of removed bits.
- `remove_lambda`          — sum of lambda risks for removed bits.
- `remove_sigma`           — sum of sigma risks for removed bits.
- `add_reliability`        — sum of reliabilities of added bits.
- `A`                      — the new information set (sorted).
"""
function suggest_tail_swaps_kicker(A::AbstractVector{<:Integer},
                                     N::Integer,
                                     result::SpectrumResult;
                                     reliability::AbstractVector{<:Real},
                                     swap_size::Integer = 1,
                                     mode::Symbol = :explicit,
                                     risk_mix::NamedTuple = (lambda = 0.0, sigma = 1.0),
                                     score::Symbol = :lambda_true,
                                     add_strategy::Symbol = :top_reliability,
                                     nremove::Integer = 4,
                                     nadd::Integer = 8,
                                     max_candidates::Integer = 256)
    length(reliability) == N || throw(ArgumentError("reliability must have length N."))
    swap_size >= 1 || throw(ArgumentError("swap_size must be >= 1."))
    add_strategy in (:top_reliability, :all_frozen) ||
        throw(ArgumentError("add_strategy must be :top_reliability or :all_frozen."))

    Aset = Set(Int.(A))
    F = sort(setdiff(collect(1:N), collect(Aset)))

    lam = risk_by_bit(result; mode = mode, score = score)
    sig = support_risk_by_bit(result; mode = mode, per_trial = true)

    lw = Float64(risk_mix.lambda)
    sw = Float64(risk_mix.sigma)

    risk = Dict{Int, Float64}()
    for j in collect(Aset)
        risk[j] = lw * get(lam, j, 0.0) + sw * get(sig, j, 0.0)
    end

    risky = sort(collect(Aset), by = j -> get(risk, j, 0.0), rev = true)
    remove_pool = first(risky, min(Int(nremove), length(risky)))

    add_pool = if add_strategy == :top_reliability
        sorted = sort(F, by = j -> Float64(reliability[j]), rev = true)
        first(sorted, min(Int(nadd), length(sorted)))
    else
        sort(F, by = j -> Float64(reliability[j]), rev = true)
    end

    sz = Int(swap_size)
    if sz > length(remove_pool) || sz > length(add_pool)
        return NamedTuple[]
    end

    remove_combos = _combinations(remove_pool, sz)
    add_combos    = _combinations(add_pool, sz)

    cands = NamedTuple[]
    for r_combo in remove_combos
        for a_combo in add_combos
            A_new = sort(union(setdiff(Int.(A), r_combo), Int.(a_combo)))
            length(A_new) == length(A) || continue
            push!(cands, (
                removes         = sort(Int.(r_combo)),
                adds            = sort(Int.(a_combo)),
                remove_risk     = sum(get(risk, r, 0.0)  for r in r_combo),
                remove_lambda   = sum(get(lam,  r, 0.0)  for r in r_combo),
                remove_sigma    = sum(get(sig,  r, 0.0)  for r in r_combo),
                add_reliability = sum(Float64(reliability[a]) for a in a_combo),
                A               = A_new,
            ))
        end
    end

    sort!(cands, by = c -> (-Float64(c.remove_risk), -Float64(c.add_reliability)))
    return first(cands, min(Int(max_candidates), length(cands)))
end

# ============================================================
# Diagnostics
# ============================================================

function influence_matrix(
    samples::Vector{PathSample{T}},
    N::Int;
    alpha::Real = 0.10,
) where {T<:AbstractFloat}
    M = fill(NaN, N, N)
    C = zeros(Int, N, N)
    first_errors = sort(unique(s.first_error for s in samples))

    for i in first_errors
        ss = [s for s in samples if s.first_error == i]
        isempty(ss) && continue
        D = Float64[s.D for s in ss]
        q = quantile(D, Float64(alpha))
        tail = [s for s in ss if s.D <= q]
        isempty(tail) && continue
        for j in 1:N
            M[i, j] = mean(Float64(s.BM[j]) for s in tail)
            C[i, j] = length(tail)
        end
    end

    return M, C
end

function print_spectrum_report(
    result::SpectrumResult;
    mode::Symbol = :explicit,
    nshow::Int = 25,
    sort_by::Symbol = :lambda_true,
)
    summaries = result.summaries[mode]
    rows = collect(values(summaries))

    if sort_by == :lambda_true
        sort!(rows, by = s -> s.lambda_true, rev = true)
    elseif sort_by == :count
        sort!(rows, by = s -> s.count, rev = true)
    elseif sort_by == :D_mean
        sort!(rows, by = s -> s.D_mean)
    elseif sort_by == :D_cvar_05
        sort!(rows, by = s -> get(s.lower_cvar, 0.05, Inf))
    else
        throw(ArgumentError("Unknown sort_by $sort_by."))
    end

    println("Dangerous-path PM spectrum")
    println("  mode                  = ", mode)
    println("  N,K,L                 = ", (result.N, result.K, result.L))
    println("  trials                = ", result.n_trials)
    println("  dismissed             = ", result.n_dismissed)
    println("  kept                  = ", result.n_kept)
    println("  trials with events    = ", result.n_trials_with_events)
    println("  Lambda true           = ", result.lambda.lambda_total_true)
    println("  FER true estimate     = ", result.lambda.fer_true)
    println("  FER bounds from Lambda= ", result.lambda.fer_bounds_true)
    println()

    @printf("%6s %8s %12s %12s %12s %12s %10s %10s\n",
            "i", "count", "lambda", "Dmean", "Dstd", "q05", "P(D<=0)", "eta*")
    @printf("%6s %8s %12s %12s %12s %12s %10s %10s\n",
            "------", "--------", "------------", "------------", "------------", "------------", "----------", "----------")

    for s in first(rows, min(nshow, length(rows)))
        q05 = get(s.quantile, 0.05, NaN)
        @printf("%6d %8d %12.4e %12.4e %12.4e %12.4e %10.3f %10.3f\n",
                s.first_error, s.count, s.lambda_true, s.D_mean, s.D_std,
                q05, s.P_D_le_0, s.dq.eta_chernoff)
    end

    return nothing
end

end # module DangerousPathSpectrumClean
