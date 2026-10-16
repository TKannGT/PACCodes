module TailAwarePM

# ============================================================
# TailAwarePM.jl
#
# Deterministic conditional Gaussian density evolution for PAC / polar
# path-metric design.
#
# Main changes relative to the MC baseline:
#   1. Check-node output moments are computed by 2D Gauss-Hermite
#      quadrature instead of Monte Carlo.
#   2. At final check-node outputs, BM moments are computed directly
#      through the check-node nonlinearity, not from a Gaussianized
#      posterior unless you explicitly choose to do that elsewhere.
#   3. Each path is scored as a random PM, not only by E[PM].  The
#      default score E[PM] - lambda*Std[PM] is a lower-tail score;
#      Chernoff and normal lower-tail scores are also provided.
#
# Required packages:
#     import Pkg
#     Pkg.add(["FastGaussQuadrature", "SpecialFunctions"])
# ============================================================

using FastGaussQuadrature
using SpecialFunctions
using LinearAlgebra
using Printf

export GHRule,
       boxplus, boxplus_minsum,
       bm_moments_gaussian, expected_bm_sign, expected_bm_bit,
       checknode_moments_gh, bm_moments_boxplus_gh,
       conditional_dega_path, pm_components, path_score,
       set_min_score, simple_search_tail,
       normal_lowtail_score, cantelli_lowtail_score,
       chernoff_lowtail_score, polarize

const SQRT2 = sqrt(2.0)
const INV_SQRT_2PI = 1.0 / sqrt(2.0 * pi)

@inline normal_pdf(x::Real) = INV_SQRT_2PI * exp(-0.5 * Float64(x)^2)
@inline normal_cdf(x::Real) = 0.5 * erfc(-Float64(x) / SQRT2)
@inline normal_quantile(p::Real) = SQRT2 * erfinv(2.0 * Float64(p) - 1.0)
@inline bit_sign(u::Integer) = u == 0 ? 1 : -1
@inline mean_sign(mu::Real) = mu >= 0 ? 1 : -1
@inline branch_metric(llr::Real, s::Integer) = max(-Float64(s) * Float64(llr), 0.0)

@inline function logaddexp(a::Float64, b::Float64)
    if a == -Inf
        return b
    elseif b == -Inf
        return a
    end
    m = max(a, b)
    return m + log(exp(a - m) + exp(b - m))
end

@inline function safe_logcdf(x::Real)
    p = normal_cdf(x)
    return p == 0.0 ? -Inf : log(p)
end

# ============================================================
# Polarization helper, copied in spirit from the baseline tools.
# ============================================================

function polarize(B::Matrix{Int})
    Gbase = [1 0; 1 1]
    G = copy(Gbase)
    n = size(B, 1) - 1
    u = B[n + 1, :]

    for i in 1:n
        k = 2^(n - i)
        comb = kron(Matrix{Int}(I, k, k), G)
        B[n + 1 - i, :] = vec(mod.(transpose(u) * comb, 2))
        G = kron(G, Gbase)
    end
    return B
end

# ============================================================
# Check-node operators.
# ============================================================

"""
    boxplus(x, y)

Stable exact check-node operation

    x boxplus y = 2atanh(tanh(x/2) tanh(y/2)).

This is preferable to evaluating tanh/atanh directly.
"""
@inline function boxplus(x::Real, y::Real)
    x = Float64(x)
    y = Float64(y)
    ax = abs(x)
    ay = abs(y)
    return sign(x) * sign(y) * min(ax, ay) +
           log1p(exp(-abs(x + y))) -
           log1p(exp(-abs(x - y)))
end

"""
    boxplus_minsum(x, y)

Min-sum approximation used by some fast decoders/baselines.
Pass `op = boxplus_minsum` to the quadrature routines to reproduce
that approximation deterministically.
"""
@inline boxplus_minsum(x::Real, y::Real) = sign(x) * sign(y) * min(abs(x), abs(y))

# ============================================================
# Gauss-Hermite rule.
# ============================================================

struct GHRule
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

function GHRule(Q::Integer = 24)
    Q >= 2 || throw(ArgumentError("Q must be at least 2."))
    nodes, weights = gausshermite(Q)  # integral exp(-x^2) f(x) dx
    return GHRule(Float64.(nodes), Float64.(weights))
end
# ============================================================
# Gaussian BM formulas.
# ============================================================

"""
    bm_moments_gaussian(mu, sigma; s = +1)

For L ~ N(mu, sigma^2), compute the mean and variance of

    BM = |L| * 1{sL < 0} = max(-sL, 0),

where s = +1 is the branch uhat = 0, and s = -1 is the branch uhat = 1.

Returns `(mean, var)`.
"""
function bm_moments_gaussian(mu::Real, sigma::Real; s::Integer = 1)
    (s == 1 || s == -1) || throw(ArgumentError("s must be +1 or -1."))
    mu = Float64(mu)
    sigma = Float64(sigma)

    if sigma <= 0.0
        b = branch_metric(mu, s)
        return b, 0.0
    end

    # T = sL ~ N(m, sigma^2), BM = -T * 1{T < 0}.
    m = s * mu
    a = m / sigma
    p = normal_cdf(-a)
    q = normal_pdf(a)

    ebm = sigma * q - m * p
    ebm2 = (sigma^2 + m^2) * p - sigma * m * q
    return max(ebm, 0.0), max(ebm2 - ebm^2, 0.0)
end

expected_bm_sign(mu::Real, sigma::Real, s::Integer) = bm_moments_gaussian(mu, sigma; s = s)[1]
expected_bm_bit(mu::Real, sigma::Real, uhat::Integer) = expected_bm_sign(mu, sigma, bit_sign(uhat))

"""
    bm_logmgf_gaussian(mu, sigma, theta; s = +1)

Log moment generating function log E[exp(theta * BM)] for the same
Gaussian BM variable as `bm_moments_gaussian`.

For lower-tail Chernoff bounds on PM, this is used with theta = -eta <= 0.
"""
function bm_logmgf_gaussian(mu::Real, sigma::Real, theta::Real; s::Integer = 1)
    (s == 1 || s == -1) || throw(ArgumentError("s must be +1 or -1."))
    mu = Float64(mu)
    sigma = Float64(sigma)
    theta = Float64(theta)

    if sigma <= 0.0
        return theta * branch_metric(mu, s)
    end

    # T = sL ~ N(m, sigma^2), BM = -T 1{T < 0}.
    # E[e^{theta BM}] = P(T >= 0) + E[e^{-theta T} 1{T < 0}]
    #                  = Phi(a) + exp(-theta*m + theta^2*sigma^2/2)
    #                    Phi(theta*sigma - a), a = m/sigma.
    m = s * mu
    a = m / sigma
    log_atom = safe_logcdf(a)
    log_cont = -theta * m + 0.5 * theta^2 * sigma^2 + safe_logcdf(theta * sigma - a)
    return logaddexp(log_atom, log_cont)
end

# ============================================================
# 2D Gauss-Hermite check-node moment routines.
# ============================================================

"""
    checknode_moments_gh(mu_x, var_x, mu_y, var_y; rule = GHRule(24), op = boxplus)

Deterministically approximates E[Z] and Var[Z] for

    Z = op(X, Y),
    X ~ N(mu_x, var_x), Y ~ N(mu_y, var_y), independent.

The default `op = boxplus` is the exact check-node update.  Use
`op = boxplus_minsum` if you want the min-sum approximation.
"""
function checknode_moments_gh(mu_x::Real, var_x::Real,
                              mu_y::Real, var_y::Real;
                              rule::GHRule = GHRule(24),
                              op::Function = boxplus)
    var_x >= 0 || throw(ArgumentError("var_x must be nonnegative."))
    var_y >= 0 || throw(ArgumentError("var_y must be nonnegative."))

    sx = sqrt(Float64(var_x))
    sy = sqrt(Float64(var_y))
    mx = Float64(mu_x)
    my = Float64(mu_y)

    mean_z = 0.0
    second_z = 0.0
    scale_x = SQRT2 * sx
    scale_y = SQRT2 * sy

    @inbounds for i in eachindex(rule.nodes)
        x = mx + scale_x * rule.nodes[i]
        wi = rule.weights[i]
        for j in eachindex(rule.nodes)
            y = my + scale_y * rule.nodes[j]
            w = wi * rule.weights[j] / pi
            z = op(x, y)
            mean_z += w * z
            second_z += w * z * z
        end
    end

    return mean_z, max(second_z - mean_z^2, 0.0)
end

"""
    bm_moments_boxplus_gh(mu_x, var_x, mu_y, var_y; s = +1, rule = GHRule(24), op = boxplus)

Computes BM moments directly for a final check-node output

    Z = op(X, Y), BM = max(-sZ, 0),

without Gaussianizing Z first.  Returns `(mean, var)`.
"""
function bm_moments_boxplus_gh(mu_x::Real, var_x::Real,
                               mu_y::Real, var_y::Real;
                               s::Integer = 1,
                               rule::GHRule = GHRule(24),
                               op::Function = boxplus)
    (s == 1 || s == -1) || throw(ArgumentError("s must be +1 or -1."))
    var_x >= 0 || throw(ArgumentError("var_x must be nonnegative."))
    var_y >= 0 || throw(ArgumentError("var_y must be nonnegative."))

    sx = sqrt(Float64(var_x))
    sy = sqrt(Float64(var_y))
    mx = Float64(mu_x)
    my = Float64(mu_y)

    mean_b = 0.0
    second_b = 0.0
    scale_x = SQRT2 * sx
    scale_y = SQRT2 * sy

    @inbounds for i in eachindex(rule.nodes)
        x = mx + scale_x * rule.nodes[i]
        wi = rule.weights[i]
        for j in eachindex(rule.nodes)
            y = my + scale_y * rule.nodes[j]
            w = wi * rule.weights[j] / pi
            z = op(x, y)
            b = branch_metric(z, s)
            mean_b += w * b
            second_b += w * b * b
        end
    end

    return mean_b, max(second_b - mean_b^2, 0.0)
end

"""
    bm_loglaplace_boxplus_gh(mu_x, var_x, mu_y, var_y, eta; s = +1, rule = GHRule(24), op = boxplus)

Computes log E[exp(-eta * BM)] for a final check-node BM, directly
through quadrature.  This is the log-Laplace transform used in the
lower-tail Chernoff bound.
"""
function bm_loglaplace_boxplus_gh(mu_x::Real, var_x::Real,
                                  mu_y::Real, var_y::Real,
                                  eta::Real;
                                  s::Integer = 1,
                                  rule::GHRule = GHRule(24),
                                  op::Function = boxplus)
    (s == 1 || s == -1) || throw(ArgumentError("s must be +1 or -1."))
    eta >= 0 || throw(ArgumentError("eta must be nonnegative."))
    var_x >= 0 || throw(ArgumentError("var_x must be nonnegative."))
    var_y >= 0 || throw(ArgumentError("var_y must be nonnegative."))

    sx = sqrt(Float64(var_x))
    sy = sqrt(Float64(var_y))
    mx = Float64(mu_x)
    my = Float64(mu_y)
    eta = Float64(eta)

    scale_x = SQRT2 * sx
    scale_y = SQRT2 * sy
    lse = -Inf

    @inbounds for i in eachindex(rule.nodes)
        x = mx + scale_x * rule.nodes[i]
        log_wi = log(rule.weights[i])
        for j in eachindex(rule.nodes)
            y = my + scale_y * rule.nodes[j]
            z = op(x, y)
            b = branch_metric(z, s)
            term = log_wi + log(rule.weights[j]) - log(pi) - eta * b
            lse = logaddexp(lse, term)
        end
    end

    return lse
end

# ============================================================
# Conditional GH-DEGA path computation.
# ============================================================

"""
    conditional_dega_path(u, m; rule = GHRule(24), op = boxplus, eta_grid = Float64[])

Run conditional Gaussian density evolution for a hypothesized path `u`.

Initial channel LLRs are modeled as N(m, 2m).  The output contains:

- `mu`, `sigma`: final posterior LLR mean/std for each bit.
- `bm_mean0`, `bm_var0`: BM moments if this final bit is decoded as 0.
- `bm_mean1`, `bm_var1`: BM moments if this final bit is decoded as 1.
- `loglap0`, `loglap1`: matrices whose column k is
      log E[exp(-eta_grid[k] * BM)]
  for branch 0 or branch 1.  These are used for Chernoff lower-tail scores.
- `B`: the polarized decision table used by the conditional update.

At final check-node outputs, BM moments/log-Laplace values are computed
through the check-node nonlinearity directly.  At final repetition-node
outputs, they use the exact Gaussian truncated-moment formulas.
"""
function conditional_dega_path(u::AbstractVector{<:Integer}, m::Real;
                               rule::GHRule = GHRule(24),
                               op::Function = boxplus,
                               eta_grid::AbstractVector{<:Real} = Float64[])
    N = length(u)
    isinteger(log2(N)) || error("N must be a power of 2; got N = $N")
    n = Int(log2(N))

    B = zeros(Int, n + 1, N)
    B[n + 1, :] = Int.(u)
    B = polarize(B)

    M = zeros(Float64, n + 1, N)
    V = zeros(Float64, n + 1, N)
    M[1, :] .= Float64(m)
    V[1, :] .= max(2.0 * Float64(m), 0.0)

    bm_mean0 = zeros(Float64, N)
    bm_var0  = zeros(Float64, N)
    bm_mean1 = zeros(Float64, N)
    bm_var1  = zeros(Float64, N)

    etas = Float64.(eta_grid)
    G = length(etas)
    loglap0 = zeros(Float64, N, G)
    loglap1 = zeros(Float64, N, G)

    for k in 2:(n + 1)
        for i in 1:2:2^(k - 1)
            for j in 1:div(N, 2^(k - 1))
                a = Int((i - 1) * N / 2^(k - 1) + j)
                b = Int( i      * N / 2^(k - 1) + j)

                # Check-node output at position a.
                M[k, a], V[k, a] = checknode_moments_gh(
                    M[k - 1, a], V[k - 1, a],
                    M[k - 1, b], V[k - 1, b];
                    rule = rule, op = op)

                # Repetition-node output at position b.
                # The sign is determined by the corresponding already-decoded bit.
                M[k, b] = (-1)^B[k, a] * M[k - 1, a] + M[k - 1, b]
                V[k, b] = V[k - 1, a] + V[k - 1, b]

                if k == n + 1
                    # Direct BM quadrature for final check-node bit a.
                    bm_mean0[a], bm_var0[a] = bm_moments_boxplus_gh(
                        M[k - 1, a], V[k - 1, a],
                        M[k - 1, b], V[k - 1, b];
                        s = 1, rule = rule, op = op)
                    bm_mean1[a], bm_var1[a] = bm_moments_boxplus_gh(
                        M[k - 1, a], V[k - 1, a],
                        M[k - 1, b], V[k - 1, b];
                        s = -1, rule = rule, op = op)

                    # Gaussian BM formulas for final repetition-node bit b.
                    sig_b = sqrt(max(V[k, b], 0.0))
                    bm_mean0[b], bm_var0[b] = bm_moments_gaussian(M[k, b], sig_b; s = 1)
                    bm_mean1[b], bm_var1[b] = bm_moments_gaussian(M[k, b], sig_b; s = -1)

                    # Optional log-Laplace transforms for Chernoff scoring.
                    @inbounds for g in eachindex(etas)
                        eta = etas[g]
                        loglap0[a, g] = bm_loglaplace_boxplus_gh(
                            M[k - 1, a], V[k - 1, a],
                            M[k - 1, b], V[k - 1, b], eta;
                            s = 1, rule = rule, op = op)
                        loglap1[a, g] = bm_loglaplace_boxplus_gh(
                            M[k - 1, a], V[k - 1, a],
                            M[k - 1, b], V[k - 1, b], eta;
                            s = -1, rule = rule, op = op)

                        loglap0[b, g] = bm_logmgf_gaussian(M[k, b], sig_b, -eta; s = 1)
                        loglap1[b, g] = bm_logmgf_gaussian(M[k, b], sig_b, -eta; s = -1)
                    end
                end
            end
        end
    end

    return (bm_mean0 = bm_mean0,
            bm_var0  = bm_var0,
            bm_mean1 = bm_mean1,
            bm_var1  = bm_var1,
            mu       = M[n + 1, :],
            sigma    = sqrt.(max.(V[n + 1, :], 0.0)),
            eta_grid = etas,
            loglap0  = loglap0,
            loglap1  = loglap1,
            B        = B)
end

@inline function get_bm_moments(dega::NamedTuple, i::Int, uhat::Integer)
    if uhat == 0
        return dega.bm_mean0[i], dega.bm_var0[i]
    elseif uhat == 1
        return dega.bm_mean1[i], dega.bm_var1[i]
    else
        throw(ArgumentError("uhat must be 0 or 1."))
    end
end

@inline function get_loglap(dega::NamedTuple, i::Int, uhat::Integer, g::Int)
    if uhat == 0
        return dega.loglap0[i, g]
    elseif uhat == 1
        return dega.loglap1[i, g]
    else
        throw(ArgumentError("uhat must be 0 or 1."))
    end
end

# ============================================================
# Path aggregation and tail scores.
# ============================================================

"""
    pm_components(dega, u, A; ...)

Aggregate BM moments for a hypothesized path into PM moments.

Returns `(E_PM, V_PM, loglap, forced)` where `loglap[g]` is

    sum_i log E[exp(-eta_grid[g] * BM_i)]

for the included BM terms.  This supports lower-tail Chernoff scoring.

Conventions:
- `A` is the information set, 1-based.
- `skip_mean_greedy_info = true` implements the paper-style convention that
  future information bits may follow the locally zero-BM extension, so they
  are skipped when the path bit agrees with sign(mu).
- `pac_forced = true` adds the baseline approximation that the post-error
  frozen bit with largest conditional mean is forced wrong by PAC.
"""
function pm_components(dega::NamedTuple,
                       u::AbstractVector{<:Integer},
                       A::AbstractVector{<:Integer};
                       subtract_pre_error::Bool = true,
                       skip_mean_greedy_info::Bool = true,
                       pac_forced::Bool = false)
    N = length(u)
    Aset = Set(Int.(A))

    err_pos = findfirst(==(1), u)
    err_pos = err_pos === nothing ? 0 : err_pos

    forced = -1
    if pac_forced && err_pos > 0
        best_mu = -Inf
        @inbounds for j in (err_pos + 1):N
            if !(j in Aset) && dega.mu[j] > best_mu
                best_mu = dega.mu[j]
                forced = j
            end
        end
    end

    E_PM = 0.0
    V_PM = 0.0
    G = length(dega.eta_grid)
    loglap = zeros(Float64, G)

    @inbounds for i in 1:N
        if subtract_pre_error && i < err_pos
            continue
        end

        uhat = Int(u[i])

        # Future information bits can choose the zero-BM extension.
        if skip_mean_greedy_info && (i in Aset) && bit_sign(uhat) == mean_sign(dega.mu[i])
            continue
        end

        # PAC-forced frozen bit is approximated as forced to uhat = 1.
        # This matches the baseline's `neg = false` treatment.
        if i == forced
            uhat = 1
        end

        e, v = get_bm_moments(dega, i, uhat)
        E_PM += e
        V_PM += v

        for g in 1:G
            loglap[g] += get_loglap(dega, i, uhat, g)
        end
    end

    return (E_PM = E_PM, V_PM = V_PM, loglap = loglap, forced = forced)
end

"""
    normal_lowtail_score(E, V; tau = 0)

Returns `-log P(PM <= tau)` under the Gaussian approximation
PM ~ N(E, V).  Larger is safer.
"""
function normal_lowtail_score(E::Real, V::Real; tau::Real = 0.0)
    E = Float64(E)
    V = Float64(V)
    tau = Float64(tau)

    if V <= 0.0
        return E <= tau ? 0.0 : Inf
    end

    z = (tau - E) / sqrt(V)
    p = normal_cdf(z)
    return p == 0.0 ? Inf : -log(p)
end

"""
    cantelli_lowtail_score(E, V; tau = 0)

Distribution-free one-sided Chebyshev/Cantelli lower-tail score.
It returns `-log(bound)` for

    P(PM <= tau) <= V / (V + (E - tau)^2),  tau < E.

Larger is safer.  This uses only mean and variance, so it is conservative.
"""
function cantelli_lowtail_score(E::Real, V::Real; tau::Real = 0.0)
    E = Float64(E)
    V = max(Float64(V), 0.0)
    tau = Float64(tau)

    if E <= tau
        return 0.0
    elseif V <= 0.0
        return Inf
    end

    bound = V / (V + (E - tau)^2)
    return -log(bound)
end

"""
    chernoff_lowtail_score(loglap, eta_grid; tau = 0)

Given `loglap[g] = log E[exp(-eta_grid[g] * PM)]`, compute the
Chernoff lower-tail exponent

    sup_eta { -eta*tau - log E[exp(-eta PM)] }.

The bound is

    P(PM <= tau) <= exp(-score).

Larger is safer.  Requires an eta grid in `conditional_dega_path`.
"""
function chernoff_lowtail_score(loglap::AbstractVector{<:Real},
                                eta_grid::AbstractVector{<:Real};
                                tau::Real = 0.0)
    length(loglap) == length(eta_grid) || throw(ArgumentError("loglap and eta_grid lengths differ."))
    isempty(eta_grid) && throw(ArgumentError("eta_grid is empty; provide eta_grid to conditional_dega_path."))

    best = 0.0
    @inbounds for g in eachindex(eta_grid)
        eta = Float64(eta_grid[g])
        eta < 0 && throw(ArgumentError("eta_grid entries must be nonnegative."))
        exponent = -eta * Float64(tau) - Float64(loglap[g])
        if exponent > best
            best = exponent
        end
    end
    return best
end

"""
    path_score(dega, u, A; mode = :lcb, lambda = 2.0, tau = 0.0, ...)

Score a single hypothesized error path.  Larger means safer.

Supported modes:
- `:mean`:       E[PM] - tau.
- `:lcb`:        E[PM] - tau - lambda*Std[PM].  This is a lower-quantile proxy.
- `:normaltail`: -log NormalApprox(PM <= tau).
- `:cantelli`:   -log CantelliBound(PM <= tau).
- `:chernoff`:   Chernoff lower-tail exponent using `eta_grid`.

`tau` is the dangerous PM threshold.  Using tau = 0 reproduces a pure
"wrong path PM should not be too small" score.  A more direct SCL proxy is
to set tau to a correct-path PM budget, for example E[PM_correct] or an
upper quantile of PM_correct.  In `simple_search_tail`, set
`correct_budget = true` to use tau = E[PM_correct] + correct_lambda*Std[PM_correct]
for each candidate information set.
"""
function path_score(dega::NamedTuple,
                    u::AbstractVector{<:Integer},
                    A::AbstractVector{<:Integer};
                    mode::Symbol = :lcb,
                    lambda::Real = 2.0,
                    tau::Real = 0.0,
                    subtract_pre_error::Bool = true,
                    skip_mean_greedy_info::Bool = true,
                    pac_forced::Bool = false)
    comp = pm_components(dega, u, A;
                         subtract_pre_error = subtract_pre_error,
                         skip_mean_greedy_info = skip_mean_greedy_info,
                         pac_forced = pac_forced)

    if mode == :mean
        return comp.E_PM - Float64(tau)
    elseif mode == :lcb
        return comp.E_PM - Float64(tau) - Float64(lambda) * sqrt(max(comp.V_PM, 0.0))
    elseif mode == :normaltail
        return normal_lowtail_score(comp.E_PM, comp.V_PM; tau = tau)
    elseif mode == :cantelli
        return cantelli_lowtail_score(comp.E_PM, comp.V_PM; tau = tau)
    elseif mode == :chernoff
        return chernoff_lowtail_score(comp.loglap, dega.eta_grid; tau = tau)
    else
        throw(ArgumentError("Unknown mode: $mode. Use :mean, :lcb, :normaltail, :cantelli, or :chernoff."))
    end
end

"""
    set_min_score(A, dega_cache, N; ...)

Compute the worst-path score over first-error positions i in A.
Returns `(min_score, worst_bit)`.
"""
function set_min_score(A::AbstractVector{<:Integer},
                       dega_cache::Dict{Int, <:NamedTuple},
                       N::Int;
                       mode::Symbol = :lcb,
                       lambda::Real = 2.0,
                       tau::Real = 0.0,
                       subtract_pre_error::Bool = true,
                       skip_mean_greedy_info::Bool = true,
                       pac_forced::Bool = false,
                       correct_budget::Bool = false,
                       correct_lambda::Real = lambda)
    tau_eff = Float64(tau)
    if correct_budget
        haskey(dega_cache, 0) || throw(ArgumentError("correct_budget=true requires dega_cache[0]."))
        u0 = zeros(Int, N)
        corr = pm_components(dega_cache[0], u0, A;
                             subtract_pre_error = false,
                             skip_mean_greedy_info = skip_mean_greedy_info,
                             pac_forced = false)
        tau_eff = corr.E_PM + Float64(correct_lambda) * sqrt(max(corr.V_PM, 0.0))
    end

    min_score = Inf
    worst_bit = -1

    for i in A
        haskey(dega_cache, i) || continue
        u_i = zeros(Int, N)
        u_i[i] = 1
        s = path_score(dega_cache[i], u_i, A;
                       mode = mode,
                       lambda = lambda,
                       tau = tau_eff,
                       subtract_pre_error = subtract_pre_error,
                       skip_mean_greedy_info = skip_mean_greedy_info,
                       pac_forced = pac_forced)
        if s < min_score
            min_score = s
            worst_bit = i
        end
    end

    return min_score, worst_bit
end

# Helper because Base.replace does not express "replace this element" clearly.
function swap_in_set(A::AbstractVector{<:Integer}, old::Integer, new::Integer)
    out = collect(Int.(A))
    idx = findfirst(==(old), out)
    idx === nothing && throw(ArgumentError("old element $old is not in A."))
    out[idx] = Int(new)
    return sort(out)
end

"""
    simple_search_tail(N, K, SNR_dB; ...)

Greedy swap search for a tail-aware information set.

Objective:

    maximize_A  min_{i in A} score_i(A),

where score_i can be mean-only, lower-confidence-bound, normal-tail,
Cantelli-tail, or Chernoff-tail depending on `mode`.  If
`correct_budget = true`, the score is measured against an upper budget
for the correct-path PM rather than against zero.

Good starting choices:
- `mode = :lcb, lambda = 2.0` for a fast, stable tail-aware search.
- `mode = :lcb, lambda = 3.0` for a more conservative lower tail.
- `mode = :chernoff, eta_grid = range(0.0, 5.0, length = 51)` when you want
  a more explicitly lower-tail objective.

Returns `(A, dega_cache)`.
"""
function simple_search_tail(N::Int, K::Int, SNR_dB::Real;
                            Q::Integer = 24,
                            op::Function = boxplus,
                            mode::Symbol = :lcb,
                            lambda::Real = 2.0,
                            tau::Real = 0.0,
                            eta_grid::AbstractVector{<:Real} = Float64[],
                            pac_forced::Bool = true,
                            correct_budget::Bool = false,
                            correct_lambda::Real = lambda,
                            max_iters::Int = 100,
                            subtract_pre_error::Bool = true,
                            skip_mean_greedy_info::Bool = true,
                            verbose::Bool = true)
    isinteger(log2(N)) || error("N must be a power of 2; got N = $N")
    0 < K <= N || throw(ArgumentError("K must satisfy 0 < K <= N."))

    rule = GHRule(Q)

    # Same convention as the baseline tools.
    SNR_dec = 10.0^(Float64(SNR_dB) / 10.0) * 2.0 * (K / N)
    m = 2.0 * SNR_dec

    if mode == :chernoff && isempty(eta_grid)
        eta_grid = collect(range(0.0, 5.0, length = 51))
    else
        eta_grid = Float64.(eta_grid)
    end

    verbose && @printf("Precomputing deterministic GH-DEGA for %d positions (Q = %d)...\n", N, Q)

    dega_cache = Dict{Int, NamedTuple}()
    dega_cache[0] = conditional_dega_path(zeros(Int, N), m;
                                          rule = rule, op = op, eta_grid = eta_grid)
    for i in 1:N
        u = zeros(Int, N)
        u[i] = 1
        dega_cache[i] = conditional_dega_path(u, m;
                                              rule = rule, op = op, eta_grid = eta_grid)
        verbose && (i % 16 == 0) && @printf("  done: %d / %d\n", i, N)
    end

    # Initial A: top-K by no-error posterior mean.
    A = sort(sortperm(dega_cache[0].mu, rev = true)[1:K])
    Frozen = sort(setdiff(1:N, A))
    verbose && println("Initial A: ", A)

    min_score, worst = set_min_score(A, dega_cache, N;
                                     mode = mode,
                                     lambda = lambda,
                                     tau = tau,
                                     subtract_pre_error = subtract_pre_error,
                                     skip_mean_greedy_info = skip_mean_greedy_info,
                                     pac_forced = pac_forced,
                                     correct_budget = correct_budget,
                                     correct_lambda = correct_lambda)
    verbose && @printf("Initial min score = %.6f  (worst bit = %d)\n", min_score, worst)

    visited = Set{Vector{Int}}()
    push!(visited, copy(A))

    for iter in 1:max_iters
        best_j = -1
        best_score = min_score

        for j in Frozen
            A_try = swap_in_set(A, worst, j)
            A_try in visited && continue

            s_try, _ = set_min_score(A_try, dega_cache, N;
                                     mode = mode,
                                     lambda = lambda,
                                     tau = tau,
                                     subtract_pre_error = subtract_pre_error,
                                     skip_mean_greedy_info = skip_mean_greedy_info,
                                     pac_forced = pac_forced,
                                     correct_budget = correct_budget,
                                     correct_lambda = correct_lambda)
            if s_try > best_score
                best_score = s_try
                best_j = j
            end
        end

        if best_j == -1
            verbose && @printf("Iter %d: no improving swap; stopping.\n", iter)
            break
        end

        A = swap_in_set(A, worst, best_j)
        Frozen = sort(setdiff(1:N, A))
        push!(visited, copy(A))

        min_score, worst = set_min_score(A, dega_cache, N;
                                         mode = mode,
                                         lambda = lambda,
                                         tau = tau,
                                         subtract_pre_error = subtract_pre_error,
                                         skip_mean_greedy_info = skip_mean_greedy_info,
                                         pac_forced = pac_forced,
                                     correct_budget = correct_budget,
                                     correct_lambda = correct_lambda)
        verbose && @printf("Iter %d: in = %3d, new min = %.6f (worst now = %d)\n",
                           iter, best_j, min_score, worst)
    end

    return A, dega_cache
end

end # module TailAwarePM
