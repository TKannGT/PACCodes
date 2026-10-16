module ClassicalPolarCodeFunc

include("../Misc.jl")
using LinearAlgebra
using Kronecker
using .Misc
using Random
using StatsBase

# ───────────────────────────────────────────────────────────────────────────────
# Small types & constants
# ───────────────────────────────────────────────────────────────────────────────

const UIntB = UInt8

# ───────────────────────────────────────────────────────────────────────────────
# Config & State
# ───────────────────────────────────────────────────────────────────────────────

struct Config{T}
    n::Int
    N::Int
    d::Int
    L::Int
    F::Vector{Bool}
    depths::Vector{Int}
    Tmat::Matrix{Int}
    need_final::Bool
    _::Type{T}  # Phantom type
end

# Array layout: Julia is column-major.  Storing (feature, path) means a
# single path occupies a contiguous column, so column-wise operations
# (copy, zero, swap) on one path are a single memcpy.  The hot inner
# decode loop iterates over paths p with a fixed bit-index, which is then
# a stride-1 access along the column dimension — the cache-friendly direction.
#
#   B    : (N,    2L)  — B[bit, path]
#   LLRs : (2N-1, 2L)  — LLRs[node, path]
#   vx   : (N,    2L)  — vx[bit, path]

mutable struct State{T}
    l::Int
    d::Int
    βIdx::Int
    PMs::Vector{T}
    B::Array{UIntB,2}    # (N, 2L)
    LLRs::Array{T,2}     # (2N-1, 2L)
    vx::Matrix{UIntB}    # (N, 2L)
    syndrome::Vector{UIntB}
    llr::Array{T}
    perm_buffer::Vector{Int}
    temp_B::Vector{T}
    temp_LLRs::Vector{T}
    temp_BMs::Vector{T}
    swap_buf_B::Vector{UIntB}  # scratch: swap one path-column of B or vx
    swap_buf_T::Vector{T}      # scratch: swap one path-column of LLRs
    override::Bool
end

# ───────────────────────────────────────────────────────────────────────────────
# Builders
# ───────────────────────────────────────────────────────────────────────────────

@inline function decimal_to_bin(number, width)
    out = Vector{Bool}(undef, width)
    for i in 1:width
        out[i] = ((number >> (width - i)) & 1) == 1
    end
    return out
end

function init_config(; n::Int, L::Int,
                      F::AbstractVector{Bool},
                      depths::AbstractVector{<:Integer},
                      Tmat::AbstractMatrix{<:Integer},
                      T::Type{<:AbstractFloat}=Float32)
    N = 1 << n
    d = n+1
    return Config{T}(
        n, N, d, L,
        collect(Int.(F)),
        collect(Int.(depths)),
        Matrix{Int}(Tmat),
        (N ∈ F),
        T
    )
end

function init_state(cfg::Config{T}; syndrome::Union{Nothing,AbstractVector}=nothing) where {T}
    d, N, L = cfg.d, cfg.N, cfg.L
    paths = 2L

    PMs  = zeros(T, paths)
    B    = fill(UIntB(0), N, paths)       # (N, 2L)
    vx   = fill(UIntB(0), N, paths)       # (N, 2L)
    LLRs = zeros(T, 2*N - 1, paths)      # (2N-1, 2L)
    llr  = zeros(T, N)
    sx   = isnothing(syndrome) ? zeros(UIntB, N) : UIntB.(syndrome)

    perm_buffer = Vector{Int}(undef, paths)
    temp_B      = Vector{T}(undef, N)
    temp_LLRs   = Vector{T}(undef, N)
    temp_BMs    = Vector{T}(undef, N)
    swap_buf_B  = Vector{UIntB}(undef, N)
    swap_buf_T  = Vector{T}(undef, 2*N - 1)
    override    = false

    return State{T}(1, d, 1, PMs, B, LLRs, vx, sx, llr,
                     perm_buffer, temp_B, temp_LLRs, temp_BMs,
                     swap_buf_B, swap_buf_T, override)
end

function set_syndrome!(cfg::Config, st::State, syndrome::AbstractVector)
    copyto!(st.syndrome, UIntB.(syndrome))
    return st
end

function set_force_decode!(cfg::Config, st::State, syndrome::AbstractVector)
    copyto!(st.syndrome, UIntB.(syndrome))
    st.override = true
    return st
end

function set_llr!(cfg::Config, st::State{T}, llr::Array{T}) where{T}
    # LLRs is (2N-1, 2L); broadcast channel LLRs into all path columns.
    st.LLRs[1:cfg.N, :] .= llr
    return st
end

function reset!(cfg::Config{T}, st::State{T}) where {T}
    st.l    = 1
    st.d    = cfg.d
    st.βIdx = 1
    fill!(st.PMs,       zero(T))
    fill!(st.B,         0x00)
    fill!(st.vx,        0x00)
    fill!(st.LLRs,      zero(T))
    fill!(st.temp_B,    zero(T))
    fill!(st.temp_LLRs, zero(T))
    fill!(st.temp_BMs,  zero(T))
    st.override = false
    return st
end

# ───────────────────────────────────────────────────────────────────────────────
# Probability combiners
# ───────────────────────────────────────────────────────────────────────────────

@inline function f(l1::T, l2::T) where {T}
    T(min(abs(l1),abs(l2))*sign(l1)*sign(l2))
end

@inline function g(l1::T, l2::T, u::UIntB) where {T}
    T(l2 + (1-2*u)*l1)
end

# ───────────────────────────────────────────────────────────────────────────────
# Polar projection
# ───────────────────────────────────────────────────────────────────────────────

function num_polar_steps(node::Int)
    if ispow2(node+1)
        return Int(log2(node+1))
    else
        t = 1
        parent1 = (node - 1) >>> 1
        while (parent1 & 0x1 == 1)
            t += 1
            parent1 = (parent1 - 1) >>> 1
        end
        return t
    end
end

@inline function polarize!(N::Int, st::State{T}, node::Int) where{T}
    n = num_polar_steps(node)
    st.βIdx -= 2^n - 1
    st.d = n+2
    for level in 1:n
        step = 1 << level
        for i in (node-step+1):step:node
            for j in 0:(step÷2-1)
                # Inner loop over paths p: B[bit, p] — varying p is contiguous.
                for p in 1:st.l
                    a = st.B[i+j,        p]
                    b = st.B[i+j+step÷2, p]
                    st.B[i+j,        p] = (a ⊻ b)
                    st.B[i+j+step÷2, p] = b
                end
            end
        end
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Convolution generator
# ───────────────────────────────────────────────────────────────────────────────

@inline function convgen_TV_fast_scalar(cfg::Config, mem::AbstractVector{UIntB}, ui::UIntB, node::Int)
    depth = cfg.depths[node]
    gen2  = @view cfg.Tmat[node - depth:node, node]
    Lg = length(gen2)

    if Lg == 1
        return ui
    end

    s = zero(UInt)
    @inbounds for k in 1:(Lg-1)
        s ⊻= (UInt(mem[k]) & UInt(gen2[k] & 0x01))
    end
    return UIntB((UInt(ui) ⊻ s) & 0x01)
end

function convgen_TV_fast(cfg::Config, mem::AbstractVecOrMat{UIntB}, ui::UIntB, node::Int; ints::Bool=false)
    depth = cfg.depths[node]
    gen2  = @view cfg.Tmat[node - depth:node, node]
    Lg = length(gen2)

    if Lg == 1
        return ui
    end

    if ndims(mem) == 1
        s = zero(UInt)
        @inbounds for k in 1:(Lg-1)
            s ⊻= (UInt(mem[k]) & UInt(gen2[k] & 0x01))
        end
        return UIntB((UInt(ui) ⊻ s) & 0x01)
    else
        # mem is (depth+1, paths): mem[k, path] — inner k-loop is contiguous.
        len, paths = size(mem)
        out = Vector{UIntB}(undef, paths)
        @inbounds for j in 1:paths
            s = zero(UInt)
            for k in 1:(len-1)
                s ⊻= (UInt(mem[k, j]) & UInt(gen2[k] & 0x01))
            end
            out[j] = UIntB((UInt(ui) ⊻ s) & 0x01)
        end
        return out
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Core helpers
# ───────────────────────────────────────────────────────────────────────────────

function updatePM_leaf!(cfg::Config, st::State{T}, node::Int) where {T}
    @inbounds for p in 1:st.l
        u = st.B[node+1, p]
        st.PMs[p] += log(1 + exp(-(1-2*u)*st.LLRs[2*cfg.N - 1, p]))
    end
    return nothing
end

function prunePaths!(cfg::Config, st::State{T}; final = false) where {T}
    L = cfg.L
    if (st.l <= L) && !final
        return nothing
    end

    perm = st.perm_buffer
    @views sortperm!(perm[1:st.l], st.PMs[1:st.l])

    # Bring path perm[i] into slot i using contiguous column swaps.
    @inbounds for i in 1:L
        target = perm[i]
        if target != i
            st.PMs[i], st.PMs[target] = st.PMs[target], st.PMs[i]

            # vx swap
            col_i = view(st.vx, :, i)
            col_t = view(st.vx, :, target)
            copyto!(st.swap_buf_B, col_i)
            copyto!(col_i, col_t)
            copyto!(col_t, st.swap_buf_B)

            # B swap
            col_i = view(st.B, :, i)
            col_t = view(st.B, :, target)
            copyto!(st.swap_buf_B, col_i)
            copyto!(col_i, col_t)
            copyto!(col_t, st.swap_buf_B)

            # LLRs swap
            col_i = view(st.LLRs, :, i)
            col_t = view(st.LLRs, :, target)
            copyto!(st.swap_buf_T, col_i)
            copyto!(col_i, col_t)
            copyto!(col_t, st.swap_buf_T)

            # Fix up perm so future iterations see the updated positions.
            for m in (i+1):L
                if perm[m] == i
                    perm[m] = target
                    break
                end
            end
        end
    end

    # Zero out slots beyond L.
    @views begin
        st.PMs[L+1:st.l] .= zero(T)
        st.vx[:, L+1:st.l] .= 0x00
        st.B[:, L+1:st.l] .= 0x00
        st.LLRs[cfg.N:(2*cfg.N-1), L+1:st.l] .= zero(T)
    end

    st.l = L
    return nothing
end

function duplicateLeafPaths!(cfg::Config, st::State{T}, node::Int, memx) where {T}
    l = st.l
    node1 = node + 1

    @inbounds for i in 1:l
        # memx is (depth+1, l): column i is the memory for path i — contiguous.
        mx = @view memx[:, i]

        conv_x0 = convgen_TV_fast_scalar(cfg, mx, UIntB(0), node1)
        conv_x1 = convgen_TV_fast_scalar(cfg, mx, UIntB(1), node1)

        # Hypothesis 0 stays in slot i.
        st.B[node1, i] = conv_x0

        # Hypothesis 1 goes in slot l+i: copy entire path column then patch.
        dst = l + i
        copyto!(view(st.B,    :, dst), view(st.B,    :, i))
        copyto!(view(st.vx,   :, dst), view(st.vx,   :, i))
        copyto!(view(st.LLRs, :, dst), view(st.LLRs, :, i))
        st.PMs[dst] = st.PMs[i]

        st.vx[node1, dst] = UIntB(1)
        st.B[node1,  dst] = conv_x1
    end

    st.l = 2l
    return nothing
end

# ───────────────────────────────────────────────────────────────────────────────
# Recursive decoder
# ───────────────────────────────────────────────────────────────────────────────

function decode!(cfg::Config, st::State{T}) where{T}
    for node in 0:cfg.N-1
        loop = decimal_to_bin(node, cfg.n)
        while(true)
            if st.d == 1
                depth = cfg.depths[node+1]
                # memx: (depth+1, l) — column per path, contiguous.
                memx = @view st.vx[node+1-depth : node+1, 1:st.l]
                if st.override || cfg.F[node+1]
                    vx_bit = st.syndrome[node+1]
                    @views st.vx[node+1, 1:st.l] .= vx_bit
                    ui_x = convgen_TV_fast(cfg, memx, vx_bit, node + 1)
                    @views st.B[node + 1, 1:st.l] .= ui_x
                    updatePM_leaf!(cfg, st, node)
                else
                    duplicateLeafPaths!(cfg, st, node, memx)
                    updatePM_leaf!(cfg, st, node)
                end
                if cfg.L == 1
                    u  = st.B[node+1, 1]
                    lr = st.LLRs[2*cfg.N - 1, 1]
                    st.temp_B[node+1]    = u
                    st.temp_BMs[node+1]  = log(1 + exp(-(1-2*u)*lr))
                    st.temp_LLRs[node+1] = lr
                end
                if st.l > cfg.L
                    prunePaths!(cfg, st)
                end
                if (node & 0x1 == 1)
                    polarize!(cfg.N, st, node+1)
                else
                    st.βIdx = node+1
                    st.d = 2
                end
                break
            else
                half  = 1 << (st.d-2)
                start = 2*cfg.N - 2^st.d + 1
                stop  = start + 2^(st.d-1) - 1
                left  = start:start+half-1
                right = half+start:stop
                # Inner loop over paths p: LLRs[node, p] — varying p is contiguous.
                @inbounds for (k, i) in enumerate(left)
                    j = right[k]
                    for p in 1:st.l
                        if !loop[cfg.n - (st.d-2)]
                            st.LLRs[stop+k, p] = f(st.LLRs[i, p], st.LLRs[j, p])
                        else
                            u = UIntB(st.B[st.βIdx, p])
                            st.LLRs[stop+k, p] = g(st.LLRs[i, p], st.LLRs[j, p], u)
                        end
                    end
                    if loop[cfg.n - (st.d-2)]
                        st.βIdx += 1
                    end
                end
                st.d -= 1
            end
        end
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────────────────

function decode_channel!(cfg::Config{T}, st::State{T}) where {T}
    decode!(cfg, st)
    prunePaths!(cfg, st; final = true)
    uhatx = st.vx[1:cfg.N, 1] .!= 0
    return uhatx
end

function decode_channel_stats!(cfg::Config{T}, st::State{T}) where {T}
    decode!(cfg, st)
    prunePaths!(cfg, st; final = true)
    return st.vx  # (N, 2L) — callers use vx[1:N, r] for path r
end

function decode_channel_all!(cfg::Config{T}, st::State{T}) where {T}
    decode!(cfg, st)
    prunePaths!(cfg, st)
    return st.temp_B, st.temp_LLRs, st.temp_BMs
end

function successive_encode_channel!(cfg::Config{T}, st::State{T}, valid_sets) where {T}
    decode!(cfg, st)
    prunePaths!(cfg, st)
    w = Weights(1.0 ./ st.PMs[1:st.l])
    for L in sample(collect(1:st.l), w, length(collect(1:st.l)); replace=false)
        uhatx = Int.(st.vx[1:cfg.N, L] .!= 0)
        numx = Misc.bin2dec(uhatx)
        if numx ∈ valid_sets
            return uhatx
        end
    end
    uhatx = Int.(st.vx[1:cfg.N, 1] .!= 0)
    return uhatx
end

function H(p)
    s = 0.0
    @inbounds for v in p
        if v > 0
            s -= v * log2(v)
        end
    end
    return s
end

export Config, State, init_config, init_state, decode_channel!, H, set_syndrome!, reset!, set_llr!, successive_encode_channel!, set_force_decode!, decode_channel_all!, decode_channel_stats!

end # module
