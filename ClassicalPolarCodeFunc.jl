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


mutable struct State{T}
    l::Int
    d::Int
    βIdx::Int 
    PMs::Vector{T}
    B::Array{UIntB,2}
    LLRs::Array{T,2}
    vx::Matrix{UIntB}
    syndrome::Vector{UIntB}
    llr::Array{T}
    # NEW: Preallocated buffers to avoid allocations
    perm_buffer::Vector{Int}
    temp_PMs::Vector{T}
end

# This decoder is actually setting the frozen bits to the syndrome bits (st.syndrome) 
# This allows it to do source coding with side information and syndrome decode
# st.syndrome defaults to 0, if this is left untouched it is essentially doing channel coding

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

    PMs = zeros(T, paths)
    B   = fill(UIntB(0), paths, N)
    vx  = fill(UIntB(0), paths, N)

    LLRs = zeros(T, paths, 2*N - 1)
    #LLRs[paths, 1:cfg.N] .= llr 
    llr = zeros(T,N)
    sx = isnothing(syndrome) ? zeros(UIntB, N) : UIntB.(syndrome)

    # NEW: Preallocate buffers
    perm_buffer = Vector{Int}(undef, paths)
    temp_PMs = Vector{T}(undef, paths)

    return State{T}(1, d, 1, PMs, B, LLRs, vx, sx, llr, perm_buffer, temp_PMs)
end


function set_syndrome!(cfg::Config, st::State, syndrome::AbstractVector)
    copyto!(st.syndrome, UIntB.(syndrome))
    return st
end

function set_llr!(cfg::Config, st::State{T}, llr::Array{T}) where{T}
    st.LLRs[:,1:cfg.N] .= llr'
    return st
end


function reset!(cfg::Config{T}, st::State{T}) where {T}
    st.l = 1
    st.d = cfg.d
    st.βIdx = 1  # Add this line!
    fill!(st.PMs, zero(T))
    fill!(st.B, 0x00)
    fill!(st.vx, 0x00)
    fill!(st.LLRs, zero(T))
    return st
end


# ───────────────────────────────────────────────────────────────────────────────
# Probability combiners
# ───────────────────────────────────────────────────────────────────────────────

@inline function f(l1::T, l2::T) where {T}
    T(
       min(abs(l1),abs(l2))*sign(l1)*sign(l2)
    )
end

@inline function g(l1::T, l2::T, u::UIntB) where {T}
    T(l2 + (1-2*u)*l1)
end


# ───────────────────────────────────────────────────────────────────────────────
# Polar projection
# ───────────────────────────────────────────────────────────────────────────────

function num_polar_steps(node::Int)
    Tlev = if ispow2(node+1)
        Int(log2(node+1))
    else
        t = 1
        parent1 = (node - 1) >>> 1
        while (parent1 & 0x1 == 1)
            t += 1
            parent1 = (parent1 - 1) >>> 1
        end
        t
    end
end

@inline function polarize!(N::Int, st::State{T}, node::Int) where{T}
    n = num_polar_steps(node)
    st.βIdx -= 2^n - 1
    st.d = n+2
    startSpot = node - 1 
    for level in 1:n
        step = 1 << level
        for i in (node-step+1):step:node
            for j in 0:(step÷2-1)
                for p in 1:st.l 
                    a = st.B[p,i+j]
                    b = st.B[p,i+j+step÷2]
                    st.B[p,i+j] = (a ⊻ b) 
                    st.B[p,i+j+step÷2] = b
                end
            end
        end
    end
end

# ───────────────────────────────────────────────────────────────────────────────
# Convolution generator with reduced branching
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
        paths, len = size(mem)
        out = Vector{UIntB}(undef, paths)
        @inbounds for j in 1:paths
            s = zero(UInt)
            for k in 1:(len-1)
                s ⊻= (UInt(mem[j, k]) & UInt(gen2[k] & 0x01))
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
    N = cfg.N
    @inbounds for p in 1:st.l
        u = st.B[p, node+1]
        st.PMs[p] += log(1 + exp(-(1-2*u)*st.LLRs[p, 2*cfg.N - 1]))
    end
    return nothing
end

# OPTIMIZED: Use sortperm! with preallocated buffer and manual swapping
function prunePaths!(cfg::Config, st::State{T}; final = false) where {T}
    L = cfg.L
    if (st.l <= L) && !final  
        return nothing
    end

    # Use preallocated buffer for permutation
    perm = st.perm_buffer
    @views sortperm!(perm[1:st.l], st.PMs[1:st.l])
    
    # Swap paths in-place to avoid allocations
    @inbounds for i in 1:L
        target = perm[i]
        if target != i
            # Swap path metrics
            st.PMs[i], st.PMs[target] = st.PMs[target], st.PMs[i]
           

            # Swap vx columns
            for j in 1:size(st.vx, 2)
                st.vx[i, j], st.vx[target, j] = st.vx[target, j], st.vx[i, j]
            end
            
            # Swap B slices
            for j in 1:size(st.B, 2)
                st.B[i, j], st.B[target, j] = st.B[target, j], st.B[i, j]
            end
            
            # Swap LLRs slices
            for j in 1:size(st.LLRs, 2)
                st.LLRs[i, j], st.LLRs[target, j] = st.LLRs[target, j], st.LLRs[i, j]
            end

            # Update perm to reflect the swap
            for m in (i+1):L
                if perm[m] == i
                    perm[m] = target
                    break
                end
            end
        end
    end

    # Reset tail
    @views begin
        st.PMs[L+1:st.l] .= zero(T)
        st.vx[L+1:st.l, :] .= 0x00
        st.B[L+1:st.l, :] .= 0x00
        st.LLRs[L+1:st.l, cfg.N:(2*cfg.N-1)] .= zero(T)
    end

    st.l = L
    return nothing
end

# OPTIMIZED: Reduce allocations and redundant operations
function duplicateLeafPaths!(cfg::Config, st::State{T}, node::Int, memx) where {T}
    l = st.l
    N = cfg.N
    node1 = node + 1
    
    @inbounds for i in 1:l
        # Cache source path index for readability
        src = i
        
        # Compute convolutions once with ui=0
        depth = cfg.depths[node1]
        mx = @view memx[i, :]
        
        # Precompute for all 4 hypotheses
        conv_x0 = convgen_TV_fast_scalar(cfg, mx, UIntB(0), node1)
        conv_x1 = convgen_TV_fast_scalar(cfg, mx, UIntB(1), node1)
        # Hypothesis 0 (0,0): stays in slot i - only update B[1, node1, :]
        st.B[src, node1] = conv_x0
        
        # Replicate to three new slots
        dst = l + i
        # Copy entire rows more efficiently
        for j in 1:size(st.B, 2)
            st.B[dst, j] = st.B[src, j]
        end

            
        for j in 1:size(st.vx, 2)
            st.vx[dst,j] = st.vx[src,j]
        end
            

        for j in 1:size(st.LLRs, 2)
            st.LLRs[dst, j] = st.LLRs[src, j]
        end
            
       st.PMs[dst] = st.PMs[src]
        
        # Hypothesis 1 (1)
        st.vx[dst, node1] = UIntB(1)
        st.B[dst, node1] = conv_x1
    end
    
    st.l = 2l
    return nothing
end



# ───────────────────────────────────────────────────────────────────────────────
# Recursive decoder
# ───────────────────────────────────────────────────────────────────────────────



function decode!(cfg::Config, st::State{T}) where{T}
    for node in 0:cfg.N-1
        loop = decimal_to_bin(node,cfg.n)
        while(true)
            if st.d == 1 # we have reached the end 
                depth = cfg.depths[node+1]
                memx = @view st.vx[1:st.l, node+1-depth : node+1]
                if cfg.F[node+1]
                    vx_bit = st.syndrome[node+1]
                    @views st.vx[1:st.l, node+1] .= vx_bit
                    ui_x = convgen_TV_fast(cfg, memx, vx_bit, node + 1)
                    @views st.B[1:st.l, node + 1] .= ui_x
                    updatePM_leaf!(cfg, st, node) 
                else
                    duplicateLeafPaths!(cfg, st, node, memx)
                    updatePM_leaf!(cfg, st, node)
                end
                if st.l > cfg.L
                    prunePaths!(cfg, st)
                end
                if (node & 0x1 == 1)
                    polarize!(cfg.N, st, node+1)
                else 
                    st.βIdx = node+1 
                    st.d = 2 # this is the option for the next node  
                end 
                #@show st.B[1,:]
                break # this will take us to the next node 
            else  
                half = 1 << (st.d-2) 
                powd = 1 << st.d 
                start = 2*cfg.N - 2^st.d + 1
                stop  = start + 2^(st.d-1) - 1
                left  = start:start+half-1
                right = half+start:stop
                #βIdxTemp = copy(st.βIdx) # this needs to reset every loop that way it can increment for each list 
                @inbounds for (k, i) in enumerate(left)
                    j = right[k]
                    for p in 1:st.l
                        if !loop[cfg.n - (st.d-2)]
                            st.LLRs[p,stop+k] = f(st.LLRs[p,i], st.LLRs[p,j])
                        else
                            u = UIntB(st.B[p, st.βIdx])
                            st.LLRs[p,stop+k] = g(st.LLRs[p,i], st.LLRs[p,j],u)
                        end
                    end
                    if loop[cfg.n - (st.d-2)]
                        st.βIdx+=1
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

    uhatx = st.vx[1, 1:cfg.N] .!= 0
    return uhatx
end

function successive_encode_channel!(cfg::Config{T}, st::State{T}, valid_sets) where {T}
    decode!(cfg, st)

    prunePaths!(cfg, st)
    w = Weights(1.0 ./ st.PMs[1:st.l])   
    for L in sample(collect(1:st.l), w, length(collect(1:st.l)); replace=false)
        uhatx = Int.(st.vx[L, 1:cfg.N] .!= 0)
        numx = Misc.bin2dec(uhatx) 
        if numx ∈ valid_sets 
            return uhatx 
        end 
    end 
    uhatx = Int.(st.vx[1, 1:cfg.N] .!= 0)
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

export Config, State, init_config, init_state, decode_channel!, H, set_syndrome!, reset!, set_llr!, successive_encode_channel!

end # module