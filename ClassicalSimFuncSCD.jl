module ClassicalSimFuncSCD

include("ClassicalPolarCodeFuncSCD.jl")
include("../Misc.jl")
using .ClassicalPolarCodeFunc, .Misc
using LinearAlgebra, Random
using Kronecker, Random
using Statistics, Distributions
using QuadGK, StatsBase
using Plots, JLD2
using Base.Threads
const UIntB = UInt8

# ============================================================================
# Create one State per thread to avoid false sharing
# ============================================================================

"""
Thread-local storage for decoders - each thread gets its own State
This eliminates false sharing and lock contention
"""
struct ThreadLocalStates{T}
    states::Vector{State{T}}
    cfgs::Vector{Config{T}}
end

function ThreadLocalStates(cfg::Config{T}) where {T}
    n_threads = nthreads()
    states = [init_state(cfg) for _ in 1:n_threads]
    cfgs = [cfg for _ in 1:n_threads]  # Config is immutable, can share
    return ThreadLocalStates{T}(states, cfgs)
end

# ============================================================================
# Noise Generation 
# ============================================================================

function genAWGN(σ, N, F, TMat, T ; rng = MersenneTwister(7), allZero = false) # p = crossover prob , x ~ bern(b)
    A = setdiff(collect(1:N), F)
    K = length(A) 
    bernoulliU = Bernoulli(.5)
    
    # Fix: Use Vector instead of matrix to avoid indexing issues
    u = zeros(UIntB, N)  # Changed from Int to UIntB, and from 1xN to N
    if !allZero # sometimes we want to look at all zero codeword 
        u[A] = rand(rng, bernoulliU, K)  # This now correctly assigns 0 or 1
    end
    x = fast_polarize(u, N, TMat)
    v = Float64.(1 .- 2 .* x)  # Modulate: 0→1, 1→-1
    y = v .+ σ .* randn(rng, N)  # Noisy channel 
    llr = 2 .* y ./ σ^2  # LLRs for AWGN 
    
    # Return info bits as Vector{UIntB}
    return Vector{UIntB}(u[A]), T.(llr) 
end 


function genBSC(p, N, F, TMat, T; rng = MersenneTwister(7))
    A = setdiff(collect(1:N), F)
    K = length(A)
    bernoulliU = Bernoulli(0.5)

    # Input vector
    u = zeros(UIntB, N)
    u[A] = rand(rng, bernoulliU, K)
    
    # Polar encode
    x = fast_polarize(u, N, TMat)   # expected to be bits {0,1}

    # Pass through BSC(p)
    flips = rand(rng, Bernoulli(p), N)
    r = x .⊻ UIntB.(flips)          # received bits after random flips

    # LLRs for BSC
    llr0 = log((1 - p) / p)
    llr = (1 .- 2 .* Float64.(r)) .* llr0

    return Vector{UIntB}(u[A]), T.(llr)
end


# ============================================================================
# parallel syndrome generation with thread-safe RNG
# ============================================================================

function make_llrs(σ, N, F, trials, TMat, T; seed = 7, allZero = false, bsc = false) 
    info_sets = Vector{Vector{UInt8}}(undef, trials)
    llrs = Vector{Vector{T}}(undef, trials)

    # Create thread-safe RNGs: one per trial with deterministic seeds
    rngs = [MersenneTwister(seed + i) for i in 1:trials]

    @threads for i in 1:trials
        # Each thread uses its own RNG - no race conditions!
        if bsc
            infobits, llr = genBSC(σ, N, F, TMat, T; rng = rngs[i])
        else
            infobits, llr = genAWGN(σ, N, F, TMat, T; rng = rngs[i], allZero = allZero)
        end    
        @assert all(infobits .>= 0) "Negative infobit detected: $(infobits)"
        @assert all(infobits .<= 1) "Invalid infobit (>1) detected: $(infobits)"
        
        info_sets[i] = infobits  # Already UIntB from genAWGN
        llrs[i] = vec(llr)
    end
    
    return info_sets, llrs
end


function parallel_decode_SCD(cfg::Config{T}, 
                        info_sets::Vector{Vector{UInt8}}, 
                        llrs::Vector{Vector{T}}, 
                        A, error_index, bits_beating_SCl, bad_paths) where {T}
    
    n_trials = sum(length, bits_beating_SCl) # all number of needed trials 
    AllB = Vector{Vector{Int}}(undef, n_trials)
    AllLLR = Vector{Vector{Int}}(undef, n_trials)
    AllBM = Vector{Vector{Int}}(undef, n_trials)
    # Create thread-local states ONCE
    tls = ThreadLocalStates(cfg)
    BM_dict = Dict() 
    LLR_dict = Dict() 
    B_dict = Dict() 
    for a in A 
        BM_dict[a] = [] 
        LLR_dict[a] = [] 
        B_dict[a] = [] 
    end 

    @threads for i in 1:n_trials
        tid = threadid()  # Get current thread ID
        st = tls.states[tid]  # Use this thread's state
        if error_index[i] == -1 
            continue 
        end 
        u = info_sets[i]
        llr = llrs[i]
        set_llr!(cfg, st, llr)
        allB = zeros(Int, (size(bits_beating_SCl[i]), cfg.N)) 
        allLLR = zeros(size(bits_beating_SCl[i], cfg.N))
        allBM = zeros(size(bits_beating_SCl[i], cfg.N))
        for b in bits_beating_SCl[i]
            path = bad_paths[b] 
            set_force_decode!(cfg, st, path)
            B, LLR, BM = decode_channel_all!(cfg, st) 
            push!(allB, B) 
            push!(allLLR, LLR)
            push!(allBM, BM)
            reset!(cfg, st)
        end 
        AllB[i] = allB
        AllLLR[i] = allLLR
        AllBM[i] = allBM
    end
    for i in 1:n_trials
        if error_index[i] == - 1 
            continue
        end 
        for b in bits_beating_SCl[i]
            push!(B_dict[b], AllB[i,b])  
            push!(LLR_dict[b], AllLLR[i,b])  
            push!(BM_dict[b], AllBM[i,b])  
        end
    end 
    set_errors = set(flatten(bits_beating_SCl)) 
    for b in set_errors
        m = mean(B_dict[b]) 
        v = var(B_dict[b]) 
        B_dict[b] = (m,v)
        m = mean(LLR_dict[b]) 
        v = var(LLR_dict[b]) 
        LLR_dict[b] = (m,v)
        m = mean(BM_dict[b]) 
        v = var(BM_dict[b]) 
        BM_dict[b] = (m,v)
    end 
    return B_dict, BM_dict, LLR_dict
end




function get_SCD_set(cfg::Config{T}, 
                        info_sets::Vector{Vector{UInt8}}, 
                        llrs::Vector{Vector{T}}, 
                        A; skip_set = nothing) where {T}
    
    n_trials = length(info_sets)
    error_index = Vector{Int}(undef, n_trials)
    bits_beating_SCl = Vector{Vector{Int}}(undef, n_trials)
    bad_paths = Vector{Vector{Vector{Int}}}(undef, n_trials)
    # Create thread-local states ONCE
    tls = ThreadLocalStates(cfg)
    

    @threads for i in 1:n_trials
        tid = threadid()  # Get current thread ID
        st = tls.states[tid]  # Use this thread's state
        u = info_sets[i]
        llr = llrs[i]
        if minimum(llr) > 0  # if it is all positive we will 100% decode correct
            error_index[i] = -1 
            bits_beating_SCl[i] = [-1]
            continue
        end
        set_llr!(cfg, st, llr)
        vhatx = decode_channel_stats!(cfg, st)
        bad_bits = []  # these will be all the paths that are beating out the list decoder 
        good_bits = 0  # this will mark if the all-zero path was even in the final output (to help distinguish between the two errors)
        for l in 1:cfg.L
            v = vhatx[1:cfg.N, l] .!= 0 # get path l
            idx = findfirst(!iszero, v) 
            if idx == nothing 
                good_bits = 1  
                continue
            end 
            append!(bat_bits, idx) 
        end 
        if length(bad_bits) == 0 # no error 
            error_index[i] = -1 # our flag to ignore 
            bits_beating_SCl[i] = [] 
            bad_paths[i] = [] 
        else 
            error_index[i] = i # this LLR was an error 
            bits_beating_SCl[i] = bad_bits # we need to care about all of these 
            bad_paths[i] = vhatx[1:length(bits_beating_SCl), 1:cfg.N]
        end
        reset!(cfg, st)
    end
    return error_index, bits_beating_SCl, bad_paths
end


# ============================================================================
# Main decode function 
# ============================================================================
function fast_polarize(u, N, TMat)
    # Use the polar code structure instead of full matrix mult
    n = Int(log2(N))
    
    # Ensure u is a vector
    temp = vec(u)
    
    # Apply TMat first (convolutional encoding)
    temp = vec(mod.(temp' * TMat, 2))
    
    # Iterative polar transform (O(N log N) instead of O(N²))
    for level in 1:n
        step = 1 << level
        half_step = step ÷ 2
        for i in 1:step:N
            for j in 0:(half_step-1)
                idx1 = i + j
                idx2 = idx1 + half_step
                a = temp[idx1]
                b = temp[idx2]
                temp[idx1] = (a ⊻ b) % 2
                temp[idx2] = b
            end
        end
    end

    return temp
end




function decode_main(σ, N, L, F, depths, TMat, trials; T=Float64, seed = 7)
    n = Int(log2(N))
    A = setdiff(collect(1:N), F)
    #println("Information set A: ", A)
    
    FB = Vector{Bool}(falses(N))
    FB[F] .= true
    info_sets, llrs = make_llrs(σ, N, F, trials, TMat, T; allZero = true, seed=seed)
    cfg = init_config(n=n, L=L, F=FB, depths=depths, Tmat=TMat, T=T)
    error_index, bits_beating_SCl, bad_paths = get_SCD_set(cfg, info_sets, llrs, A) 
    cfg = init_config(n=n, L=1, F=FB, depths=depths, Tmat=TMat, T=T)
    error_index, bits_beating_SCl, bad_paths = parallel_decode_SCD(cfg, info_sets, llrs, A, error_index, bits_beating_SCl)
    return error_index, bits_beating_SCl, bad_paths
end 


export decode_main, fast_polarize

end