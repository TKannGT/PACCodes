module ClassicalSimFunc

include("ClassicalPolarCodeFunc.jl")
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
    n_threads = Base.Threads.nthreads()
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

function make_llrs(σ, N, F, trials, TMat, T; seed = 7, allZero = false, bsc = false, fastmode = false) 
    info_sets = Vector{Vector{UInt8}}(undef, trials)
    llrs = Vector{Vector{T}}(undef, trials)
    nosyndrome = Vector{UInt8}(undef, trials)
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
        if fastmode
            if sign.(llr) == (1 .- 2 .*infobits) # no sydrome so we can skip it (the boolean logic is backwards but oh well)
                nosyndrome[i] = 0x00  
            else 
                nosyndrome[i] = 0x01 
            end 
        end
    end
    return info_sets, llrs, nosyndrome
end


function parallel_decode(cfg::Config{T}, 
                        info_sets::Vector{Vector{UInt8}}, 
                        llrs::Vector{Vector{T}}, 
                        A; skip_set = nothing) where {T}
    
    n_trials = length(info_sets)
    results = Vector{Int}(undef, n_trials)
    if skip_set === nothing 
        skip_set = trues(n_trials)
    end 
    # Create thread-local states ONCE
    tls = ThreadLocalStates(cfg)
    @threads for i in 1:n_trials
        tid = threadid()  # Get current thread ID
        st = tls.states[tid]  # Use this thread's state
        if skip_set[i]
            u = info_sets[i]
            llr = llrs[i]
            set_llr!(cfg, st, llr)
            uhat = decode_channel!(cfg, st)
            # Compute bit errors (XOR gives 0 or 1, sum gives error count)
            errors = sum(uhat[A] .⊻ u)
            result = errors == 0 ? 0 : 1  # 1 if correct, -1 if errors
            results[i] = result
        else # if it passed with L = 1 then we know its good 
            results[i] = 0 
        end 
        reset!(cfg, st)
    end

    return results
end


function parallel_decode_llr(cfg::Config{T}, 
                        info_sets::Vector{Vector{UInt8}}, 
                        llrs::Vector{Vector{T}}, 
                        A; skip_set = nothing) where {T}
    
    n_trials = length(info_sets)
    results = Vector{Int}(undef, n_trials)
    llr_results = zeros(T, (n_trials, length(llrs[1])))
    # Create thread-local states ONCE
    tls = ThreadLocalStates(cfg)
    
    if skip_set === nothing 
        skip_set = trues(n_trials)
    end

    @threads for i in 1:n_trials
        tid = threadid()  # Get current thread ID
        st = tls.states[tid]  # Use this thread's state
        if skip_set[i] 
            u = info_sets[i]
            set_llr!(cfg, st, llr)
            uhat, llr = decode_channel_returnLLRs!(cfg, st)
            uA = uhat[A]
            A_error = u .⊻ uA 
            if sum(A_error) > 0 
                errorPos = findfirst(x -> x == 1, A_error)
                errorPos = A[errorPos]
            else 
                errorPos = 0 
            end 
            #uhat[A] = u # here we are making uhat correct (this is just because info_sets does not have the frozen bits. same as doing zeros(N) and then adding in)
            #signA = 1 .- 2 .* uhat[A] # this makes u go from 0 -> 1 , 1 -> -1
            results[i] = errorPos
            llr_results[i,:] = llr #.* signA #if an LLR was negative and was supposed to be negative, it becomes positive. If it is the wrong sign, it stays the wrong sign. I am hoping this emulates the simplicity of all zero codeword
        else 
            results[i] = 0
        end 
        reset!(cfg, st)
    end
    fer = sum(sign.(results))/n_trials # anything above 0 was an error 
    error_dict = Dict() 
    llr_dict = Dict() 
    for i in 1:n_trials
        bitN = results[i]
        if bitN > 0 
            if bitN ∈ keys(error_dict) 
                error_dict[bitN] += 1  # basically countmap but as a dictionary 
                llr_dict[bitN] .+= llr_results[i, :]  
            else 
                error_dict[bitN] = 1 
                llr_dict[bitN] = llr_results[i, :] 
            end
        end
    end 
    for i in keys(error_dict) 
        llr_dict[i] = llr_dict[i] ./ error_dict[i] #normalize 
    end 
    return fer, error_dict, llr_dict# pm_results
end

function parallel_decode_pm(cfg::Config{T}, 
                        info_sets::Vector{Vector{UInt8}}, 
                        llrs::Vector{Vector{T}}, 
                        A; skip_set = nothing) where {T}
    
    n_trials = length(info_sets)
    results = Vector{Int}(undef, n_trials)
    pm_set = zeros(T, (n_trials, length(llrs[1])))
    gap_pos = Vector{Int}(undef, n_trials)
    # Create thread-local states ONCE
    tls = ThreadLocalStates(cfg)
    

    @threads for i in 1:n_trials
        tid = threadid()  # Get current thread ID
        st = tls.states[tid]  # Use this thread's state
        u = info_sets[i]
        llr = llrs[i]
        set_llr!(cfg, st, llr)
        error, pm_gap, error_pos = decode_channel_stats!(cfg, st)
        results[i] = error
        gap_pos[i] = error_pos
        pm_set[i] = pm_gap
        reset!(cfg, st)
    end
    fer = sum(results)/n_trials 
    error_dict = Dict() 
    pm_dict = Dict() 
    error_dict_cond = Dict() 
    for i in 1:n_trials
        bitN = gap_pos[i]
        if bitN ∈ keys(error_dict) 
            error_dict[bitN] += 1  # basically countmap but as a dictionary 
            pm_dict[bitN] += pm_set[i]  
            error_dict_cond[bitN] += results[i] # this one is only going up if it caused an error 
        else 
            error_dict[bitN] = 1 
            pm_dict[bitN] = pm_set[i]  
            error_dict_cond[bitN] = results[i]
        end
    end 
    for i in keys(error_dict) 
        pm_dict[i] = pm_dict[i] ./ error_dict[i] #normalize 
    end 
    return fer, error_dict, pm_dict, error_dict_cond
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


function decode_bsc_rate(p, N, L, F, depths, TMat, trials; T=Float64, seed = 7)
    n = Int(log2(N))
    A = setdiff(collect(1:N), F)
    #println("Information set A: ", A)
    
    FB = Vector{Bool}(falses(N))
    FB[F] .= true
    info_sets, llrs = make_llrs(p, N, F, trials, TMat, T; seed=seed, bsc=true)
    cfg = init_config(n=n, L=L, F=FB, depths=depths, Tmat=TMat, T=T)
    result = parallel_decode(cfg, info_sets, llrs, A)
    
    prev_bad = collect(1:trials)
    prev_bad = prev_bad[result .== 1]
    return sum(result)/trials 
end 




function decode_sim(σ, N, L, F, depths, TMat, trials; T=Float64, seed = 7, fast = false, prev_bad = nothing, l_fast = 1)
    n = Int(log2(N))
    A = setdiff(collect(1:N), F)
    #println("Information set A: ", A)
    
    FB = Vector{Bool}(falses(N))
    FB[F] .= true
    info_sets, llrs, nosyndrome = make_llrs(σ, N, F, trials, TMat, T; seed=seed, fastmode = fast)
    if !isnothing(prev_bad) # if we are going over multiple snrs (bad to better), we only need to focus on the ones that failed at the previous snr (assuming identical code construction)
        # I believe there should be no scenario where a code passes on one SNR but fails on a better SNR 
        # Emperically this worked and in theory it makes sense
        info_sets = info_sets[prev_bad]
        llrs = llrs[prev_bad]
    end 
    if fast         
        # in false mode we try everything with a list size of 1 (L times speedup) 
        # we then go over all the failing bits with list size L 
        # importantly: there are scenarios where you actually succeed with L = 1 but NOT with L = L (example: L = 2, PMs go in order [✔,❌] -> [❌,❌,✔,❌]; the L = 2 will get tricked) 
                                                                                                            # L = 1, PMs go in oder  [✔,❌] -> [✔,❌]; the L = 1 won't be tricked) 
        # this means that this will strictly improve FER 
        # this is okay because it is better complexity so I don't think it is cheating to use 
        # no, it is cheating because in the real world you wouldn't know if it's an error 
        cfg = init_config(n=n, L=l_fast, F=FB, depths=depths, Tmat=TMat, T=T)
        result = parallel_decode(cfg, info_sets, llrs, A; skip_set = Bool.(nosyndrome))
        cfg = init_config(n=n, L=L, F=FB, depths=depths, Tmat=TMat, T=T)
        result = parallel_decode(cfg, info_sets, llrs, A; skip_set = Bool.(result))
    else 
        cfg = init_config(n=n, L=L, F=FB, depths=depths, Tmat=TMat, T=T)
        result = parallel_decode(cfg, info_sets, llrs, A)
    end

    # Return success rate (fraction of correct decodings)
    if prev_bad === nothing 
        prev_bad = collect(1:trials)
    end
    prev_bad = prev_bad[result .== 1]
    return sum(result)/trials, prev_bad 
end 


export decode_sim, fast_polarize

end