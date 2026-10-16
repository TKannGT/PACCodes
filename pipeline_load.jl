# ============================================================
# pipeline_load.jl
#
# Small utility for reading the per-run output folders produced by
# `RunGenericPACSetThenSparseTV_SigmaLLR.jl` (or its variants).  Each
# run writes files like:
#
#   pac_sparse_pipeline_out/<timestamp>/
#       A_selected.txt          ← info set from Stage 1
#       T_constructed.csv       ← sparse precoder from Stage 2
#       T_generic.csv           ← the generic PAC Tmat used in Stage 1
#       A_after_stage3.txt      ← present only if Stage 3 ran and committed
#       T_after_stage3.csv      ← present only if Stage 3 ran and committed
#       pac_edges.csv
#       set_history.csv
#       summary.txt
#
# This module loads the (A, T) pair from such a folder so downstream
# scripts (pac_weight_demo, pac_spectrum_search, etc.) can analyze the
# code without re-typing the info set and precoder.
#
# Usage:
#     include(joinpath(@__DIR__, "pipeline_load.jl"))
#     run = load_pipeline_output("path/to/pac_sparse_pipeline_out/20260105_142312")
#     run.A         # → Vector{Int} (1-indexed positions)
#     run.T         # → Matrix{Int} (N × N, upper-triangular, 1s on diagonal)
#     run.N, run.K  # → code dimensions
#
# Pass `which = :stage3` or `:constructed` or `:generic` to override the
# default selection.  Default is `:best`, which prefers Stage 3 if it ran,
# falls back to Stage 2 (constructed) otherwise.
# ============================================================

using DelimitedFiles
using Printf

"""
    load_pipeline_output(outdir; which = :best, verbose = true)

Read a pipeline output folder and return (A, T) plus metadata.

Arguments:
- `outdir`: path to one of the timestamped run directories.
- `which`:  which (A, T) to load.
    * `:best`        — Stage 3 if it ran, else Stage 2 constructed (default).
    * `:stage3`      — A_after_stage3.txt + T_after_stage3.csv (errors if missing).
    * `:constructed` — A_selected.txt + T_constructed.csv.
    * `:generic`     — A_selected.txt + T_generic.csv.
- `verbose`: print a short summary of what was loaded.

Returns a NamedTuple with fields:
    A          ::Vector{Int}      sorted info-set positions (1-indexed)
    T          ::Matrix{Int}      N×N precoder matrix (upper-triangular, diag = 1)
    N, K       ::Int              code dimensions
    n_edges    ::Int              off-diagonal nonzero count in T
    A_path     ::String           which A file was read
    T_path     ::String           which T file was read
    outdir     ::String           the input outdir
    which      ::Symbol           selection that was actually used
"""
function load_pipeline_output(outdir::AbstractString;
                                which::Symbol = :best,
                                verbose::Bool = true)
    isdir(outdir) ||
        throw(ArgumentError("Not a directory: $outdir"))

    # Candidate file paths.
    A_stage3       = joinpath(outdir, "A_after_stage3.txt")
    A_selected     = joinpath(outdir, "A_selected.txt")
    T_stage3       = joinpath(outdir, "T_after_stage3.csv")
    T_constructed  = joinpath(outdir, "T_constructed.csv")
    T_generic      = joinpath(outdir, "T_generic.csv")

    # Resolve which pair to load.
    A_path, T_path, used = if which === :best
        if isfile(A_stage3) && isfile(T_stage3)
            (A_stage3, T_stage3, :stage3)
        elseif isfile(A_selected) && isfile(T_constructed)
            (A_selected, T_constructed, :constructed)
        else
            throw(ArgumentError("No (A, T) pair found in $outdir"))
        end
    elseif which === :stage3
        (isfile(A_stage3) && isfile(T_stage3)) ||
            throw(ArgumentError("Stage 3 outputs not present in $outdir"))
        (A_stage3, T_stage3, :stage3)
    elseif which === :constructed
        (isfile(A_selected) && isfile(T_constructed)) ||
            throw(ArgumentError("constructed outputs not present in $outdir"))
        (A_selected, T_constructed, :constructed)
    elseif which === :generic
        (isfile(A_selected) && isfile(T_generic)) ||
            throw(ArgumentError("generic outputs not present in $outdir"))
        (A_selected, T_generic, :generic)
    else
        throw(ArgumentError("which must be :best, :stage3, :constructed, or :generic; got $which"))
    end

    # Load A: one position per line, 1-indexed.
    A_raw = readdlm(A_path, Int)
    A = sort(unique(vec(A_raw)))

    # Load T: comma-separated N × N integer matrix.
    T_raw = readdlm(T_path, ',', Int)
    size(T_raw, 1) == size(T_raw, 2) ||
        throw(ArgumentError("T at $T_path is not square (got $(size(T_raw)))"))
    T = Matrix{Int}(T_raw)
    N = size(T, 1)

    # Sanity checks.
    maximum(A) <= N || throw(ArgumentError("A has position > N (max(A)=$(maximum(A)), N=$N)"))
    minimum(A) >= 1 || throw(ArgumentError("A has position < 1 (min(A)=$(minimum(A)))"))
    @inbounds for j in 1:N
        T[j, j] == 1 ||
            throw(ArgumentError("T[$j, $j] = $(T[j,j]) (expected 1 on the diagonal)"))
    end
    @inbounds for j in 1:N, i in (j + 1):N
        T[i, j] == 0 ||
            throw(ArgumentError("T[$i, $j] = $(T[i,j]) below diagonal (T must be upper-triangular)"))
    end

    K = length(A)
    n_edges = count(==(1), T) - N  # subtract diagonal

    if verbose
        @printf("Loaded pipeline output\n")
        @printf("  folder:    %s\n", outdir)
        @printf("  selection: %s\n", String(used))
        @printf("  A from:    %s\n", basename(A_path))
        @printf("  T from:    %s\n", basename(T_path))
        @printf("  N = %d, K = %d, R = %.4f\n", N, K, K / N)
        @printf("  off-diagonal edges in T: %d\n", n_edges)
    end

    return (A = A, T = T, N = N, K = K, n_edges = n_edges,
            A_path = A_path, T_path = T_path, outdir = outdir, which = used)
end

"""
    list_pipeline_runs(parent_dir; latest_first = true)

List timestamped run subdirectories under `parent_dir` (e.g.,
`pac_sparse_pipeline_out`), sorted by modification time.  Useful for
finding the most recent run interactively.
"""
function list_pipeline_runs(parent_dir::AbstractString;
                              latest_first::Bool = true)
    isdir(parent_dir) ||
        throw(ArgumentError("Not a directory: $parent_dir"))
    entries = filter(p -> isdir(joinpath(parent_dir, p)), readdir(parent_dir))
    paths = [joinpath(parent_dir, e) for e in entries]
    sort!(paths; by = mtime, rev = latest_first)
    return paths
end

"""
    pipeline_edges(T)

Return the list of off-diagonal (src, tgt) edges in a Tmat as a
Vector{Tuple{Int,Int}}.  Filters to upper-triangular entries with value 1.
Useful for piping into scripts that take edges as tuples.
"""
function pipeline_edges(T::AbstractMatrix)
    N = size(T, 1)
    edges = Tuple{Int,Int}[]
    @inbounds for i in 1:N, j in (i + 1):N
        if Int(T[i, j]) != 0
            push!(edges, (i, j))
        end
    end
    return edges
end

"""
    pipeline_edges_in_A(T, A)

Same as `pipeline_edges` but filters to edges whose source bit is in the
info set `A`.  Frozen-source edges contribute nothing to the codebook,
so this is the relevant edge list for weight enumeration and most
downstream analysis.
"""
function pipeline_edges_in_A(T::AbstractMatrix, A::AbstractVector{<:Integer})
    A_set = Set(Int.(A))
    N = size(T, 1)
    edges = Tuple{Int,Int}[]
    @inbounds for i in 1:N, j in (i + 1):N
        if Int(T[i, j]) != 0 && i in A_set
            push!(edges, (i, j))
        end
    end
    return edges
end

# ============================================================
# Quick REPL example (uncomment to run after include):
# ============================================================
# parent = joinpath(@__DIR__, "pac_sparse_pipeline_out")
# runs   = list_pipeline_runs(parent)
# run    = load_pipeline_output(runs[1])
# println("\nA = ", run.A)
# println("\nEdges (in-A only): ", pipeline_edges_in_A(run.T, run.A))