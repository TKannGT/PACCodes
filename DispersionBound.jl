
  
using Distributions 
using QuadGK

N= 128
R= 85/N
EbN0dB=2:0.1:4.6
EbN0=10 .^(EbN0dB/10); 
SNR=EbN0*2*R; 
Zmin = -9; Zmax = 9;
epsilonNA = zeros(length(SNR))
domain = (Zmin, Zmax)
for j ∈ 1:length(SNR) 
    rho = SNR[j]
    f(z) = exp(-z.^2/2)/sqrt(2*pi) .* (log2(2) - log2(1+exp(-2*rho-2*sqrt(rho)*z)));
    C, _ = quadgk(z -> f(z), Zmin, Zmax, rtol=1e-8)
    g(z) = exp(-z .^2/2)/sqrt(2*pi) .* (log2(2) - log2(1+exp(-2*rho-2*sqrt(rho)*z))-C) .^2
    V, _ = quadgk(z -> g(z), Zmin, Zmax, rtol=1e-8)
    x = (C-R+0.5/N*log2(N))./(sqrt(V/N))
   epsilonNA[j] = 1 - cdf(Normal(0, 1), x)
end 

println(collect(EbN0dB))
println(epsilonNA)