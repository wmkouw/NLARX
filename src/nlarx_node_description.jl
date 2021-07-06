using ForneyLab
using LinearAlgebra
import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, differentialEntropy, Interface, Variable, slug, ProbabilityDistribution,
				  unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov,
                  collectAverageEnergyInbounds, localEdgeToRegion, ultimatePartner, region, Region, isClamped, currentInferenceAlgorithm
export NLatentAutoregressiveX, NLARX

"""
Description:

    A Nonlinear Latent Autoregressive model with eXogenous input (NLARX).

    The node function is a Gaussian with mean-precision parameterization:

    f(y, θ, x, η, u, γ) = 𝒩(y | A(θ,x) + B(η)u, V(γ)),

    where A(θ,x) is a nonlinear state update, consisting of a data shift
    operation Sx and a nonlinear function of coefficients θ and the previous
    state x; s*g(θ,x) where S = |0 .. 0; I .. 0| and s = [1 .. 0]'. B(η)u a
    scaled linear additive control and V(γ) a covariance matrix based on
    process precision γ.

Interfaces:

    1. y (output vector)
    2. θ (autoregression coefficients)
    3. x (input vector)
    4. η (control coefficients)
    5. u (control)
    6. γ (precision)

Construction:

    NLatentAutoregressiveX(y, θ, x, η, u, γ, g=g, id=:some_id)
"""


mutable struct NLatentAutoregressiveX <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    g::Function # Scalar function between autoregression coefficients and state variable

    function NLatentAutoregressiveX(y, θ, x, η, u, γ; g::Function, id=generateId(NLatentAutoregressiveX))
        @ensureVariables(y, x, θ, η, u, γ)
        self = new(id, Array{Interface}(undef, 6), Dict{Symbol,Interface}(), g)
        addNode!(currentGraph(), self)
        self.i[:y] = self.interfaces[1] = associate!(Interface(self), y)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:θ] = self.interfaces[3] = associate!(Interface(self), θ)
        self.i[:η] = self.interfaces[4] = associate!(Interface(self), η)
        self.i[:u] = self.interfaces[5] = associate!(Interface(self), u)
        self.i[:γ] = self.interfaces[6] = associate!(Interface(self), γ)
        return self
    end
end

slug(::Type{NLatentAutoregressiveX}) = "NLARX"

function averageEnergy(::Type{NLatentAutoregressiveX},
					   g::Function,
                       marg_y::ProbabilityDistribution{Multivariate},
                       marg_x::ProbabilityDistribution{Multivariate},
                       marg_θ::ProbabilityDistribution{Multivariate},
                       marg_η::ProbabilityDistribution{Univariate},
                       marg_u::ProbabilityDistribution{Univariate},
                       marg_γ::ProbabilityDistribution{Univariate})

    # Expectations of marginal beliefs
    my, Vy = unsafeMeanCov(marg_y)
    mx, Vx = unsafeMeanCov(marg_x)
    mθ, Vθ = unsafeMeanCov(marg_θ)
    mη, vη = unsafeMeanCov(marg_η)
    mu, vu = unsafeMeanCov(marg_u)
    mγ = unsafeMean(marg_γ)

    # Gradient of supplied nonlinear function
	Jθ, Jx = gradient(g, mθ, mx)

    # Compute
    Eg = g(mθ, mx)
    Eg2 = Eg*Eg' + Jx'*Vx*Jx + Jθ'*Vθ*Jθ

    # Expand square and pre-compute terms
    sq1 = my[1]^2 + Vy[1,1]
	sq2 = my[1]*(Eg + mη*mu)
	sq3 = Eg2 + 2*Eg*mη*mu + (mη^2 + vη)*(mu + vu)

	# Compute average energy
	AE = 1/2*log(2*π) -1/2*unsafeLogMean(marg_γ) +1/2*mγ*(sq1 -2*sq2 + sq3)

    # correction
    AE += differentialEntropy(marg_y)
    marg_y1 = ProbabilityDistribution(Univariate, GaussianMeanVariance, m=my[1], v=Vy[1,1])
    AE -= differentialEntropy(marg_y1)

    return AE
end

function collectAverageEnergyInbounds(node::NLatentAutoregressiveX)
    inbounds = Any[]

	# Push function to calling signature (g needs to be defined in user scope)
	push!(inbounds, Dict{Symbol, Any}(:g => node.g, :keyword => false))

    local_edge_to_region = localEdgeToRegion(node)

    encountered_regions = Region[] # Keep track of encountered regions
    for node_interface in node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_region = region(node_interface.node, node_interface.edge)

        if isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(copy(inbound_interface.node), ProbabilityDistribution)) # Copy Clamp before assembly to prevent overwriting dist_or_msg field
        elseif !(current_region in encountered_regions)
            # Collect marginal entry from marginal dictionary (if marginal entry is not already accepted)
            target = local_edge_to_region[node_interface.edge]
            current_inference_algorithm = currentInferenceAlgorithm()
            push!(inbounds, current_inference_algorithm.target_to_marginal_entry[target])
        end

        push!(encountered_regions, current_region)
    end

    return inbounds
end 
