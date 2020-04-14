# Shuowen Chen and Yang Ming
# Code for innovation project

##################### 0. Import some packages
using LinearAlgebra, Interpolations, Plots, Statistics, Distributed;
# Distributed is the parallelization package in Julia

##################### 1. Setting Parameters
# Fixed industrial profits
pibar = 1;
nbar = 1; # the largest difference of firms' technology states
nGrid = collect(-nbar:nbar); # grid of techonology gaps (negative means A is falling behind)

# R&D cost function R(a) = kappa1*a + 0.5*kappa2*a^2
# A and B denote two firms within the industry
κ1A = 0.0;
κ2A = 3.0;
κ1B = 0.0;
κ2B = 5.0;

# discount factors
β₁ = 0.1;
β₂ = 0.1;

# hazard rate of innovation
λ₁ = 3;
λ₂ = 3;

# grid for R&D Effort (we use the grid search method)
aGridNum = 3000;
aGridMax = 5.0;
aGrid = range(0, aGridMax, length = aGridNum);

##################### 2. Some Auxiliary Functions
# profit for each firm at state n. Only depends on techonology gap
function profit(n, identity, pibar = pibar, nbar = nbar)
    # identity: 1 denotes firm A while 2 denotes firm B
    if identity == 1
        profit = pibar*((n/nbar)+1)*0.5;
    elseif identity == 2
        profit = pibar*((-n/nbar)+1)*0.5;
    end
    return(profit)
end

# R&D cost for each firm at each grid of aGrid
function rdcost(aGrid, identity, κ1A = κ1A, κ2A = κ2A, κ1B = κ1B, κ2B = κ2B)
    # identity: 1 denotes firm A while 2 denotes firm B
    if identity == 1
        cost = κ1A.*aGrid .+ 0.5.*κ2A.*aGrid.^2
    elseif identity == 2
        cost = κ1B.*aGrid .+ 0.5.*κ2B.*aGrid.^2
    end
    return(cost)
end

##################### 3. Value and Policy Function Iterations
# Placeholder for value and policy functions (also functions as initial guess)
vA = zeros(1, length(nGrid));
vB = zeros(1, length(nGrid));
aA = zeros(1, length(nGrid));
aB = zeros(1, length(nGrid));

#####################
# proposition 1
function vpfi(vAinitial = vA, vBinitial = vB, aAinitial = aA, aBinitial = aB, vits = 0, vdiff = 1, vtol = 1E-13, vmaxits = 500,
    atol = 1E-13, amaxits = 500)
    # placeholders for updated policy functions
    aAupdate = zeros(1, length(nGrid));
    aBupdate = zeros(1, length(nGrid));
    # placeholders for updated value functions
    vAupdate = zeros(1, length(nGrid));
    vBupdate = zeros(1, length(nGrid));
    while vdiff > vtol && vits <= vmaxits # value function iteration
        adiff = 1;
        aits = 0;
        # Each time before VFI, first get policy functions: for each state grid, search over aGrid for the optimal R&D action
        while adiff > atol && aits <= amaxits
            for n = 1:length(nGrid)
                if n == 1
                    # vAinitial[n+1] because reduce the gap
                    valueA = (1 ./(λ₁.*aGrid.+β₁)) .* (profit(nGrid[n],1).-rdcost(aGrid,1).+λ₁.*aGrid.*vAinitial[2]);
                    aAupdate[n] = aGrid[argmax(valueA)];
                    aBupdate[n] = 0; # firm b won't conduct R&D
                elseif n > 1 && n < length(nGrid)
                    # now that both firms innovate, either reduce or enlarge the gap
                    valueA = (1 ./(λ₁.*aGrid.+λ₂.*aBinitial[n].+β₁)) .* (profit(nGrid[n],1).-rdcost(aGrid,1).+
                    λ₁.*aGrid.*vAinitial[n+1] .+ λ₂.*aBinitial[n].*vAinitial[n-1]);
                    valueB = (1 ./(λ₁.*aAinitial[n].+λ₂.*aGrid.+β₂)) .* (profit(nGrid[n],2).-rdcost(aGrid,2).+
                    λ₁.*aAinitial[n].*vBinitial[n+1] .+ λ₂.*aGrid.*vBinitial[n-1]);
                    aAupdate[n] = aGrid[argmax(valueA)];
                    aBupdate[n] = aGrid[argmax(valueB)];
                elseif n == length(nGrid) # reverse of n == 1 case
                    # vBinitial[n-1] because reduce the gap
                    valueB = (1 ./(λ₂.*aGrid.+β₂)) .* (profit(nGrid[end],2).-rdcost(aGrid,2).+λ₂.*aGrid.*vBinitial[end-1]);
                    aAupdate[n] = 0;
                    aBupdate[n] = aGrid[argmax(valueB)];
                end
            end
            # update the difference
            adiff = maximum(max.(abs.(aAupdate - aAinitial), abs.(aBupdate - aBinitial)));
            aits = aits + 1;
            # Note: Julia pass by reference: if a=b, then changing elements in b also changes a. Therefore need to add copy() or deepcopy()
            aAinitial = copy(aAupdate);
            aBinitial = copy(aBupdate);
        end
        # Now that we have the policy function, we update the value function
        for n = 2:(length(nGrid)-1)
            vAupdate[n] = (1 /(λ₁*aAupdate[n]+λ₂*aBupdate[n]+β₁)) * (profit(nGrid[n],1)-rdcost(aAupdate[n],1)+λ₁*aAupdate[n]*vAinitial[n+1]+
            λ₂*aBupdate[n]*vAinitial[n-1]);
            vBupdate[n] = (1 /(λ₁*aAupdate[n]+λ₂*aBupdate[n]+β₂)) * (profit(nGrid[n],2)-rdcost(aBupdate[n],2)+λ₁*aAupdate[n]*vBinitial[n+1]+
            λ₂*aBupdate[n]*vBinitial[n-1]);
        end
        vAupdate[1] = (1/(λ₁*aAupdate[1]+β₁)) * (profit(nGrid[1],1)-rdcost(aAupdate[1],1)+λ₁*aAupdate[1]*vAinitial[2]);
        vBupdate[1] = (1/(λ₁*aAupdate[1]+β₂)) * (profit(nGrid[1],2)+λ₁*aAupdate[1]*vBinitial[2]);
        vAupdate[end] = (1/(λ₂*aBupdate[end]+β₁)) * (profit(nGrid[end],1)+λ₂*aBupdate[end]*vAinitial[end-1]);
        vBupdate[end] = (1/(λ₂*aBupdate[end]+β₂)) * (profit(nGrid[end],2)-rdcost(aBupdate[end],2)+λ₂*aBupdate[end]*vBinitial[end-1]);
        # update the difference
        vdiff = maximum(max.(abs.(vAupdate - vAinitial), abs.(vBupdate - vBinitial)));
        vits = vits + 1;
        vAinitial = copy(vAupdate);
        vBinitial = copy(vBupdate);
        # if either firm reaches the maximum grid in R&D effort, break the while loop
        if maximum(max.(aAinitial, aBinitial)) == aGrid[end]
            break;
        end
    end
    # return outputs
    return(vA = vAupdate, vB = vBupdate, aA = aAupdate, aB = aBupdate);
end




testnew = vpfi();
