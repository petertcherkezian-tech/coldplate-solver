// ============================================================================
// File: Utilities/iterations.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Implements a single SIMPLE/SIMPLEC iteration on the staggered Cartesian grid.
//   This is the heart of the CFD solver, executing the following steps each iteration:
//
//   1. SAVE OLD VELOCITIES
//      Store u, v for residual calculation and pseudo-transient term
//
//   2. SOLVE U-MOMENTUM (x-direction)
//      - Assemble coefficient matrix with diffusion + convection (FOU or SOU)
//      - Add pseudo-transient term: ρV/Δt * (u - u_old) for stability
//      - Add 2.5D sink term: -(5μ/2Ht²) * u for out-of-plane friction
//      - Apply under-relaxation: u* = α*u_new + (1-α)*u_old
//      - Compute d-coefficient for velocity correction
//
//   3. SOLVE V-MOMENTUM (y-direction)
//      Same process as U-momentum but for vertical velocity
//
//   4. SET VELOCITY BOUNDARY CONDITIONS
//      Apply inlet (fixed velocity), outlet (zero gradient), wall (no-slip)
//
//   5. SOLVE PRESSURE CORRECTION (Poisson equation)
//      - Assemble sparse matrix from continuity equation
//      - Solve using direct LDLT or iterative SOR
//      - Returns mass residual (continuity error)
//
//   6. CORRECT PRESSURE
//      p = p + α_p * p'  (relaxed pressure update)
//
//   7. CORRECT VELOCITIES
//      u = u* + d_E * (p'_P - p'_E)  where d = Δx / a_P (SIMPLE)
//      v = v* + d_N * (p'_P - p'_N)  or d = Δx / (a_P - Σa_nb) (SIMPLEC)
//
//   8. MASS BALANCE CORRECTION
//      Scale outlet velocity to enforce global mass conservation
//
//   9. UPDATE CFL RAMP
//      Increase pseudo-CFL as residuals decrease for faster convergence
//
// SIMPLE vs SIMPLEC:
//   The key difference is in the velocity correction d-coefficient:
//   - SIMPLE:  d = Δx / (a_P + Σa_nb)  -- neglects neighbor corrections
//   - SIMPLEC: d = Δx / (a_P - Σa_nb)  -- accounts for neighbor corrections
//   SIMPLEC is more consistent and often allows larger relaxation factors.
//
// Brinkman Penalization:
//   For topology optimization, solid regions are penalized rather than blocked:
//   - Add source term: F = -α(γ) * u, where α = α_max * (1 - γ)
//   - γ = 1 (fluid): no drag, γ = 0 (solid): maximum drag
//   - This allows gradient-based optimization with continuous γ values
//
// Pseudo-Transient Time Stepping:
//   Adds artificial unsteady term to steady equations for stability:
//   - ρV/Δt * (u - u_old) added to momentum source
//   - Δt computed from CFL: Δt = CFL * Δx / |u|
//   - CFL increases as solution converges (residual-based ramping)
//   - At true steady state, (u - u_old) → 0, so term vanishes
//
// ============================================================================
#include "SIMPLE.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

// Define PI if not already defined (for SOR optimal omega calculation)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// calculateStep: Perform one complete SIMPLE iteration
// ============================================================================
// This function orchestrates a single outer iteration of the SIMPLE algorithm.
// It solves momentum equations, pressure correction, and velocity correction.
//
// Parameters:
//   pressureIterations [out]: Number of inner pressure sweeps (for logging)
//
// Returns:
//   Maximum residual (max of |Δu|, |Δv|, mass imbalance)
// ============================================================================
double SIMPLE::calculateStep(int& pressureIterations)
{
    // -------------------------------------------------------------------------
    // Cell volume and pseudo-transient setup
    // -------------------------------------------------------------------------
    const double vol = hx * hy;                        // Cell volume [m³/m depth]
    const bool pseudoActive = enablePseudoTimeStepping; // Is pseudo-transient enabled?
    
    // Thread-local reduction variables for OpenMP parallelization
    double localResidU = 0.0;        // Max |u_new - u_old|
    double localResidV = 0.0;        // Max |v_new - v_old|
    double localResidMass = 0.0;     // Max mass imbalance
    
    // Transient term residual tracking (should → 0 at steady state)
    double localTransResidU = 0.0;   // Max |ρV/Δt * (u_new - u_old)|
    double localTransResidV = 0.0;   // Max |ρV/Δt * (v_new - v_old)|

    // -------------------------------------------------------------------------
    // Pseudo-transient Δt calculation setup
    // -------------------------------------------------------------------------
    // Use minimum cell size for CFL stability
    const double hChar = std::min(hx, hy);
    
    // Statistics for pseudo-Δt (for diagnostic logging)
    double uDtMin = timeStep, uDtMax = 0.0, uDtSum = 0.0;
    double vDtMin = timeStep, vDtMax = 0.0, vDtSum = 0.0;
    long long uDtCount = 0;
    long long vDtCount = 0;

    // Lambda to update min/max/avg statistics for pseudo-Δt
    auto updateStats = [&](double dt, double& minDt, double& maxDt, double& sumDt, long long& count) {
        minDt = std::min(minDt, dt);
        maxDt = std::max(maxDt, dt);
        sumDt += dt;
        count++;
    };

    // -------------------------------------------------------------------------
    // Lambda: Compute local pseudo-Δt based on CFL condition
    // -------------------------------------------------------------------------
    // Δt = CFL * Δx / |u|, clamped to global maximum timeStep
    // This allows faster convergence in low-velocity regions while maintaining
    // stability in high-velocity regions.
    auto computeLocalDt = [&](double normalVel) -> double {
        if (!pseudoActive) return std::numeric_limits<double>::infinity();
        if (!useLocalPseudoTime) return timeStep;  // Use global Δt if local is disabled
        double speed = std::max(std::abs(normalVel), minPseudoSpeed);  // Prevent divide-by-zero
        double dtCfl = pseudoCFL * hChar / speed;  // CFL-based local Δt
        return std::min(timeStep, dtCfl);  // Clamp to global maximum
    };

    // -------------------------------------------------------------------------
    // Step 0: Save old velocities for residual calculation
    // -------------------------------------------------------------------------
    uOld = u;
    vOld = v;

    // -------------------------------------------------------------------------
    // Velocity bounds for stability (prevent unbounded growth)
    // -------------------------------------------------------------------------
    const double maxVel = 3.0 * std::max(std::abs(inletVelocity), 0.1);
    
    // -------------------------------------------------------------------------
    // Convection scheme setup
    // -------------------------------------------------------------------------
    const bool useSOU = (convectionScheme == 1);  // Second-order upwind?
    
    // 2.5D model: scale convection by 6/7 (accounts for parabolic profile)
    const double convScale = enableTwoPointFiveD ? (6.0 / 7.0) : 1.0;
    
    // 2.5D sink coefficient: -(5μ/2Ht²) * multiplier for parallel-plate friction
    // Base is (5/2)μ/Ht²; multiplier ≈4.8 gives 12μ/Ht² (Poiseuille friction)
    const double sinkCoeff = (enableTwoPointFiveD && Ht_channel > 0.0)
                                 ? twoPointFiveDSinkMultiplier * (5.0 * eta / (2.0 * Ht_channel * Ht_channel))
                                 : 0.0;
    const double sinkDiag = sinkCoeff * vol;  // Diagonal contribution from sink term
    
    // =========================================================================
    // STEP 1: U-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    #pragma omp parallel for collapse(2) reduction(max:localResidU,localTransResidU) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            
            // Fast mask check instead of checkBoundaries + isSolidCell
            if (!fluidU(*this, i, j) || checkBoundaries(i, j) == 1.0) {
                uStar(i, j) = 0.0;
                dE(i, j) = 0.0;
                continue;
            }

            double De = eta * hy / hx;
            double Dn = eta * hx / hy;

            // Face velocities for flux calculation
            double ue = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j) + u(i, j + 1))));
            double uw = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j - 1) + u(i, j))));
            double vn = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i - 1, j) + v(i - 1, j + 1))));
            double vs = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i, j) + v(i, j + 1))));

            double Fe = convScale * rho * hy * ue;
            double Fw = convScale * rho * hy * uw;
            double Fn = convScale * rho * hx * vn;
            double Fs = convScale * rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            double aE = De + std::max(0.0, -Fe);
            double aW = De + std::max(0.0, Fw);
            double aN = Dn + std::max(0.0, -Fn);
            double aS = Dn + std::max(0.0, Fs);

            double Sdc = useSOU ? computeSOUCorrectionU(*this, i, j, Fe, Fw, Fn, Fs) : 0.0;

            double sumA = aE + aW + aN + aS;
            double dtLocal = computeLocalDt(u(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, uDtMin, uDtMax, uDtSum, uDtCount);
            }
            double transCoeffLocal = rho * vol / dtLocal;
            
            // Brinkman penalization: alpha * vol adds drag in porous/solid regions
            double alphaLocal = alphaAtU(*this, i, j);
            double brinkmanDrag = alphaLocal * vol;

            double aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal + sinkDiag + brinkmanDrag;
            aP0 = std::max(aP0, 1e-10);

            // d-coefficient for velocity correction: u' = d * (p'_P - p'_E)
            // SIMPLE:  d = A / a_P           (a_P already includes neighbor coefficients)
            // SIMPLEC: d = A / (a_P - Σa_nb) (more consistent pressure-velocity coupling)
            double dDenom = useSIMPLEC ? std::max(aP0 - sumA, 1e-12) : std::max(aP0, 1e-12);
            dE(i, j) = hy / dDenom;

            double Sp = (p(i, j) - p(i, j + 1)) * hy;
            Sp += transCoeffLocal * uOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            double aP = aP0 / uvAlpha;
            Sp += (1.0 - uvAlpha) / uvAlpha * aP0 * u(i, j);

            double uNew = (aE * u(i, j + 1) + aW * u(i, j - 1) +
                          aN * u(i - 1, j) + aS * u(i + 1, j) + Sp) / aP;

            uNew = std::max(-maxVel, std::min(maxVel, uNew));

            double diff = std::abs(uNew - u(i, j));
            if (diff > localResidU) localResidU = diff;
            
            // Track transient term residual: |transCoeff * (uNew - u)|
            // At true steady state, this should be ~0
            // This measures how much the pseudo-transient term is affecting the solution
            double transTermU = transCoeffLocal * diff;
            if (transTermU > localTransResidU) localTransResidU = transTermU;
            
            uStar(i, j) = uNew;
        }
    }
    residU = localResidU;
    transientResidU = localTransResidU;

    // =========================================================================
    // STEP 2: V-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    localResidV = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(max:localResidV,localTransResidV) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            
            // Fast mask check instead of checkBoundaries + isSolidCell
            if (!fluidV(*this, i, j) || checkBoundaries(i, j) == 1.0) {
                vStar(i, j) = 0.0;
                dN(i, j) = 0.0;
                continue;
            }

            double De = eta * hy / hx;
            double Dn = eta * hx / hy;

            // Face velocities for flux calculation
            double ue = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j) + u(i + 1, j))));
            double uw = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j - 1) + u(i + 1, j - 1))));
            double vn = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i, j) + v(i + 1, j))));
            double vs = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i - 1, j) + v(i, j))));

            double Fe = convScale * rho * hy * ue;
            double Fw = convScale * rho * hy * uw;
            double Fn = convScale * rho * hx * vn;
            double Fs = convScale * rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            double aE = De + std::max(0.0, -Fe);
            double aW = De + std::max(0.0, Fw);
            double aN = Dn + std::max(0.0, -Fn);
            double aS = Dn + std::max(0.0, Fs);

            double Sdc = useSOU ? computeSOUCorrectionV(*this, i, j, Fe, Fw, Fn, Fs) : 0.0;

            double sumA = aE + aW + aN + aS;
            double dtLocal = computeLocalDt(v(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, vDtMin, vDtMax, vDtSum, vDtCount);
            }
            double transCoeffLocal = rho * vol / dtLocal;
            
            // Brinkman penalization: alpha * vol adds drag in porous/solid regions
            double alphaLocal = alphaAtV(*this, i, j);
            double brinkmanDrag = alphaLocal * vol;

            double aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal + sinkDiag + brinkmanDrag;
            aP0 = std::max(aP0, 1e-10);

            // d-coefficient for velocity correction: v' = d * (p'_P - p'_N)
            // SIMPLE:  d = A / a_P           (a_P already includes neighbor coefficients)
            // SIMPLEC: d = A / (a_P - Σa_nb) (more consistent pressure-velocity coupling)
            double dDenom = useSIMPLEC ? std::max(aP0 - sumA, 1e-12) : std::max(aP0, 1e-12);
            dN(i, j) = hx / dDenom;

            double Sp = (p(i, j) - p(i + 1, j)) * hx;
            Sp += transCoeffLocal * vOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            double aP = aP0 / uvAlpha;
            Sp += (1.0 - uvAlpha) / uvAlpha * aP0 * v(i, j);

            double vNew = (aE * v(i, j + 1) + aW * v(i, j - 1) +
                          aN * v(i + 1, j) + aS * v(i - 1, j) + Sp) / aP;

            vNew = std::max(-maxVel, std::min(maxVel, vNew));

            double diff = std::abs(vNew - v(i, j));
            if (diff > localResidV) localResidV = diff;
            
            // Track transient term residual: |transCoeff * (vNew - v)|
            double transTermV = transCoeffLocal * diff;
            if (transTermV > localTransResidV) localTransResidV = transTermV;
            
            vStar(i, j) = vNew;
        }
    }
    residV = localResidV;
    transientResidV = localTransResidV;

    setVelocityBoundaryConditions(uStar, vStar);

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && uDtCount > 0) {
        pseudoStatsU.min = uDtMin;
        pseudoStatsU.max = uDtMax;
        pseudoStatsU.avg = uDtSum / double(uDtCount);
        pseudoStatsU.samples = uDtCount;
        pseudoStatsU.valid = true;
    } else {
        pseudoStatsU.valid = false;
    }

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && vDtCount > 0) {
        pseudoStatsV.min = vDtMin;
        pseudoStatsV.max = vDtMax;
        pseudoStatsV.avg = vDtSum / double(vDtCount);
        pseudoStatsV.samples = vDtCount;
        pseudoStatsV.valid = true;
    } else {
        pseudoStatsV.valid = false;
    }

    // =========================================================================
    // STEP 3: PRESSURE CORRECTION (Iterative SOR or Direct Sparse Solver)
    // =========================================================================
    bool directSolverSucceeded = solvePressureSystem(pressureIterations, localResidMass);
    residMass = localResidMass;

    // =========================================================================
    // STEP 4: CORRECT PRESSURE (Parallelized with precomputed masks)
    // =========================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            if (fluidP(*this, i, j)) {
                p(i, j) += pAlpha * pStar(i, j);
            }
        }
    }

    // Set pressure boundary conditions (outlet reference pressure = 0.0)
    // No normalization needed - boundary condition handles reference pressure
    setPressureBoundaryConditions(p);

    // =========================================================================
    // STEP 5: CORRECT VELOCITIES (Parallelized)
    // =========================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            if (checkBoundaries(i, j) == 1.0) continue;
            if (alphaAtU(*this, i, j) > 1e6) continue;
            
            double du = dE(i, j) * (pStar(i, j) - pStar(i, j + 1));
            du = std::max(-maxVel, std::min(maxVel, du));
            u(i, j) = uStar(i, j) + du;
            u(i, j) = std::max(-maxVel, std::min(maxVel, u(i, j)));
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            if (checkBoundaries(i, j) == 1.0) continue;
            if (alphaAtV(*this, i, j) > 1e6) continue;
            
            double dv = dN(i, j) * (pStar(i, j) - pStar(i + 1, j));
            dv = std::max(-maxVel, std::min(maxVel, dv));
            v(i, j) = vStar(i, j) + dv;
            v(i, j) = std::max(-maxVel, std::min(maxVel, v(i, j)));
        }
    }

    setVelocityBoundaryConditions(u, v);

    // =========================================================================
    // STEP 6: MASS BALANCE CORRECTION (Serial - small loop)
    // =========================================================================
    double massIn = 0.0, massOut = 0.0;
    for (int i = 1; i < M; ++i) {
        massIn += rho * u(i, 0) * hy;
        massOut += rho * u(i, N - 1) * hy;
    }

    if (std::abs(massOut) > 1e-10 && std::abs(massIn) > 1e-10) {
        double ratio = massIn / massOut;
        if (ratio > 0.5 && ratio < 2.0) {
            for (int i = 1; i < M; ++i) {
                u(i, N - 1) *= ratio;
            }
        }
    }

    return std::max({residU, residV, residMass});
}
