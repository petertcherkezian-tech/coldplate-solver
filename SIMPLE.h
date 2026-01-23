// ============================================================================
// File: SIMPLE.h
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Core interface and parameters for the incompressible laminar SIMPLE/SIMPLEC
//   solver on a structured Cartesian grid using the Finite Volume Method (FVM).
//
//   This header serves as the CENTRAL REGISTRY for:
//     1. All solver control parameters (relaxation, CFL ramps, convergence criteria)
//     2. Algorithm toggles (SIMPLE vs SIMPLEC, direct vs SOR pressure solve)
//     3. Physics options (2.5D model, Brinkman penalization for topology optimization)
//     4. Staggered grid field storage (u, v, p matrices)
//     5. Geometry and boundary condition data structures
//     6. Function declarations for utilities in Utilities/*.cpp
//
//   STAGGERED GRID LAYOUT (MAC - Marker And Cell):
//   ┌─────────────────────────────────────────────────────────────────────────┐
//   │  The staggered grid stores variables at different locations:            │
//   │                                                                         │
//   │     p(i,j)         p(i,j+1)                                            │
//   │       ●──────u(i,j)──────●      ● = pressure (cell centers)            │
//   │       │               │          ──│ = u-velocity (vertical faces)      │
//   │       │               │          ═══ = v-velocity (horizontal faces)    │
//   │    v(i,j)   CELL    v(i,j+1)                                           │
//   │       │    (i,j)      │                                                │
//   │       │               │         Grid indexing:                         │
//   │       ●──────u(i+1,j)─●           u: (M+1) x N  (at vertical faces)    │
//   │     p(i+1,j)      p(i+1,j+1)      v: M x (N+1)  (at horizontal faces)  │
//   │                                    p: (M+1) x (N+1) (cell centers+ghost)│
//   └─────────────────────────────────────────────────────────────────────────┘
//
//   COORDINATE SYSTEM:
//     - i = row index (y-direction), increases downward (fluid flow direction)
//     - j = column index (x-direction), increases to the right (main flow direction)
//     - Origin at top-left corner of the domain
//
// ============================================================================
#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <omp.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

class SIMPLE {

public:

    // =========================================================================
    // CONVERGENCE CONTROL
    // =========================================================================
    // These parameters determine when the solver stops iterating.
    // The solver runs until EITHER:
    //   1. All residuals drop below epsilon, OR
    //   2. Pressure drop stabilizes (if usePressureDropConvergence=true), OR
    //   3. maxIterations is reached
    
    double epsilon       = 1e-7;      // Target residual threshold for convergence (lower = tighter)
    int    maxIterations = 100000;    // Maximum outer SIMPLE iterations (safety limit)
    
    // =========================================================================
    // OUTPUT & RESTART CONTROL
    // =========================================================================
    // Checkpointing allows resuming from saved state if solver crashes or
    // you want to refine the solution further.
    
    int    saveStateInterval = 600;   // Save checkpoint every N iterations (0 = disable)
    bool   reuseInitialFields = false;// If true, load u.txt/v.txt/p.txt as initial guess
    std::string restartDirectory = "ExportFiles";  // Directory for restart files

    // =========================================================================
    // PRESSURE SAMPLING PLANES
    // =========================================================================
    // Physical x-coordinates [m] where pressure is sampled to compute pressure drop.
    // "Core" planes exclude inlet/outlet buffers for cleaner pressure drop measurement.
    // "Full" planes span the entire domain (auto-calculated: 0 to N*hx).
    
    double xPlaneCoreInlet  = 0.01;   // Core inlet sampling plane [m] (after inlet buffer)
    double xPlaneCoreOutlet = 0.050;  // Core outlet sampling plane [m] (before outlet buffer)

    // =========================================================================
    // PRESSURE-DROP BASED CONVERGENCE
    // =========================================================================
    // Alternative convergence criterion: stop when pressure drop stabilizes.
    // Useful when residuals oscillate but the engineering quantity of interest
    // (pressure drop) has already converged.
    
    bool   usePressureDropConvergence = true;   // Enable this convergence mode
    int    dpConvergenceWindow = 1500;          // Moving window size [iterations]
    double dpConvergencePercent = 1.0;          // Converged if std_dev < this % of mean
    bool   usePressureDropSlopeGate = false;    // Additional check: slope must be near-zero
    int    dpSlopeWindowIters = 1000;           // Window for slope calculation
    double dpSlopeMaxDelta   = 10.0;            // Max allowed Δp change [Pa] over window
    
    // =========================================================================
    // PARALLELIZATION (OpenMP)
    // =========================================================================
    // Multi-threading for momentum loops and matrix assembly.
    // Best performance with number of PHYSICAL cores (not hyperthreads).
    
    int numThreads = 8;  // Number of OpenMP threads (recommend: physical core count)
    
    // =========================================================================
    // ALGORITHM SELECTION: SIMPLE vs SIMPLEC
    // =========================================================================
    // SIMPLE  (Semi-Implicit Method for Pressure-Linked Equations):
    //   - Original Patankar-Spalding algorithm
    //   - Uses under-relaxation for stability
    //   - d = A / a_P  where a_P already includes neighbor coefficients
    //
    // SIMPLEC (SIMPLE-Consistent):
    //   - Improved variant by Van Doormaal & Raithby
    //   - Uses d = A / (a_P - Σa_nb)  [more consistent pressure-velocity coupling]
    //   - Often allows larger relaxation factors → faster convergence
    
    bool useSIMPLEC = false;  // true = SIMPLEC, false = standard SIMPLE
    
    // =========================================================================
    // PRESSURE CORRECTION SOLVER OPTIONS
    // =========================================================================
    // Two options for solving the pressure-correction (Poisson) equation:
    //   1. Direct: Eigen SimplicialLDLT factorization (fast, memory-intensive)
    //   2. Iterative: Successive Over-Relaxation (SOR) with red-black ordering
    
    bool useDirectPressureSolver = true;  // true = direct LDLT, false = SOR
    bool parallelizeDirectSolver = true;  // Parallelize matrix assembly (recommended)
    
    // SOR-specific parameters (only used when useDirectPressureSolver = false):
    int    maxPressureIter = 50;   // Max inner pressure sweeps per outer iteration
    double sorOmega = 0.0;         // SOR relaxation factor (0 = auto-compute optimal ω)
    double pTol = 1e-4;            // Pressure correction convergence tolerance

    // =========================================================================
    // CONVECTION SCHEME
    // =========================================================================
    // Controls spatial discretization of convective (advective) terms:
    //   0 = First-Order Upwind (FOU): Simple, stable, diffusive
    //   1 = Second-Order Upwind (SOU): Higher accuracy, uses deferred correction
    //       to maintain diagonal dominance while improving flux approximation
    
    int convectionScheme = 0;  // 0 = FOU, 1 = SOU

    // =========================================================================
    // 2.5D REDUCED-ORDER MODEL
    // =========================================================================
    // For microchannel heat sinks with tall, thin fins, the 2.5D model captures
    // out-of-plane (z-direction) effects without solving full 3D equations:
    //   - Convection scaled by 6/7 (accounts for non-uniform velocity profile)
    //   - Adds linear drag term: F = -(5μ/2Ht²) * u  [parallel-plate friction]
    //
    // Reference: "Two-layer microchannel model" for topology optimization
    
    bool   enableTwoPointFiveD = true;    // Enable 2.5D out-of-plane corrections
    double Ht_channel = 0.0;              // Out-of-plane channel height [m] (from geometry file)
    double twoPointFiveDSinkMultiplier = 4.8; // Multiplier on base sink coefficient
        // Base = (5/2)μ/Ht²; multiplier ≈4.8 gives 12μ/Ht² (parallel-plate friction)

    // =========================================================================
    // DENSITY-BASED TOPOLOGY OPTIMIZATION (Brinkman Penalization)
    // =========================================================================
    // For topology optimization, geometry is described by a continuous density
    // field γ (gamma) rather than binary solid/fluid:
    //   - γ = 1.0 → pure fluid (no drag)
    //   - γ = 0.0 → pure solid (high drag = blocked flow)
    //   - 0 < γ < 1 → transition zone (gradual drag)
    //
    // Brinkman penalization adds a Darcy-like drag term to momentum:
    //   F = -α(γ) * u,  where α(γ) = α_max * (1 - γ)
    //
    // α_max is computed from the Darcy number: α_max = μ / (Da * L²)
    // Lower Da → stronger penalization → more "solid-like" behavior
    //
    // Reference: Haertel et al., "Topology optimization of microchannel heat sinks"
    
    double brinkmanDarcyNumber = 1e-5;  // Darcy number Da (lower = stronger penalization)
    double brinkmanAlphaMax = 0.0;      // Computed at runtime: μ / (Da * L_c²)

    // =========================================================================
    // PSEUDO-TRANSIENT TIME STEPPING
    // =========================================================================
    // Pseudo-transient continuation adds artificial time derivative to steady
    // equations: ∂u/∂t + [steady terms] = 0. This improves robustness for
    // difficult problems by allowing the solution to "evolve" toward steady state.
    //
    // The pseudo time step can be:
    //   - Global: same Δt everywhere (simpler, more stable)
    //   - Local (CFL-based): Δt = CFL * Δx / |u|  (faster convergence)
    //
    // CFL ramping increases CFL as residuals decrease, accelerating convergence
    // once the solution is close to steady state.
    
    double timeStepMultiplier = 0.01;     // Global Δt = multiplier * (Lx / U_inlet)
    double timeStep = 0.0;                // Computed global pseudo time step [s]
    bool   enablePseudoTimeStepping = true; // Master switch for pseudo-transient
    double pseudoCFL = 0.1;               // Current CFL number (updated by ramp)
    double minPseudoSpeed = 0.05;         // Minimum velocity for CFL calculation [m/s]
    bool   useLocalPseudoTime = true;     // true = local CFL, false = global Δt
    bool   logPseudoDtStats = false;      // Print min/max/avg pseudo-Δt each iteration
    
    double pseudoRefLength = 0.0;         // Reference length for CFL [m] (0 = auto)

    // CFL ramping parameters (residual-based acceleration):
    bool   enableCflRamp    = true;       // Enable automatic CFL increase
    double pseudoCFLInitial = 0.1;        // Starting CFL (conservative)
    double pseudoCFLMax     = 5;          // Maximum CFL (aggressive)
    double cflRampStartRes  = 5e-4;       // Begin ramping when residual drops below this
    double cflRampExponent  = 0.8;        // CFL ∝ (Res_start / Res_current)^exponent
    double cflRampSmooth    = 0.1;        // Smoothing factor (0.1 = 90% old + 10% new)
    
    // Transient residual tracking (monitors convergence to true steady state):
    // At true steady state, these should approach zero
    double transientResidU = 0.0;   // max |ρV/Δt * (u_new - u_old)| for U-momentum
    double transientResidV = 0.0;   // max |ρV/Δt * (v_new - v_old)| for V-momentum

    // =========================================================================
    // GRID DIMENSIONS & SPACING
    // =========================================================================
    // Structured Cartesian grid with uniform spacing.
    // These values are read from ExportFiles/fluid_params.txt.
    
    int M = 0;       // Number of rows (y-direction cells)
    int N = 0;       // Number of columns (x-direction cells)
    int N_in_buffer = 0;   // Inlet buffer zone columns (for thermal cropping)
    int N_out_buffer = 0;  // Outlet buffer zone columns (for thermal cropping)
    double hy = 0.0; // Cell height [m] (Δy)
    double hx = 0.0; // Cell width [m] (Δx)

    // =========================================================================
    // FLUID PROPERTIES
    // =========================================================================
    // Properties for water at 20°C. These are HARDCODED here and NOT read from
    // the geometry file, ensuring consistent physics across geometry variations.
    
    double rho = 997.0;       // Fluid density [kg/m³]
    double eta = 0.00089;     // Dynamic viscosity [Pa·s] (μ)

    // =========================================================================
    // INLET VELOCITY & RAMPING
    // =========================================================================
    // Inlet ramping gradually increases velocity from 10% to 100% of target
    // over rampSteps iterations. This prevents numerical instabilities at startup.
    
    double targetVel        = 0.0;    // Final inlet velocity [m/s] (from geometry file)
    double inletVelocity    = 0.0;    // Current inlet velocity [m/s] (may be ramped)
    bool   enableInletRamp  = false;  // Enable gradual inlet velocity ramp
    int    rampSteps        = 1000;   // Iterations to reach full inlet velocity

    // =========================================================================
    // UNDER-RELAXATION FACTORS
    // =========================================================================
    // Under-relaxation stabilizes the iterative process by blending new values
    // with old: φ_new = α·φ_calculated + (1-α)·φ_old
    //
    // Lower values = more stable but slower convergence
    // Typical ranges: velocity 0.3-0.8, pressure 0.1-0.3
    
    double uvAlpha = 0.5;    // Velocity under-relaxation factor (0 < α ≤ 1)
    double pAlpha  = 0.2;    // Pressure under-relaxation factor (0 < α ≤ 1)

    // =========================================================================
    // RESIDUAL TRACKING
    // =========================================================================
    // Residuals measure how far the current solution is from satisfying the
    // governing equations. They should decrease each iteration toward epsilon.
    
    double residU    = 1.0;  // Maximum U-velocity change (|u_new - u_old|)
    double residV    = 1.0;  // Maximum V-velocity change (|v_new - v_old|)
    double residMass = 1.0;  // Maximum mass imbalance (continuity equation error)

    // =========================================================================
    // PSEUDO-TIME STEP STATISTICS
    // =========================================================================
    // Diagnostic structure for tracking local pseudo-Δt distribution.
    // Useful for debugging CFL-based time stepping.
    
    struct PseudoDtStats {
        double min = 0.0;       // Minimum pseudo-Δt in the domain
        double max = 0.0;       // Maximum pseudo-Δt in the domain
        double avg = 0.0;       // Average pseudo-Δt
        long long samples = 0;  // Number of cells sampled
        bool valid = false;     // Whether stats were computed this iteration
    };

    PseudoDtStats pseudoStatsU;  // Stats for U-momentum pseudo-Δt
    PseudoDtStats pseudoStatsV;  // Stats for V-momentum pseudo-Δt

    // =========================================================================
    // STAGGERED VELOCITY FIELDS
    // =========================================================================
    // Velocities are stored at cell FACES (not centers) to ensure natural
    // coupling with pressure gradients and prevent checkerboard oscillations.
    //
    // u: horizontal velocity at VERTICAL faces, size (M+1) x N
    //    u(i,j) = velocity at the left face of cell (i-1,j) / right face of (i-1,j-1)
    //
    // v: vertical velocity at HORIZONTAL faces, size M x (N+1)
    //    v(i,j) = velocity at the bottom face of cell (i,j-1) / top face of (i-1,j-1)
    
    Eigen::MatrixXd u;       // Current x-velocity field
    Eigen::MatrixXd v;       // Current y-velocity field
    Eigen::MatrixXd uStar;   // Intermediate u (after momentum solve, before correction)
    Eigen::MatrixXd vStar;   // Intermediate v (after momentum solve, before correction)
    Eigen::MatrixXd uOld;    // Previous iteration u (for residual calculation)
    Eigen::MatrixXd vOld;    // Previous iteration v (for residual calculation)

    // =========================================================================
    // PRESSURE FIELD (Cell-Centered)
    // =========================================================================
    // Pressure is stored at cell CENTERS, with ghost cells at boundaries.
    // Size: (M+1) x (N+1) to accommodate ghost layers.
    //
    // pStar is the pressure CORRECTION (p' in SIMPLE algorithm), not the
    // intermediate pressure. Final pressure: p = p_old + α_p * p'
    
    Eigen::MatrixXd p;       // Pressure field [Pa]
    Eigen::MatrixXd pStar;   // Pressure correction field [Pa]

    // =========================================================================
    // MOMENTUM EQUATION COEFFICIENTS
    // =========================================================================
    // These "d" coefficients appear in the velocity correction equations:
    //   u' = dE * (p'_P - p'_E)  for U-momentum
    //   v' = dN * (p'_P - p'_N)  for V-momentum
    //
    // They represent the sensitivity of velocity to pressure gradient.
    
    Eigen::MatrixXd dE;      // d-coefficient for U-velocity correction
    Eigen::MatrixXd dN;      // d-coefficient for V-velocity correction
    Eigen::MatrixXd b;       // Work array (used internally)

    // =========================================================================
    // GEOMETRY FIELDS
    // =========================================================================
    // cellType: Raw geometry input (0=fluid, 1=solid, 0-1=buffer in V3)
    // gamma:    Porosity/density field (1=fluid, 0=solid) - inverted from cellType
    // alpha:    Brinkman penalization coefficient (high = solid, low = fluid)
    
    Eigen::MatrixXd cellType;  // Geometry from file (0=fluid, 1=solid)
    Eigen::MatrixXd alpha;     // Brinkman α field: α = α_max * (1 - γ)
    Eigen::MatrixXd gamma;     // Density field: γ = 1 (fluid), γ = 0 (solid)

    // =========================================================================
    // PRECOMPUTED FLUID MASKS
    // =========================================================================
    // Boolean masks indicating which locations are in fluid regions.
    // Precomputed once after geometry load for fast lookup during iteration.
    // Using flat std::vector instead of 2D array for cache efficiency.
    
    std::vector<bool> isFluidU;   // true if u(i,j) is in fluid (neither neighbor is solid)
    std::vector<bool> isFluidV;   // true if v(i,j) is in fluid (neither neighbor is solid)
    std::vector<bool> isFluidP;   // true if p(i,j) is in fluid (cell itself is not solid)
    void buildFluidMasks();       // Compute masks from cellType (called after geometry load)

    // =========================================================================
    // CONSTRUCTOR & INITIALIZATION
    // =========================================================================
    
    SIMPLE();                                           // Constructor: loads params, geometry, initializes fields
    void loadParameters(const std::string& paramsFile); // Read grid size, spacing, inlet velocity from file
    void initializeMemory();                            // Allocate all field matrices

    // =========================================================================
    // MAIN SOLVER METHODS
    // =========================================================================
    
    void runIterations();                               // Main iteration loop: runs until convergence or max iterations
    double calculateStep(int& pressureIterations);      // Perform ONE SIMPLE iteration (momentum → pressure → correction)

    // =========================================================================
    // SOLVER SUBCOMPONENTS (implemented in Utilities/*.cpp)
    // =========================================================================
    
    bool solvePressureSystem(int& pressureIterations, double& localResidMass);  // Pressure correction solve
    void updateCflRamp(double currRes);                 // Update pseudo-CFL based on current residual

    // =========================================================================
    // GEOMETRY LOADING
    // =========================================================================
    
    void loadTopology(const std::string& fileName);     // Read cellType from geometry_fluid.txt
    void loadDensityField();                            // Convert cellType → gamma (invert 0↔1)
    void buildAlphaFromDensity();                       // Compute α = α_max * (1 - γ)

    // =========================================================================
    // BOUNDARY CONDITIONS
    // =========================================================================
    
    void setVelocityBoundaryConditions(Eigen::MatrixXd& uIn, Eigen::MatrixXd& vIn);  // Apply inlet/outlet/wall BCs
    void setPressureBoundaryConditions(Eigen::MatrixXd& pIn);                         // Apply pressure BCs (outlet = 0)
    double checkBoundaries(int i, int j);               // Returns 1.0 if (i,j) is at domain boundary, 0.0 otherwise
    void paintBoundaries();                             // Debug: export boundary map to file

    // =========================================================================
    // OUTPUT METHODS
    // =========================================================================
    
    void saveAll();                                     // Save all fields (u, v, p, VTK) to ExportFiles/
    void saveMatrix(Eigen::MatrixXd inputMatrix, std::string fileName);  // Save single matrix to text file
    void initLogFiles(std::ofstream& residFile, std::ofstream& dpFile);  // Create residual/pressure log files
    void printIterationHeader() const;                  // Print column headers to console
    void writeIterationLogs(std::ofstream& residFile,   // Write iteration data to log files
                            std::ofstream& dpFile,
                            int iter,
                            double corePressureDrop,
                            double fullPressureDrop,
                            double coreStaticDrop,
                            double fullStaticDrop);
    void printIterationRow(int iter,                    // Print one row of iteration data to console
                           double residMassVal,
                           double residUVal,
                           double residVVal,
                           double maxTransRes,
                           double corePressureDrop,
                           double fullPressureDrop,
                           double iterTimeMs,
                           int pressureIterations) const;
    void printStaticDp(int iter,                        // Print static pressure drop (debug)
                       double coreStaticDrop,
                       double fullStaticDrop) const;

};

// =============================================================================
// INLINE HELPER FUNCTIONS
// =============================================================================
// These helpers provide fast lookup for fluid masks and interpolated values
// at staggered grid locations. They are used extensively in the momentum
// assembly loops (iterations.cpp) for efficiency.

// -----------------------------------------------------------------------------
// Fluid Mask Lookups
// -----------------------------------------------------------------------------
// Check if a given u/v/p location is in the fluid region.
// Uses precomputed boolean masks stored as flat vectors for cache efficiency.
// Returns true if the location should be solved, false if it's solid (skip).

// Check if u-velocity at (i,j) is in fluid region
// u-grid has size (M+1) x N, so linear index = i * N + j
inline bool fluidU(const SIMPLE& s, int i, int j) {
    return s.isFluidU[i * s.N + j];
}

// Check if v-velocity at (i,j) is in fluid region
// v-grid has size M x (N+1), so linear index = i * (N+1) + j
inline bool fluidV(const SIMPLE& s, int i, int j) {
    return s.isFluidV[i * (s.N + 1) + j];
}

// Check if pressure at (i,j) is in fluid region
// p-grid has size (M+1) x (N+1), so linear index = i * (N+1) + j
inline bool fluidP(const SIMPLE& s, int i, int j) {
    return s.isFluidP[i * (s.N + 1) + j];
}

// -----------------------------------------------------------------------------
// Alpha (Brinkman Penalization) Interpolation
// -----------------------------------------------------------------------------
// Since alpha is defined at cell CENTERS but u/v are at cell FACES, we need
// to interpolate alpha to the velocity locations using a simple average of
// the two adjacent cells.
//
// For u(i,j): average alpha from cells (i-1, j-1) and (i-1, j)  [left & right]
// For v(i,j): average alpha from cells (i-1, j-1) and (i, j-1)  [top & bottom]

// Get Brinkman alpha at U-velocity location (i,j)
// Returns the average of alpha from the two cells sharing this u-face
inline double alphaAtU(const SIMPLE& s, int i, int j) {
    const auto& alpha = s.alpha;
    const int M = s.M;
    const int N = s.N;
    
    // Boundary check: return 0 (no drag) if outside valid interior
    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 0.0;
    
    // Cell indices for the two cells sharing this u-face
    int ci = i - 1;      // Row index in cellType/alpha
    int cj1 = j - 1;     // Left cell column
    int cj2 = j;         // Right cell column
    
    // Additional bounds checks (defensive programming)
    if (ci < 0 || ci >= alpha.rows()) return 0.0;
    if (cj1 < 0 || cj1 >= alpha.cols()) return 0.0;
    if (cj2 < 0 || cj2 >= alpha.cols()) return alpha(ci, cj1);
    
    // Return arithmetic average of left and right cell alphas
    return 0.5 * (alpha(ci, cj1) + alpha(ci, cj2));
}

// Get Brinkman alpha at V-velocity location (i,j)
// Returns the average of alpha from the two cells sharing this v-face
inline double alphaAtV(const SIMPLE& s, int i, int j) {
    const auto& alpha = s.alpha;
    const int M = s.M;
    const int N = s.N;
    
    // Boundary check
    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 0.0;
    
    // Cell indices for the two cells sharing this v-face
    int ci1 = i - 1;     // Top cell row
    int ci2 = i;         // Bottom cell row
    int cj = j - 1;      // Column index
    
    // Additional bounds checks
    if (cj < 0 || cj >= alpha.cols()) return 0.0;
    if (ci1 < 0 || ci1 >= alpha.rows()) return 0.0;
    if (ci2 < 0 || ci2 >= alpha.rows()) return alpha(ci1, cj);
    
    // Return arithmetic average of top and bottom cell alphas
    return 0.5 * (alpha(ci1, cj) + alpha(ci2, cj));
}

// -----------------------------------------------------------------------------
// Gamma (Density/Porosity) Interpolation
// -----------------------------------------------------------------------------
// Same logic as alpha interpolation, but for the gamma (porosity) field.
// Gamma is used in density-based topology optimization:
//   gamma = 1.0 → fluid (full permeability)
//   gamma = 0.0 → solid (zero permeability)
//   0 < gamma < 1 → transition zone

// Get porosity gamma at U-velocity location (i,j)
// Returns the average of gamma from the two cells sharing this u-face
// Default to 1.0 (fluid) at boundaries for stability
inline double gammaAtU(const SIMPLE& s, int i, int j) {
    const auto& gamma = s.gamma;
    const int M = s.M;
    const int N = s.N;
    
    // Default to fluid at boundaries (conservative for flow)
    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 1.0;
    
    int ci = i - 1;
    int cj1 = j - 1;
    int cj2 = j;
    
    if (ci < 0 || ci >= gamma.rows()) return 1.0;
    if (cj1 < 0 || cj1 >= gamma.cols()) return 1.0;
    if (cj2 < 0 || cj2 >= gamma.cols()) return gamma(ci, cj1);
    
    return 0.5 * (gamma(ci, cj1) + gamma(ci, cj2));
}

// Get porosity gamma at V-velocity location (i,j)
// Returns the average of gamma from the two cells sharing this v-face
inline double gammaAtV(const SIMPLE& s, int i, int j) {
    const auto& gamma = s.gamma;
    const int M = s.M;
    const int N = s.N;
    
    // Default to fluid at boundaries
    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 1.0;
    
    int ci1 = i - 1;
    int ci2 = i;
    int cj = j - 1;
    
    if (cj < 0 || cj >= gamma.cols()) return 1.0;
    if (ci1 < 0 || ci1 >= gamma.rows()) return 1.0;
    if (ci2 < 0 || ci2 >= gamma.rows()) return gamma(ci1, cj);
    
    return 0.5 * (gamma(ci1, cj) + gamma(ci2, cj));
}

// -----------------------------------------------------------------------------
// Second-Order Upwind (SOU) Deferred Correction Functions
// -----------------------------------------------------------------------------
// These compute the deferred correction source term for second-order upwind.
// The correction is: Sdc = Σ_face [ F * (φ_SOU - φ_FOU) ]
// This adds higher-order accuracy while maintaining the stable FOU matrix.
// Implemented in convection.cpp.

// Compute SOU deferred correction for U-momentum at location (i,j)
// Fe, Fw, Fn, Fs are the mass flux rates at east/west/north/south faces
double computeSOUCorrectionU(const SIMPLE& s, int i, int j,
                             double Fe, double Fw, double Fn, double Fs);

// Compute SOU deferred correction for V-momentum at location (i,j)
double computeSOUCorrectionV(const SIMPLE& s, int i, int j,
                             double Fe, double Fw, double Fn, double Fs);

// -----------------------------------------------------------------------------
// Pseudo-Time Statistics Logging
// -----------------------------------------------------------------------------
// Prints min/max/avg pseudo-Δt statistics for debugging CFL-based time stepping.
// Called once per iteration if logPseudoDtStats is enabled.

void logPseudoStats(const SIMPLE& solver, const char* label, const SIMPLE::PseudoDtStats& stats);

// =============================================================================
// POST-PROCESSING: Pressure Sampling at Physical Locations
// =============================================================================
// These structures and functions allow sampling pressure and velocity at
// arbitrary physical x-coordinates (not just cell boundaries). This enables
// accurate pressure-drop measurements independent of mesh alignment.
//
// Implemented in postprocessing.cpp.

// Structure to hold metrics sampled at a single vertical plane (constant x)
struct PlaneMetrics {
    double flowArea   = 0.0;  // Total open (fluid) cross-sectional area [m²/m depth]
    double massFlux   = 0.0;  // Total mass flow rate through plane [kg/s per m depth]
    double avgStatic  = 0.0;  // Area-weighted average STATIC pressure [Pa]
    double avgDynamic = 0.0;  // Area-weighted average DYNAMIC pressure [Pa] = ½ρV²
    double avgTotal   = 0.0;  // Mass-weighted average TOTAL pressure [Pa] = static + dynamic
    bool   valid      = false;// True if at least one fluid cell was found at this x
};

// Sample pressure and velocity fields at a given physical x-coordinate [m]
// Uses linear interpolation between adjacent cell columns.
// Skips rows where either adjacent cell is solid (conservative for accuracy).
PlaneMetrics samplePlaneAtX(const SIMPLE& solver, double xPhysical);

// Debug helper: print detailed plane metrics to console
void printPlaneInfo(const char* name, double xPhysical, const PlaneMetrics& m);

