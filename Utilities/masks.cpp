// ============================================================================
// File: Utilities/masks.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Precomputes boolean fluid/solid masks for all staggered grid locations.
//   These masks enable fast lookup during momentum assembly to skip solid cells
//   without expensive geometric checks each iteration.
//
// MASK COMPUTATION:
//   
//   cellType conventions:
//     cellType = 0       → pure fluid
//     0 < cellType < 1   → porous (Brinkman penalization, must be solved)
//     cellType = 1       → true solid wall, impermeable
//
//   PERMEABILITY is the single criterion: cellType < 1 means permeable.
//
//   FACE MASKS (isFluidU, isFluidV):
//     A face is active ONLY if BOTH adjacent cells are permeable.
//     This enforces no-penetration at solid walls while allowing flow through
//     porous regions. Brinkman drag handles resistance in porous cells.
//
//   PRESSURE MASK (isFluidP):
//     A cell is active if and only if it is permeable (cellType < 1).
//     Solid cells have no pressure DOF - wall BCs are handled by the pressure
//     solver setting coupling coefficients to zero for solid neighbors.
//
// STORAGE:
//   Masks are stored as 1D std::vector<bool> for cache efficiency.
//   Linear index = i * cols + j
//
// ============================================================================
#include "SIMPLE.h"
#include <iostream>

// ============================================================================
// buildFluidMasks: Precompute fluid/solid masks for all grid locations
// ============================================================================
void SIMPLE::buildFluidMasks() {
    isFluidU.assign((M + 1) * N, false);
    isFluidV.assign(M * (N + 1), false);
    isFluidP.assign((M + 1) * (N + 1), false);

    const int pCols = N + 1;
    const int uCols = N;
    const int vCols = N + 1;

    // -------------------------------------------------------------------------
    // Helper: Check if a cell (0-based ci, cj) is permeable (fluid or porous)
    // -------------------------------------------------------------------------
    auto isPermeable = [&](int ci, int cj) -> bool {
        if (ci < 0 || ci >= M || cj < 0 || cj >= N) return false;
        return cellType(ci, cj) < 0.99999; // cellType < 1
    };

    // -------------------------------------------------------------------------
    // Step 1: Build PRESSURE mask (simplest: permeable cells only)
    // -------------------------------------------------------------------------
    int solidCount = 0;
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            int ci = i - 1;
            int cj = j - 1;
            bool permeable = isPermeable(ci, cj);
            isFluidP[i * pCols + j] = permeable;
            if (!permeable) solidCount++;
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: Build FACE masks (based on permeability of adjacent cells)
    // -------------------------------------------------------------------------
    // A face is active only if BOTH adjacent cells are permeable.
    // This enforces no-penetration at solid walls.
    // -------------------------------------------------------------------------

    // U-faces: vertical faces between cells
    // u(i,j) with i in [1,M], j in [0,N-1]
    // For interior faces: u(i,j) sits between cells (i-1, j-1) and (i-1, j) in 0-based
    for (int i = 1; i <= M; ++i) {
        for (int j = 0; j < N; ++j) {
            int ci = i - 1;  // 0-based row
            
            if (j == 0) {
                // Inlet boundary: active if first interior cell is permeable
                isFluidU[i * uCols + j] = isPermeable(ci, 0);
            } else {
                // Interior/outlet: between cells (ci, j-1) and (ci, j)
                isFluidU[i * uCols + j] = isPermeable(ci, j - 1) && isPermeable(ci, j);
            }
        }
    }

    // V-faces: horizontal faces between cells
    // v(i,j) with i in [0,M-1], j in [1,N]
    // For interior faces: v(i,j) sits between cells (i-1, j-1) and (i, j-1) in 0-based
    for (int i = 0; i < M; ++i) {
        for (int j = 1; j <= N; ++j) {
            int cj = j - 1;  // 0-based column
            
            if (i == 0) {
                // Top boundary: no flux through domain wall
                isFluidV[i * vCols + j] = false;
            } else {
                // Interior: between cells (i-1, cj) and (i, cj)
                isFluidV[i * vCols + j] = isPermeable(i - 1, cj) && isPermeable(i, cj);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------
    int uC = 0, vC = 0, pC = 0;
    for (auto b : isFluidU) uC += b;
    for (auto b : isFluidV) vC += b;
    for (auto b : isFluidP) pC += b;

    std::cout << "Masks: U=" << uC << " V=" << vC << " P=" << pC 
              << " | Solid cells: " << solidCount << std::endl;
}

