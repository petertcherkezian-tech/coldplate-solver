// ============================================================================
// File: Utilities/output.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Handles all file I/O and data export for the CFD solver, including:
//   - Per-iteration console output and log file writing
//   - Final solution field export (u, v, p as text and VTK)
//   - Thermal domain cropping for coupled thermal analysis
//
// OUTPUT FILES:
//   1. ExportFiles/residuals.txt: Convergence history log
//      Format: Iter Mass U V Core_dP Full_dP
//
//   2. ExportFiles/pressure_drop_history.txt: Pressure drop evolution
//      Format: Iter Core_Total Full_Total Core_Static Full_Static
//
//   3. ExportFiles/u.txt, v.txt, p.txt: Raw staggered fields (for restart)
//
//   4. ExportFiles/u_full.txt, v_full.txt: Cell-centered velocity fields
//
//   5. ExportFiles/u_thermal.txt, v_thermal.txt, pressure_thermal.txt:
//      Cropped to thermal domain (heatsink region only, excluding buffers)
//
//   6. ExportFiles/fluid_results.vtk: Visualization file for ParaView
//      Contains: pressure, velocity vectors, cellType, gamma (density), alpha
//
// THERMAL CROPPING:
//   The CFD domain may include inlet/outlet buffer zones for flow development.
//   For thermal analysis, we crop to just the heatsink region:
//   - Skip N_in_buffer columns from left (inlet buffer)
//   - Skip N_out_buffer columns from right (outlet buffer)
//   - Export only the central N_thermal = N - N_in_buffer - N_out_buffer columns
//
// VTK FORMAT:
//   Uses Legacy VTK ASCII format (STRUCTURED_POINTS) for compatibility.
//   Cell data includes scalar fields and velocity vectors.
//
// ============================================================================
#include "SIMPLE.h"
#include <iomanip>

// ============================================================================
// saveMatrix: Write a 2D Eigen matrix to a text file
// ============================================================================
void SIMPLE::saveMatrix(Eigen::MatrixXd inputMatrix, std::string fileName)
{
    std::string fullPath = "ExportFiles/" + fileName + ".txt";
    std::ofstream out(fullPath);
    if (!out.is_open()) {
        std::cerr << "Error: could not open file '" << fullPath << "' for writing.\n";
        return;
    }

    int rows = static_cast<int>(inputMatrix.rows());
    int cols = static_cast<int>(inputMatrix.cols());

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << inputMatrix(i, j);
            if (j < cols - 1) {
                out << "\t";
            }
        }
        out << "\n";
    }
}

// ------------------------------------------------------------------
// Main Save Function (Includes Thermal Slicing & VTK)
// ------------------------------------------------------------------
void SIMPLE::saveAll()
{
    // Optionally save raw staggered fields for restart
    saveMatrix(u, "u");
    saveMatrix(v, "v");
    saveMatrix(p, "p");

    // 1. Interpolate velocities to cell centers (Required for VTK & Thermal)
    // Zero out velocity in solid cells (cellType == 1) for correct visualization
    Eigen::MatrixXd uCenter = Eigen::MatrixXd::Zero(M, N);
    Eigen::MatrixXd vCenter = Eigen::MatrixXd::Zero(M, N);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (cellType(i, j) < 0.99999) {  // Only interpolate for permeable cells
                uCenter(i, j) = 0.5 * (u(i, j) + u(i+1, j));
                vCenter(i, j) = 0.5 * (v(i, j) + v(i, j+1));
            }
            // Solid cells remain zero (initialized above)
        }
    }

    // 2. Save Full Domain Data (Text format)
    saveMatrix(uCenter, "u_full");
    saveMatrix(vCenter, "v_full");
    saveMatrix(p, "pressure_full");
    
    // ---------------------------------------------------------
    // 3. CALCULATE PRESSURE GRADIENT (FULL DOMAIN)
    // ---------------------------------------------------------
    Eigen::MatrixXd pGradX = Eigen::MatrixXd::Zero(M, N);  // dp/dx
    Eigen::MatrixXd pGradY = Eigen::MatrixXd::Zero(M, N);  // dp/dy
    Eigen::MatrixXd pGradMag = Eigen::MatrixXd::Zero(M, N); // |∇p|
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // Only compute gradient for permeable cells
            if (cellType(i, j) < 0.99999) {
                // dp/dx using central difference where possible
                if (j > 0 && j < N - 1) {
                    pGradX(i, j) = (p(i, j + 1) - p(i, j - 1)) / (2.0 * hx);
                } else if (j == 0) {
                    pGradX(i, j) = (p(i, j + 1) - p(i, j)) / hx;  // Forward diff
                } else {
                    pGradX(i, j) = (p(i, j) - p(i, j - 1)) / hx;  // Backward diff
                }
                
                // dp/dy using central difference where possible
                if (i > 0 && i < M - 1) {
                    pGradY(i, j) = (p(i + 1, j) - p(i - 1, j)) / (2.0 * hy);
                } else if (i == 0) {
                    pGradY(i, j) = (p(i + 1, j) - p(i, j)) / hy;  // Forward diff
                } else {
                    pGradY(i, j) = (p(i, j) - p(i - 1, j)) / hy;  // Backward diff
                }
                
                pGradMag(i, j) = std::sqrt(pGradX(i, j) * pGradX(i, j) + 
                                           pGradY(i, j) * pGradY(i, j));
            }
        }
    }

    // ---------------------------------------------------------
    // 4. EXPORT THERMAL DATA (CROPPED)
    // ---------------------------------------------------------
    int N_thermal = N - N_in_buffer - N_out_buffer;
    
    if (N_thermal <= 0) {
        std::cerr << "Error: Thermal domain size is <= 0. Check buffer sizes." << std::endl;
    } else {
        Eigen::MatrixXd uThermal = Eigen::MatrixXd::Zero(M, N_thermal);
        Eigen::MatrixXd vThermal = Eigen::MatrixXd::Zero(M, N_thermal);
        Eigen::MatrixXd pThermal = Eigen::MatrixXd::Zero(M, N_thermal);
        Eigen::MatrixXd pGradThermal = Eigen::MatrixXd::Zero(M, N_thermal);
        
        // Slice the matrix: Skip 'N_in_buffer' columns
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N_thermal; ++j) {
                int src_j = j + N_in_buffer;
                uThermal(i, j) = uCenter(i, src_j);
                vThermal(i, j) = vCenter(i, src_j);
                pThermal(i, j) = p(i, src_j);
                pGradThermal(i, j) = pGradMag(i, src_j);
            }
        }
        saveMatrix(uThermal, "u_thermal");
        saveMatrix(vThermal, "v_thermal");
        saveMatrix(pThermal, "pressure_thermal");
        saveMatrix(pGradThermal, "p_gradient");
    }

    // ---------------------------------------------------------
    // 5. EXPORT FLUID VTK (FULL DOMAIN)
    // ---------------------------------------------------------
    std::string vtkFile = "ExportFiles/fluid_results.vtk";
    std::ofstream vtk(vtkFile);
    
    if (vtk.is_open()) {
        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "SIMPLE CFD Results\n";
        vtk << "ASCII\n";
        vtk << "DATASET STRUCTURED_POINTS\n";
        vtk << "DIMENSIONS " << N << " " << M << " 1\n"; 
        vtk << "ORIGIN 0 0 0\n";
        vtk << "SPACING " << hx << " " << hy << " 1\n";
        vtk << "POINT_DATA " << N * M << "\n";

        // Pressure
        vtk << "SCALARS pressure double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < M; ++i) {     
            for (int j = 0; j < N; ++j) { 
                vtk << p(i, j) << "\n";
            }
        }

        // Velocity Vectors
        vtk << "VECTORS velocity double\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                vtk << uCenter(i, j) << " " << vCenter(i, j) << " 0.0\n";
            }
        }

        // Cell Type (Geometry) - continuous values 0=fluid, 1=solid
        vtk << "SCALARS cellType double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                vtk << cellType(i, j) << "\n";
            }
        }
        
        // Density field (gamma): 1=fluid, 0=solid, intermediate=buffer
        vtk << "SCALARS Density double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                vtk << gamma(i, j) << "\n";
            }
        }
        
        // Brinkman alpha field (penalization strength)
        vtk << "SCALARS Alpha double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                vtk << alpha(i, j) << "\n";
            }
        }
        
        // Pressure Gradient Magnitude
        vtk << "SCALARS PressureGradient double 1\n";
        vtk << "LOOKUP_TABLE default\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                vtk << pGradMag(i, j) << "\n";
            }
        }
        
        // Pressure Gradient Vector
        vtk << "VECTORS pressure_gradient double\n";
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                vtk << pGradX(i, j) << " " << pGradY(i, j) << " 0.0\n";
            }
        }
        
        std::cout << "Saved VTK to " << vtkFile << std::endl;
    }

    std::cout << "Data saved to ExportFiles/" << std::endl;
}

void SIMPLE::initLogFiles(std::ofstream& residFile, std::ofstream& dpFile) {
    residFile.open("ExportFiles/residuals.txt");
    dpFile.open("ExportFiles/pressure_drop_history.txt");
    residFile << "Iter MassResid UResid VResid Core_dP_AfterInletBuffer(Pa) "
              << "Full_dP_FullSystem(Pa)" << std::endl;
    dpFile << "Iter Core_Total(Pa) Full_Total(Pa) Core_Static(Pa) Full_Static(Pa)" << std::endl;
    printIterationHeader();
}

void SIMPLE::printIterationHeader() const {
    std::cout << "Starting simulation..." << std::endl;
    std::cout << std::setw(8) << "Iter"
              << std::setw(14) << "Mass"
              << std::setw(14) << "U-vel"
              << std::setw(14) << "V-vel"
              << std::setw(14) << "TransRes"
              << std::setw(16) << "Core dP (Pa)"
              << std::setw(16) << "Full dP (Pa)"
              << std::setw(10) << "Time (ms)"
              << std::setw(6) << "P-It" << std::endl;
    std::cout << std::string(122, '-') << std::endl;
}

void SIMPLE::writeIterationLogs(std::ofstream& residFile,
                                std::ofstream& dpFile,
                                int iter,
                                double corePressureDrop,
                                double fullPressureDrop,
                                double coreStaticDrop,
                                double fullStaticDrop) {
    residFile << iter << " "
              << residMass << " "
              << residU << " "
              << residV << " "
              << corePressureDrop << " "
              << fullPressureDrop << std::endl;

    dpFile << iter << " "
           << corePressureDrop << " "
           << fullPressureDrop << " "
           << coreStaticDrop << " "
           << fullStaticDrop << std::endl;
}

void SIMPLE::printIterationRow(int iter,
                               double residMassVal,
                               double residUVal,
                               double residVVal,
                               double maxTransRes,
                               double corePressureDrop,
                               double fullPressureDrop,
                               double iterTimeMs,
                               int pressureIterations) const {
    std::cout << std::setw(8) << iter
              << std::setw(14) << std::scientific << std::setprecision(3) << residMassVal
              << std::setw(14) << residUVal
              << std::setw(14) << residVVal
              << std::setw(14) << maxTransRes
              << std::setw(16) << std::fixed << std::setprecision(1) << corePressureDrop
              << std::setw(16) << fullPressureDrop
              << std::setw(10) << std::fixed << std::setprecision(1) << iterTimeMs
              << std::setw(6) << pressureIterations
              << std::endl;
}

void SIMPLE::printStaticDp(int iter,
                           double coreStaticDrop,
                           double fullStaticDrop) const {
    std::cout << "         Static dP (Core/Full): " 
              << std::setw(12) << std::fixed << std::setprecision(1) << coreStaticDrop << " / "
              << std::setw(12) << fullStaticDrop << " Pa"
              << std::endl;
}

void SIMPLE::paintBoundaries() {
    Eigen::MatrixXd BCs = Eigen::MatrixXd::Zero(M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            BCs(i, j) = checkBoundaries(i, j);
        }
    }
    saveMatrix(BCs, "BC");
}