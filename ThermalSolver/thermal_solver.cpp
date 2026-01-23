// ============================================================================
// File: ThermalSolver/thermal_solver.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   3D Conjugate Heat Transfer (CHT) solver using Finite Volume Method (FVM)
//   with AMG-preconditioned IDR(s) iterative solver (AMGCL library).
//
// PHYSICS:
//   Solves the steady-state energy equation:
//     ρ * Cp * (u·∇T) = ∇·(k∇T) + Q
//
//   where:
//   - T = temperature [°C]
//   - u = velocity field (from CFD solver) [m/s]
//   - k = thermal conductivity [W/m-K]
//   - ρ = density [kg/m³]
//   - Cp = specific heat capacity [J/kg-K]
//   - Q = heat source [W/m³]
//
// MESH STRUCTURE:
//   3D structured grid with non-uniform z-spacing:
//   - Bottom (z < 0): Solid base plate (nz_solid layers, k = k_solid)
//   - Top (z > 0): Fluid/fin region (nz_fluid layers)
//     - Fluid cells: k = k_fluid, with velocity-driven advection
//     - Solid cells: k = k_solid, conduction only
//
// DENSITY-BASED APPROACH:
//   Uses continuous gamma field from geometry generator:
//   - gamma = 1.0 → pure fluid
//   - gamma = 0.0 → pure solid
//   - 0 < gamma < 1 → buffer zone (linear interpolation of properties)
//
//   Thermal conductivity interpolation:
//     k(gamma) = k_solid * (1 - gamma) + k_fluid * gamma
//
// VELOCITY PROFILE:
//   For 2.5D approximation, applies parabolic z-profile to 2D velocity:
//     u_3D(x,y,z) = u_2D(x,y) * 4*z*(1-z)  where z normalized to [0,1]
//
// SOLVER:
//   - AMGCL library with IDR(s) Krylov method (s=8 shadow vectors)
//   - AMG preconditioner with smoothed aggregation coarsening
//   - SPAI0 or ILU0 relaxation
//   - Sparse matrix in CSR format
//
// BOUNDARY CONDITIONS:
//   - Bottom (z=0): Heat flux Q = q * Area (from input file)
//   - Top (z=H): Adiabatic (no heat flux)
//   - Inlet (x=0): Fixed temperature T = T_inlet
//   - Outlet (x=L): Zero gradient (convective outflow)
//   - Y-boundaries: Adiabatic (symmetry or walls)
//
// OUTPUT:
//   - thermal_results_3d.vtk: VTK file with temperature, velocity, region, etc.
//   - Console: Temperature statistics and energy balance
//
// COMPILE:
//   g++ -std=c++17 -fopenmp -DAMGCL_NO_BOOST -O3 thermal_solver.cpp \
//       -Iamgcl -o thermal_solver.exe
//
// ============================================================================

 #include <iostream>
 #include <fstream>
 #include <sstream>
 #include <vector>
 #include <string>
 #include <cmath>
 #include <chrono>
 #include <algorithm>
 #include <tuple>
 #include <memory>
 #include <iomanip>
 
 #include <amgcl/make_solver.hpp>
 #include <amgcl/solver/bicgstab.hpp>
 #include <amgcl/amg.hpp>
 #include <amgcl/coarsening/smoothed_aggregation.hpp>
 #include <amgcl/relaxation/spai0.hpp>
 #include <amgcl/relaxation/ilu0.hpp>
 #include <amgcl/solver/idrs.hpp>
 #include <amgcl/backend/builtin.hpp>
 
 using namespace std;
 
 struct SparseMatrix {
     int n = 0;
     vector<int> row_ptr, col_idx;
     vector<double> values;
     
     void from_coo(int size, vector<tuple<int,int,double>>& coo) {
         n = size;
         sort(coo.begin(), coo.end());
         
         row_ptr.assign(n + 1, 0);
         for (auto& [r, c, v] : coo) row_ptr[r + 1]++;
         for (int i = 1; i <= n; i++) row_ptr[i] += row_ptr[i-1];
         
         col_idx.resize(coo.size());
         values.resize(coo.size());
         vector<int> pos(n, 0);
         
         for (auto& [r, c, v] : coo) {
             int idx = row_ptr[r] + pos[r];
             col_idx[idx] = c;
             values[idx] = v;
             pos[r]++;
         }
         
         vector<int> new_col; vector<double> new_val;
         vector<int> new_ptr(n + 1, 0);
         
         for (int r = 0; r < n; r++) {
             new_ptr[r] = new_col.size();
             for (int i = row_ptr[r]; i < row_ptr[r+1]; i++) {
                 if (!new_col.empty() && new_ptr[r] < (int)new_col.size() && new_col.back() == col_idx[i]) {
                     new_val.back() += values[i];
                 } else {
                     new_col.push_back(col_idx[i]);
                     new_val.push_back(values[i]);
                 }
             }
         }
         new_ptr[n] = new_col.size();
         
         row_ptr = move(new_ptr);
         col_idx = move(new_col);
         values = move(new_val);
     }
     size_t nnz() const { return values.size(); }
 };
 
 struct Params {
     int Nx = 0, Ny = 0, nz_solid = 2, nz_fluid = 7;
     double Lx = 0, Ly = 0, H_b = 0.0005, H_t = 0.004;
    double k_s = 400.0, k_f = 0.6, rho = 998.0, Cp = 4180.0, T_inlet = 30.0;
    double rtol = 1e-6; int maxiter = 1000;
    // Conductivity interpolation: 0=linear, 1=RAMP (for TO)
    int k_interp_mode = 0;  // 0=linear (stable), 1=RAMP (paper Eq.13 for TO)
    double qk = 1.0;  // RAMP penalization parameter (paper uses [1,3,10,30] sequence)
    
    double dx() const { return Lx / Nx; }
     double dy() const { return Ly / Ny; }
     double dz_s() const { return H_b / nz_solid; }
     double dz_f() const { return H_t / nz_fluid; }
     int Nz() const { return nz_solid + nz_fluid; }
     int n_dofs() const { return Nx * Ny * Nz(); }
 };
 
 vector<vector<double>> read_2d(const string& path, int& rows, int& cols) {
     ifstream f(path); if (!f) { cerr << "Cannot open: " << path << endl; return {}; }
     vector<vector<double>> data; string line;
     while (getline(f, line)) {
         if (line.empty() || line[0] == '#') continue;
         istringstream iss(line); vector<double> row; double v;
         while (iss >> v) row.push_back(v);
         if (!row.empty()) data.push_back(row);
     }
     rows = data.size(); cols = rows > 0 ? data[0].size() : 0;
     return data;
 }
 
 class ThermalSolver3D {
 public:
     Params p;
     vector<double> gamma, u_xy, v_xy, q0, dz_cells, T;
     vector<double> U3d, V3d, W3d;
     
     ThermalSolver3D(const Params& params) : p(params) {
         dz_cells.resize(p.Nz());
         for (int k = 0; k < p.nz_solid; k++) dz_cells[k] = p.dz_s();
         for (int k = 0; k < p.nz_fluid; k++) dz_cells[p.nz_solid + k] = p.dz_f();
     }
     
     int idx(int i, int j, int k) const { return k * p.Nx * p.Ny + j * p.Nx + i; }
     double harm(double a, double b) const { return 2*a*b/(a+b+1e-30); }
     
     // Region: 0=solid (base or fin), 1=fluid
    // Buffer zones (0 < gamma < 1) treated as fluid for advection
    int region(int i, int j, int k) const { 
        if (k < p.nz_solid) return 0;  // Base plate is always solid
        double g = gamma[j*p.Nx+i];
        return (g > 0.01) ? 1 : 0;  // Any gamma > 0 means fluid (includes buffers)
    }
     
     // Thermal conductivity with configurable interpolation
   // Mode 0 (Linear): k = k_s * (1-gamma) + k_f * gamma
   // Mode 1 (RAMP): k = k_f + (k_s - k_f) * (1-gamma) / (1 + qk*gamma)  [Paper Eq. 13]
   double kval(int i, int j, int k) const { 
       if (k < p.nz_solid) return p.k_s;  // Base plate always solid conductivity
       double g = gamma[j*p.Nx+i];
       
       if (p.k_interp_mode == 1) {
           // RAMP interpolation (Zeng & Lee paper, Eq. 13)
           // For optimization: penalizes intermediate gamma toward extremes
           return p.k_f + (p.k_s - p.k_f) * (1.0 - g) / (1.0 + p.qk * g);
       } else {
           // Linear interpolation (default, current best for buffer zones)
           return p.k_s * (1.0 - g) + p.k_f * g;
       }
   }
    double pois(int k) const {
        if (k < p.nz_solid) return 0;
        double z = (k - p.nz_solid + 0.5) * p.dz_f() / p.H_t;
        return 4.0 * z * (1 - z);  // Max = 1.0 at centerline (for max velocity input)
    }
     
    void solve() {
         int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz(), n = p.n_dofs();
         double dx = p.dx(), dy = p.dy(), rhoCp = p.rho * p.Cp, T_in = p.T_inlet;
         double Az = dx * dy;
         
        // Initial T
        T.assign(n, T_in);

       // --- SINGLE-PASS MULTICORE SOLVER (V3 Pure Architecture + IDR(s)) ---
        cout << "  Assembling matrix (Multicore)..." << flush;
        auto t0 = chrono::high_resolution_clock::now();
        
        vector<tuple<int,int,double>> global_coo;
        vector<double> b(n, 0.0);

        #pragma omp parallel
        {
            vector<tuple<int,int,double>> local_coo;
            
            #pragma omp for schedule(static)
            for (int k = 0; k < Nz; k++) {
                double dz_k = dz_cells[k], Ax = dy * dz_k, Ay = dx * dz_k, Az_cell = dx * dy;
                for (int j = 0; j < Ny; j++) {
                    for (int i = 0; i < Nx; i++) {
                        int P = idx(i, j, k);
                        double diag = 0, kP = kval(i,j,k), po = pois(k);
                        double uP = (region(i,j,k) == 1) ? u_xy[j*Nx+i]*po : 0;
                        double vP = (region(i,j,k) == 1) ? v_xy[j*Nx+i]*po : 0;
                        
                        // ========== X FACES (Upwind) ==========
                        if (i > 0) {
                            double kf = harm(kP, kval(i-1,j,k)), D = kf*Ax/dx;
                            diag += D; local_coo.push_back({P, idx(i-1,j,k), -D});
                            if (region(i,j,k)==1 && region(i-1,j,k)==1) {
                                double uW = u_xy[j*Nx+i-1]*po, F = rhoCp * 0.5*(uP + uW) * Ax;
                                diag += max(-F, 0.0); local_coo.push_back({P, idx(i-1,j,k), -max(F, 0.0)});
                            }
                        } else if (region(i,j,k)==1) {
                            double D_bc = 2.0*kP*Ax/dx, F_bc = rhoCp*uP*Ax;
                            diag += D_bc; b[P] += D_bc * T_in;
                            if (F_bc > 0) b[P] += F_bc * T_in; else diag += -F_bc;
                        }
                        if (i < Nx-1) {
                            double kf = harm(kP, kval(i+1,j,k)), D = kf*Ax/dx;
                            diag += D; local_coo.push_back({P, idx(i+1,j,k), -D});
                            if (region(i,j,k)==1 && region(i+1,j,k)==1) {
                                double uE = u_xy[j*Nx+i+1]*po, F = rhoCp * 0.5*(uP + uE) * Ax;
                                diag += max(F, 0.0); local_coo.push_back({P, idx(i+1,j,k), -max(-F, 0.0)});
                            }
                        } else if (region(i,j,k)==1 && uP > 0) diag += rhoCp * uP * Ax;

                        // ========== Y FACES (Upwind) ==========
                        if (j > 0) {
                            double kf = harm(kP, kval(i,j-1,k)), D = kf*Ay/dy;
                            diag += D; local_coo.push_back({P, idx(i,j-1,k), -D});
                            if (region(i,j,k)==1 && region(i,j-1,k)==1) {
                                double vS = v_xy[(j-1)*Nx+i]*po, F = rhoCp * 0.5*(vP + vS) * Ay;
                                diag += max(-F, 0.0); local_coo.push_back({P, idx(i,j-1,k), -max(F, 0.0)});
                            }
                        }
                        if (j < Ny-1) {
                            double kf = harm(kP, kval(i,j+1,k)), D = kf*Ay/dy;
                            diag += D; local_coo.push_back({P, idx(i,j+1,k), -D});
                            if (region(i,j,k)==1 && region(i,j+1,k)==1) {
                                double vN = v_xy[(j+1)*Nx+i]*po, F = rhoCp * 0.5*(vP + vN) * Ay;
                                diag += max(F, 0.0); local_coo.push_back({P, idx(i,j+1,k), -max(-F, 0.0)});
                            }
                        } else if (region(i,j,k)==1 && vP > 0) diag += rhoCp * vP * Ay;

                        // ========== Z FACES (Diffusion Only) ==========
                        if (k > 0) {
                            double kN = kval(i,j,k-1), dz_P = dz_cells[k], dz_N = dz_cells[k-1];
                            double kf = (dz_P+dz_N)/(dz_P/kP + dz_N/kN);
                            double D = kf * Az_cell / (0.5*(dz_P+dz_N));
                            diag += D; local_coo.push_back({P, idx(i,j,k-1), -D});
                        } else b[P] += q0[j*Nx+i] * Az_cell;
                        if (k < Nz-1) {
                            double kN = kval(i,j,k+1), dz_P = dz_cells[k], dz_N = dz_cells[k+1];
                            double kf = (dz_P+dz_N)/(dz_P/kP + dz_N/kN);
                            double D = kf * Az_cell / (0.5*(dz_P+dz_N));
                            diag += D; local_coo.push_back({P, idx(i,j,k+1), -D});
                        }
                        local_coo.push_back({P, P, diag + 1e-12}); 
                    }
                }
            }
            #pragma omp critical
            global_coo.insert(global_coo.end(), local_coo.begin(), local_coo.end());
        }
        
        SparseMatrix A; A.from_coo(n, global_coo);
        cout << " done (" << chrono::duration<double>(chrono::high_resolution_clock::now()-t0).count() << "s)" << endl;

        // --- STABLE MULTICORE SOLVER (IDR(s) + SPAI0) ---
        typedef amgcl::backend::builtin<double> Backend;
        typedef amgcl::make_solver<
            amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
            amgcl::solver::idrs<Backend>
        > Solver;
        Solver::params prm;
        prm.solver.s = 8; // High shadow space for stability
        prm.solver.smoothing = true; // Residual smoothing for monotonicity
        prm.solver.tol = p.rtol; prm.solver.maxiter = p.maxiter;

        cout << "  Building Parallel Preconditioner (once)..." << flush;
        auto t1 = chrono::high_resolution_clock::now();
        auto Am = make_shared<amgcl::backend::crs<double>>(n, n, A.row_ptr, A.col_idx, A.values);
        Solver solver(Am, prm);
        cout << " " << chrono::duration<double>(chrono::high_resolution_clock::now()-t1).count() << "s" << endl;

        cout << "  Solving Single Pass (IDR(s), s=8, All Cores)..." << flush;
        auto t2 = chrono::high_resolution_clock::now();
        auto [iters, res] = solver(b, T);
        cout << " " << iters << " iters, res=" << scientific << res << " (" << fixed << setprecision(2) << chrono::duration<double>(chrono::high_resolution_clock::now()-t2).count() << "s)" << endl;
         // Build 3D velocity field for VTK output
         U3d.resize(n, 0); V3d.resize(n, 0); W3d.resize(n, 0);
         for (int k = 0; k < Nz; k++) {
             double po = pois(k);
             for (int j = 0; j < Ny; j++) {
                 for (int i = 0; i < Nx; i++) {
                     int P = idx(i, j, k);
                     if (region(i, j, k) == 1) {
                         U3d[P] = u_xy[j*Nx+i] * po;
                         V3d[P] = v_xy[j*Nx+i] * po;
                     }
                 }
             }
         }
         
         // Compute temperature statistics
         double Tbase_min=1e30, Tbase_max=-1e30;
         double Tfin_min=1e30, Tfin_max=-1e30;
         double Tfl_min=1e30, Tfl_max=-1e30;
         
         for (int k = 0; k < Nz; k++)
             for (int j = 0; j < Ny; j++)
                 for (int i = 0; i < Nx; i++) {
                     double t = T[idx(i,j,k)];
                     if (k < p.nz_solid) { 
                         Tbase_min = min(Tbase_min,t); Tbase_max = max(Tbase_max,t); 
                     } else if (region(i,j,k) == 1) {
                         Tfl_min = min(Tfl_min,t); Tfl_max = max(Tfl_max,t);
                     } else {
                         Tfin_min = min(Tfin_min,t); Tfin_max = max(Tfin_max,t);
                     }
                 }
         
         cout << "\nResults:" << endl;
         cout << "  T_base:  [" << fixed << setprecision(2) << Tbase_min << ", " << Tbase_max << "] C" << endl;
         cout << "  T_fin:   [" << Tfin_min << ", " << Tfin_max << "] C" << endl;
         cout << "  T_fluid: [" << Tfl_min << ", " << Tfl_max << "] C" << endl;
         
         // Energy balance
         double Q_in = 0, mdot = 0, mdotT = 0;
         for (int j = 0; j < Ny; j++) 
             for (int i = 0; i < Nx; i++) 
                 Q_in += q0[j*Nx+i] * dx * dy;
         
         for (int k = p.nz_solid; k < Nz; k++) {
             double A = dy * dz_cells[k], po = pois(k);
             for (int j = 0; j < Ny; j++) {
                 int i = Nx - 1;
                 if (region(i,j,k) == 1) {
                     double u = u_xy[j*Nx+i] * po;
                     if (u > 0) { 
                         double m = p.rho * u * A; 
                         mdot += m; 
                         mdotT += m * T[idx(i,j,k)]; 
                     }
                 }
             }
         }
         
         double T_out = mdot > 0 ? mdotT/mdot : T_in;
         double Q_out = mdot * p.Cp * (T_out - T_in);
         
         cout << "\n[Energy Balance]" << endl;
         cout << "  Q_in  = " << Q_in << " W" << endl;
         cout << "  T_out = " << T_out << " C (mass-avg)" << endl;
         cout << "  Q_out = " << Q_out << " W" << endl;
         cout << "  Imbalance = " << 100*(Q_in-Q_out)/(Q_in+1e-30) << "%" << endl;
     }
     
     void save_vtk(const string& path) {
        ofstream f(path);
        int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz();
        f << "# vtk DataFile Version 3.0\nThermal3D\nASCII\nDATASET RECTILINEAR_GRID\n";
        f << "DIMENSIONS " << Nx+1 << " " << Ny+1 << " " << Nz+1 << "\n";
        f << "X_COORDINATES " << Nx+1 << " double\n"; for(int i=0;i<=Nx;i++) f << i*p.dx() << " "; f << "\n";
        f << "Y_COORDINATES " << Ny+1 << " double\n"; for(int j=0;j<=Ny;j++) f << j*p.dy() << " "; f << "\n";
        f << "Z_COORDINATES " << Nz+1 << " double\n"; double z=0; for(int k=0;k<=Nz;k++){f<<z<<" ";if(k<Nz)z+=dz_cells[k];} f<<"\n";
        
        f << "CELL_DATA " << Nx*Ny*Nz << "\n";
        f << "SCALARS Temperature double 1\nLOOKUP_TABLE default\n";
        for(int k=0;k<Nz;k++) for(int j=0;j<Ny;j++) for(int i=0;i<Nx;i++) f << T[idx(i,j,k)] << "\n";
        
        f << "SCALARS Region int 1\nLOOKUP_TABLE default\n";
        for(int k=0;k<Nz;k++) for(int j=0;j<Ny;j++) for(int i=0;i<Nx;i++) f << region(i,j,k) << "\n";
        
        // Density field (gamma): 1=fluid, 0=solid, intermediate=buffer
        f << "SCALARS Density double 1\nLOOKUP_TABLE default\n";
        for(int k=0;k<Nz;k++) for(int j=0;j<Ny;j++) for(int i=0;i<Nx;i++) {
            if (k < p.nz_solid) f << "0\n";  // Base plate is solid
            else f << gamma[j*Nx+i] << "\n";
        }
        
        // Thermal conductivity field
        f << "SCALARS ThermalConductivity double 1\nLOOKUP_TABLE default\n";
        for(int k=0;k<Nz;k++) for(int j=0;j<Ny;j++) for(int i=0;i<Nx;i++) f << kval(i,j,k) << "\n";
        
        f << "VECTORS Velocity double\n";
        for(int k=0;k<Nz;k++) for(int j=0;j<Ny;j++) for(int i=0;i<Nx;i++) {
            int P = idx(i,j,k);
            f << U3d[P] << " " << V3d[P] << " " << W3d[P] << "\n";
        }
        
        cout << "Saved: " << path << endl;
    }
};

int main(int argc, char* argv[]) {
     string dir = argc > 1 ? argv[1] : "../ExportFiles";
     cout << "=== 3D CHT Solver ===" << endl;
     
     Params p;
     int M, N; double dx, dy, Ht, qf;
     ifstream pf(dir + "/thermal_params.txt");
     pf >> M >> N >> dx >> dy >> qf >> Ht;
     p.H_t = Ht;
     
     int rows, cols;
     // Load geometry_thermal.txt (supports binary or continuous values)
     auto gam = read_2d(dir + "/geometry_thermal.txt", rows, cols);
     if (gam.empty()) {
         cerr << "Error: Could not load geometry_thermal.txt" << endl;
         return 1;
     }
     
     // Count buffer cells to report geometry type
    int bufferCells = 0;
    for(int j=0;j<(int)gam.size();j++) {
        for(int i=0;i<(int)gam[j].size();i++) {
            double val = gam[j][i];
            if (val > 0.01 && val < 0.99) bufferCells++;
        }
    }
    cout << "Loaded geometry_thermal.txt (" << bufferCells << " buffer cells)" << endl;
     
     p.Ny = rows; p.Nx = cols; p.Lx = cols * dx; p.Ly = rows * dy;
     
     cout << "Grid: " << p.Nx << "x" << p.Ny << "x" << p.Nz() << " = " << p.n_dofs() << " DOFs" << endl;
     
     ThermalSolver3D s(p);
     s.gamma.resize(rows*cols); 
     // Convert to gamma convention: gamma=1 (fluid), gamma=0 (solid)
     for(int j=0;j<rows;j++) for(int i=0;i<cols;i++) {
         double val = gam[j][i];
         // geometry file: 0=fluid, 1=solid (or continuous 0.0-1.0)
         s.gamma[j*cols+i] = 1.0 - val;  // Invert to gamma convention
     }
     
     auto u = read_2d(dir + "/u_thermal.txt", rows, cols);
     auto v = read_2d(dir + "/v_thermal.txt", rows, cols);
     s.u_xy.assign(p.Ny*p.Nx, 0); s.v_xy.assign(p.Ny*p.Nx, 0);
     if (!u.empty()) 
         for(int j=0;j<p.Ny&&j<(int)u.size();j++) 
             for(int i=0;i<p.Nx&&i<(int)u[j].size();i++) { 
                 s.u_xy[j*p.Nx+i]=u[j][i]; 
                 s.v_xy[j*p.Nx+i]=v[j][i]; 
             }
     s.q0.assign(p.Ny*p.Nx, qf);
     
     s.solve();
     s.save_vtk(dir + "/thermal_results_3d.vtk");
     return 0;
 }
 