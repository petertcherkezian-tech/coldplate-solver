import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. PHYSICAL GEOMETRY DEFINITIONS (FIXED CONSTANTS)
# =============================================================================
# Units: meters

# Overall Domain
Lx_inlet    = 0.010      # 10 mm Inlet Buffer
Lx_heatsink = 0.040      # 40 mm Heatsink Length
Lx_outlet   = 0.010      # 10 mm Outlet Buffer
Ly_total    = 0.030      # 30 mm Total Height

# === MINIMUM FEATURE SIZE ENFORCEMENT ===
MIN_FEATURE_SIZE = 0.0004 # 0.4 mm Minimum Feature Size (Fluid or Solid)
min_solid_width = 0.002  # 2 mm Minimum solid width at top/bottom (boundary condition)

# Out-of-plane channel height for 2.5D model (Ht in governing equations)
Ht_channel  = 4/1000     # 4 mm fin height

# Random Seed for reproducible geometry
RANDOM_SEED = 42

# Brinkman buffer layers (0 = no buffer, any positive integer for smooth transition)
NUM_BUFFER_LAYERS = 6

# Channel / network settings
num_channels = 20         # Number of inlet "lanes" (and outlet lanes)
U_inlet_phys = 1         # m/s
HeatFlux     = 100.0     # W/cm^2

# --- INLET/OUTLET CHANNEL LOGIC ---
# 1.0 means channels extend through whole inlet (no open plenum).
inlet_channel_extension_frac = 1
inlet_plenum_fraction = max(0.0, min(1.0, 1.0 - inlet_channel_extension_frac))

# =============================================================================
# 2. MESH REFINEMENT SETTINGS
# =============================================================================
REFINEMENT_FACTOR = 4
BASE_CELLS_PER_CHANNEL = 5  # base cells per nominal width (used to set dx)

# Use a nominal width to set dx; actual widths are quantized to integer cells.
w_nominal = 0.0010  # 1.0 mm nominal for resolution choice

target_cells_channel = BASE_CELLS_PER_CHANNEL * REFINEMENT_FACTOR
dx = w_nominal / target_cells_channel
dy = dx

# Ensure minimum interior solid width is at least 3.5 cells to prevent singularities
min_solid_interior = max(MIN_FEATURE_SIZE, 3.5 * dx)
# Ensure minimum channel width is at least 3.5 cells
w_channel_min = max(0.0005, 3.5 * dx)
w_channel_max = max(0.0007, w_channel_min + 0.0002)

print("-" * 60)
print(f"GEOMETRY GENERATION V4 (Refinement: {REFINEMENT_FACTOR}x)")
print(f"WITH BRINKMAN BUFFER LAYERS")
print("-" * 60)
print(f"Nominal Width for dx: {w_nominal*1000:.3f} mm")
print(f"Target Cells per Nominal Width: {target_cells_channel}")
print(f"Computed Cell Size (dx): {dx*1e6:.1f} microns")

# Grid Dimensions
N_inlet    = int(round(Lx_inlet / dx))
N_heatsink = int(round(Lx_heatsink / dx))
N_outlet   = int(round(Lx_outlet / dx))
N_total    = N_inlet + N_heatsink + N_outlet

M_total    = int(round(Ly_total / dy))

# Inlet plenum split
N_inlet_plenum = int(round(N_inlet * inlet_plenum_fraction))

print(f"Grid Size: {N_total} x {M_total}")
print(f"Inlet Plenum Ends at Cell: {N_inlet_plenum}")

# Thermal cropping indices (USED LATER)
x_start_fin = N_inlet
x_end_fin   = N_inlet + N_heatsink

# =============================================================================
# 4. BUILD GEOMETRY MATRIX (STRAIGHT IN/OUT + MEANDERING HEATSINK)
# =============================================================================
# 1 = Solid, 0 = Fluid
geometry = np.ones((M_total, N_total), dtype=int)

# --- A. Create Inlet Plenum (Fully Open) ---
if N_inlet_plenum > 0:
    geometry[:, 0:N_inlet_plenum] = 0

# ----------------------------
# RANDOM NETWORK KNOBS
# ----------------------------
rng = np.random.default_rng(RANDOM_SEED)

# Routing resolution (routing step in fine-grid cells)
ROUTE_STEP_CELLS = 4

# Simultaneous Routing Parameters
SPLIT_PROB = 0.05        # Probability to split if width allows
WANDER_INTENSITY = 0.8   # Standard deviation of y-perturbation per step (in routing units)
MERGE_OVERLAP_FACTOR = 0.3 # Fraction of sum-of-widths to trigger merge

# Manhattan / Stepped Routing
MANHATTAN_ROUTING = True
TURN_STEP_INTERVAL = 10   # Steps between potential turns
TURN_MAGNITUDE = 2      # Standard deviation for the "jump" when turning

# =====================================================================
# Helper functions
# =====================================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def to_fine(i_r, j_r):
    """Routing coords -> fine-grid coords (x=i_f, y=j_f)."""
    return int(i_r * ROUTE_STEP_CELLS), int(j_r * ROUTE_STEP_CELLS)

def carve_disk(geom, i_f, j_f, w_cells):
    """Carve a small square 'disk' centered at (i_f,j_f) with width w_cells."""
    r = max(1, w_cells // 2)
    j0 = clamp(j_f - r, 1, M_total - 2)
    j1 = clamp(j_f + r + 1, 1, M_total - 1)
    i0 = clamp(i_f - r, 0, N_total - 1)
    i1 = clamp(i_f + r + 1, 0, N_total)
    geom[j0:j1, i0:i1] = 0

def carve_segment(geom, a, b, w_cells):
    """Carve an axis-aligned segment between fine points a and b."""
    (i0, j0), (i1, j1) = a, b
    if i0 == i1:
        step = 1 if j1 >= j0 else -1
        for j in range(j0, j1 + step, step):
            carve_disk(geom, i0, j, w_cells)
    elif j0 == j1:
        step = 1 if i1 >= i0 else -1
        for i in range(i0, i1 + step, step):
            carve_disk(geom, i, j0, w_cells)
    else:
        # Simple interpolation for non-axis aligned (used for bridges)
        dist = max(abs(i1-i0), abs(j1-j0))
        if dist == 0: return
        for s in range(dist + 1):
            t = s / dist
            fi = int(i0 + t * (i1 - i0))
            fj = int(j0 + t * (j1 - j0))
            carve_disk(geom, fi, fj, w_cells)

def carve_straight_x(geom, x0, x1, j_f, w_cells):
    """Carve straight in x from x0 to x1 at constant y=j_f (fine coords)."""
    step = 1 if x1 >= x0 else -1
    for i in range(x0, x1 + step, step):
        carve_disk(geom, i, j_f, w_cells)

# =====================================================================
# Generate inlet/outlet lane rows and carve
# =====================================================================
Rx = max(3, N_total // ROUTE_STEP_CELLS)
Ry = max(3, M_total // ROUTE_STEP_CELLS)

# Convert key x positions to routing coords
x_hs_start_r   = max(0, x_start_fin   // ROUTE_STEP_CELLS)
x_hs_end_r     = max(1, (x_end_fin-1) // ROUTE_STEP_CELLS)   # last routing col in heatsink

# Calculate Y-limits in Routing Coords
dy_route = dy * ROUTE_STEP_CELLS
j_r_min_margin = int(np.ceil(min_solid_width / dy_route))
j_r_max_margin = int(np.floor((Ly_total - min_solid_width) / dy_route))

# Ensure we don't go out of grid bounds (keeping the original 1 buffer just in case)
j_r_min_margin = max(1, j_r_min_margin)
j_r_max_margin = min(Ry - 2, j_r_max_margin)

print(f"Routing Y-limits: {j_r_min_margin} to {j_r_max_margin}")

# -------------------------
# 1) INITIALIZE INLET CHANNELS
# -------------------------
active_channels = []
for k in range(num_channels):
    # Distribute k uniformly between j_r_min_margin and j_r_max_margin
    # fraction goes from 0 to 1
    frac = (k + 0.5) / num_channels
    j_r = j_r_min_margin + frac * (j_r_max_margin - j_r_min_margin)
    
    j_r = int(round(j_r))
    j_r = clamp(j_r, j_r_min_margin, j_r_max_margin)
    
    # Random initial width
    w_phys = rng.uniform(w_channel_min, w_channel_max)
    
    active_channels.append({
        'y': float(j_r),
        'w': w_phys,
        'id': k,
        'last_x': x_hs_start_r,
        'last_y': float(j_r),
        'cooldown': 0  # Steps until eligible for merge
    })
    
    # Carve Straight Inlet
    w_cells = max(1, int(round(w_phys / dx)))
    _, j_f = to_fine(x_hs_start_r, j_r)
    carve_straight_x(geometry, N_inlet_plenum, x_start_fin, j_f, w_cells)

# -------------------------
# 2) SIMULTANEOUS ROUTING (MERGE/SPLIT)
# -------------------------
next_id = num_channels

for x_r in range(x_hs_start_r + 1, x_hs_end_r + 1):
    
    # Decrement Cooldowns
    for ch in active_channels:
        if ch['cooldown'] > 0:
            ch['cooldown'] -= 1

    # A. Move / Wander
    for ch in active_channels:
        delta = 0.0
        
        if MANHATTAN_ROUTING:
            # Only turn at specific intervals
            step_idx = x_r - x_hs_start_r
            if step_idx % TURN_STEP_INTERVAL == 0:
                delta = rng.normal(0, TURN_MAGNITUDE)
        else:
            # Continuous wandering
            delta = rng.normal(0, WANDER_INTENSITY)
            
        ch['y'] = clamp(ch['y'] + delta, j_r_min_margin, j_r_max_margin)

    # B. Draw Movement Segment (from x-1 to x)
    for ch in active_channels:
        w_cells = max(1, int(round(ch['w'] / dx)))
        
        if MANHATTAN_ROUTING:
            # L-shaped carving: Horizontal then Vertical
            # 1. Horizontal: (last_x, last_y) -> (x_r, last_y)
            p_start = to_fine(ch['last_x'], int(round(ch['last_y'])))
            p_corner = to_fine(x_r, int(round(ch['last_y'])))
            carve_segment(geometry, p_start, p_corner, w_cells)
            
            # 2. Vertical: (x_r, last_y) -> (x_r, y)
            p_end = to_fine(x_r, int(round(ch['y'])))
            carve_segment(geometry, p_corner, p_end, w_cells)
        else:
            # Direct diagonal carving
            p_prev = to_fine(ch['last_x'], int(round(ch['last_y'])))
            p_curr = to_fine(x_r, int(round(ch['y'])))
            carve_segment(geometry, p_prev, p_curr, w_cells)
        
        # Update last pos
        ch['last_x'] = x_r
        ch['last_y'] = ch['y']

    # C. Merge Logic
    active_channels.sort(key=lambda c: c['y'])
    merged_channels = []
    
    if active_channels:
        curr = active_channels[0]
        for next_ch in active_channels[1:]:
            # Check cooldowns
            if curr['cooldown'] > 0 or next_ch['cooldown'] > 0:
                merged_channels.append(curr)
                curr = next_ch
                continue

            # Check distance in physical units (approx)
            # dy is physical height of one fine cell
            # y in routing coords. 
            # Physical y distance = (next_ch.y - curr.y) * ROUTE_STEP_CELLS * dy
            # But widths are in meters.
            
            y_dist_phys = (next_ch['y'] - curr['y']) * ROUTE_STEP_CELLS * dy
            width_sum = curr['w'] + next_ch['w']
            gap = y_dist_phys - 0.5 * width_sum
            
            # Merge if the wall between them is too thin
            # Resolution-Aware: add a buffer of 1.5 dx to account for rounding jitter
            if gap < (min_solid_interior + 1.5 * dx):
                # MERGE
                new_w = curr['w'] + next_ch['w']
                # Weighted average for new y
                new_y = (curr['y'] * curr['w'] + next_ch['y'] * next_ch['w']) / new_w
                
                print(f"MERGE at x={x_r}: w={curr['w']:.5f}+{next_ch['w']:.5f} -> {new_w:.5f}")

                # Create merged channel
                new_ch = {
                    'y': new_y,
                    'w': new_w,
                    'id': next_id,
                    'last_x': x_r,
                    'last_y': new_y,
                    'cooldown': 0
                }
                next_id += 1
                
                # Draw Bridges from old tips to new merge point
                p_curr_old = to_fine(x_r, int(round(curr['y'])))
                p_next_old = to_fine(x_r, int(round(next_ch['y'])))
                p_merge    = to_fine(x_r, int(round(new_y)))
                
                w_curr_cells = max(1, int(round(curr['w'] / dx)))
                w_next_cells = max(1, int(round(next_ch['w'] / dx)))
                
                carve_segment(geometry, p_curr_old, p_merge, w_curr_cells)
                carve_segment(geometry, p_next_old, p_merge, w_next_cells)
                
                curr = new_ch
            else:
                merged_channels.append(curr)
                curr = next_ch
        merged_channels.append(curr)
        active_channels = merged_channels

    # D. Split Logic
    final_channels = []
    for ch in active_channels:
        # Check if wide enough to split (must result in at least 2 * w_min)
        if ch['w'] >= 2.0 * w_channel_min and rng.random() < SPLIT_PROB:
            # SPLIT
            # Ensure upper bound is at least lower bound (handle float precision)
            high = max(w_channel_min, ch['w'] - w_channel_min)
            w1 = rng.uniform(w_channel_min, high)
            w2 = ch['w'] - w1
            
            # Calculate Safe Separation to avoid immediate re-merge
            # We need gap >= min_solid_interior
            # gap = sep_phys - 0.5*(w1+w2)
            # sep_phys >= min_solid_interior + 0.5*(w1+w2)
            
            sep_phys_min = min_solid_interior + 0.5 * (w1 + w2)
            # Convert to routing cells (approx y distance)
            # sep_cells = sep_phys / (ROUTE_STEP_CELLS * dy)
            min_sep_cells = sep_phys_min / (ROUTE_STEP_CELLS * dy)
            
            sep = max(2.0, min_sep_cells * 1.1) # Add 10% buffer
            
            print(f"SPLIT at x={x_r}: w={ch['w']:.5f} -> {w1:.5f}, {w2:.5f} (sep={sep:.1f})")

            y1 = clamp(ch['y'] - sep/2, j_r_min_margin, j_r_max_margin)
            y2 = clamp(ch['y'] + sep/2, j_r_min_margin, j_r_max_margin)
            
            # Set cooldown to prevent immediate re-merge
            cooldown_steps = 8
            
            ch1 = {'y': y1, 'w': w1, 'id': next_id,   'last_x': x_r, 'last_y': y1, 'cooldown': cooldown_steps}
            ch2 = {'y': y2, 'w': w2, 'id': next_id+1, 'last_x': x_r, 'last_y': y2, 'cooldown': cooldown_steps}
            next_id += 2
            
            # Draw Bridges
            p_orig = to_fine(x_r, int(round(ch['y'])))
            p_new1 = to_fine(x_r, int(round(y1)))
            p_new2 = to_fine(x_r, int(round(y2)))
            
            w1_cells = max(1, int(round(w1 / dx)))
            w2_cells = max(1, int(round(w2 / dx)))
            
            carve_segment(geometry, p_orig, p_new1, w1_cells)
            carve_segment(geometry, p_orig, p_new2, w2_cells)
            
            final_channels.append(ch1)
            final_channels.append(ch2)
        else:
            final_channels.append(ch)
    
    active_channels = final_channels

# -------------------------
# 3) STRAIGHT OUTLET REGION
# -------------------------
# Extend whatever channels remain at the end
for ch in active_channels:
    w_cells = max(1, int(round(ch['w'] / dx)))
    _, j_f = to_fine(x_hs_end_r, int(round(ch['y'])))
    carve_straight_x(geometry, x_end_fin, N_total - 1, j_f, w_cells)

# Ensure top/bottom walls solid
geometry[0, :] = 1
geometry[-1, :] = 1

# --- 4.5 VOXEL CLEANUP (MORPHOLOGICAL SMOOTHING) ---
# Detect and remove isolated 1-cell clusters (fluid or solid) to prevent CFD bottlenecks
print("Cleaning topology...")
def cleanup_topology(geom):
    M, N = geom.shape
    changed = False
    # Copy to avoid feedback during sweep
    new_geom = geom.copy()
    for j in range(1, M-1):
        for i in range(1, N-1):
            # If a pixel is different from all its 4 orthogonal neighbors, it's a singularity
            neighbors = [geom[j-1, i], geom[j+1, i], geom[j, i-1], geom[j, i+1]]
            if all(n != geom[j, i] for n in neighbors):
                new_geom[j, i] = neighbors[0] # Flip to match neighbors
                changed = True
    return new_geom, changed

geometry, _ = cleanup_topology(geometry)

# =============================================================================
# 5. ADD BRINKMAN BUFFER LAYERS
# =============================================================================
# Create cellType field with buffer layers at solid-fluid interfaces
# CFD solver expects: cellType = 0 (fluid), cellType = 1 (solid)
# Buffer layers use intermediate values for smooth Brinkman transition
# Solver computes: gamma = 1.0 - cellType, alpha = alpha_max * (1 - gamma)

print(f"Adding {NUM_BUFFER_LAYERS} Brinkman buffer layer(s)...")

# Start with binary geometry: 0 = fluid, 1 = solid
cellType_final = geometry.astype(float).copy()

if NUM_BUFFER_LAYERS > 0:
    # Iteratively add buffer layers from solid boundary outward
    # Layer 1 (adjacent to solid): highest cellType value (closest to solid)
    # Layer N (furthest from solid): lowest cellType value (closest to fluid)
    
    for layer in range(1, NUM_BUFFER_LAYERS + 1):
        # cellType value for this layer: linearly interpolate from 0.5 (layer 1) to near 0
        # Layer 1: 0.5, Layer 2: 0.25, Layer 3: 0.125, etc. (or linear: 0.5, 0.33, 0.17...)
        # Using linear spacing: layer_value = 0.5 * (NUM_BUFFER_LAYERS - layer + 1) / NUM_BUFFER_LAYERS
        layer_value = 0.5 * (NUM_BUFFER_LAYERS - layer + 1) / NUM_BUFFER_LAYERS
        
        # Create a copy to detect neighbors without feedback
        cellType_prev = cellType_final.copy()
        
        for j in range(1, M_total - 1):
            for i in range(1, N_total - 1):
                # If this is still a pure fluid cell (cellType = 0)
                if cellType_prev[j, i] < 0.01:
                    # Check if any neighbor is solid (layer 1) or previous buffer layer
                    has_target_neighbor = False
                    for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nj, ni = j + dj, i + di
                        if 0 <= nj < M_total and 0 <= ni < N_total:
                            neighbor_val = cellType_prev[nj, ni]
                            if layer == 1:
                                # First layer: look for solid neighbors
                                if neighbor_val > 0.99:  # Solid
                                    has_target_neighbor = True
                                    break
                            else:
                                # Subsequent layers: look for previous buffer layer
                                prev_layer_value = 0.5 * (NUM_BUFFER_LAYERS - layer + 2) / NUM_BUFFER_LAYERS
                                if abs(neighbor_val - prev_layer_value) < 0.01:
                                    has_target_neighbor = True
                                    break
                    
                    if has_target_neighbor:
                        cellType_final[j, i] = layer_value

print(f"Buffer layers added: {np.sum((cellType_final > 0.01) & (cellType_final < 0.99))} cells")

# =============================================================================
# 6. EXPORT FILES WITH BUFFER LAYERS
# =============================================================================
OUTPUT_DIR = "ExportFiles"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Export fluid parameters (unchanged)
with open(os.path.join(OUTPUT_DIR, "fluid_params.txt"), "w") as f:
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {U_inlet_phys} {N_inlet} {N_outlet} {Ht_channel:.9f}")

# Export geometry with buffer layers (cellType field)
# CFD solver expects: 0 = fluid, 1 = solid, intermediate = buffer
# Solver computes gamma = 1.0 - cellType
np.savetxt(os.path.join(OUTPUT_DIR, "geometry_fluid.txt"), cellType_final, fmt='%.6f', delimiter='\t')

# Export thermal geometry (cropped to heatsink region)
geo_therm = cellType_final[:, x_start_fin:x_end_fin]
q_flux_wm2 = HeatFlux * 10000.0

with open(os.path.join(OUTPUT_DIR, "thermal_params.txt"), "w") as f:
    f.write(f"{M_total} {N_heatsink} {dy:.9f} {dx:.9f} {q_flux_wm2} {Ht_channel:.9f}")

np.savetxt(os.path.join(OUTPUT_DIR, "geometry_thermal.txt"), geo_therm, fmt='%.6f', delimiter='\t')

print(f"Files saved to {OUTPUT_DIR}/")
print(f"cellType field: 0.0=fluid, 0.25/0.5=buffer, 1.0=solid")

# =============================================================================
# 7. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Binary geometry (original)
axes[0].imshow(geometry, cmap='gray_r', origin='lower', interpolation='nearest')
axes[0].set_title(
    f"Binary Geometry (Refinement {REFINEMENT_FACTOR})\n"
    f"Manhattan={MANHATTAN_ROUTING} (Int={TURN_STEP_INTERVAL}) | SPLIT_PROB={SPLIT_PROB}"
)
axes[0].set_xlabel("X (Cells)")
axes[0].set_ylabel("Y (Cells)")
axes[0].axvline(x=N_inlet_plenum, color='r', linestyle='--', label='Plenum End', linewidth=0.8)
axes[0].axvline(x=N_inlet, color='g', linestyle='--', label='Heatsink Start', linewidth=0.8)
axes[0].axvline(x=N_inlet + N_heatsink, color='b', linestyle='--', label='Heatsink End', linewidth=0.8)
axes[0].legend(fontsize=8)

# Right: cellType field with buffer layers
im = axes[1].imshow(cellType_final, cmap='viridis_r', origin='lower', interpolation='nearest', vmin=0, vmax=1)
axes[1].set_title(
    f"cellType Field (with Brinkman Buffers)\n"
    f"0.0=Fluid, 0.25/0.5=Buffer, 1.0=Solid"
)
axes[1].set_xlabel("X (Cells)")
axes[1].set_ylabel("Y (Cells)")
axes[1].axvline(x=N_inlet_plenum, color='r', linestyle='--', label='Plenum End', linewidth=0.8)
axes[1].axvline(x=N_inlet, color='g', linestyle='--', label='Heatsink Start', linewidth=0.8)
axes[1].axvline(x=N_inlet + N_heatsink, color='b', linestyle='--', label='Heatsink End', linewidth=0.8)
cbar = plt.colorbar(im, ax=axes[1])
cbar.set_label("cellType (0=fluid, 1=solid)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('geometry_preview_v4.png', dpi=150)
plt.show()

print("\n" + "="*60)
print("GeometryGeneratorV4 Complete!")
print("="*60)
