function plot_results(xc, yc, h, u, v, lx, ly, final_time)
    # Compute velocity magnitude
    vel = sqrt.(u.^2 .+ v.^2)
    
    # Create heatmaps for h, u, v and vel
    p1 = heatmap(xc, yc, h', title="Water Depth (h)",
                 xlabel="x", ylabel="y", aspect_ratio=1)
    p2 = heatmap(xc, yc, u', title="Velocity u",
                 xlabel="x", ylabel="y", aspect_ratio=1)
    p3 = heatmap(xc, yc, v', title="Velocity v",
                 xlabel="x", ylabel="y", aspect_ratio=1)
    p4 = heatmap(xc, yc, vel', title="Velocity Magnitude",
                 xlabel="x", ylabel="y", aspect_ratio=1)
    
    # Determine grid dimensions
    nx, ny = size(h)
    n_diag = min(nx, ny)  # for diagonal extraction
    
    # Extract diagonal slices (45° slice through the domain)
    slice_h   = [h[i, i] for i in 1:n_diag]
    slice_u   = [u[i, i] for i in 1:n_diag]
    slice_v   = [v[i, i] for i in 1:n_diag]
    slice_vel = [vel[i, i] for i in 1:n_diag]
        
    # Compute a coordinate along the 45° slice.
    # For each diagonal point (xc[i], yc[i]) the distance from the start is:
    s = [sqrt((xc[i] - xc[1])^2 + (yc[i] - yc[1])^2) for i in 1:n_diag]
    
    # Create slice plots for each variable
    p5 = plot(s, slice_h, lw=2, marker=:circle,
              xlabel="Distance along 45° slice", ylabel="h",
              title="Diagonal Slice of h")
    p6 = plot(s, slice_u, lw=2, marker=:circle,
              xlabel="Distance along 45° slice", ylabel="u",
              title="Diagonal Slice of u")
    p7 = plot(s, slice_v, lw=2, marker=:circle,
              xlabel="Distance along 45° slice", ylabel="v",
              title="Diagonal Slice of v")
    p8 = plot(s, slice_vel, lw=2, marker=:circle,
              xlabel="Distance along 45° slice", ylabel="vel",
              title="Diagonal Slice of vel")
    
    # Combine heatmaps in one column and slices in the other.
    # Arrange them in a layout with 4 rows and 2 columns.
    display(plot(p1, p5, p2, p6, p3, p7, p4, p8, layout=(4,2), size=(1400,1000),
     title="Results at t = $final_time"))
end

# Example usage:
# plot_results(xc, yc, h, u, v, lx, ly, final_time)
