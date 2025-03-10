function plot_results(xc, yc, h, u, v, lx, ly, final_time)
    # Compute velocity magnitude
    vel = sqrt.(u.^2 .+ v.^2)
    
    # Create heatmaps for water depth and velocity magnitude
    p1 = heatmap(xc, yc, h', title="Final Water Depth", 
                 xlabel="x", ylabel="y", aspect_ratio=1)
    p2 = heatmap(xc, yc, vel', title="Final Velocity Magnitude", 
                 xlabel="x", ylabel="y", aspect_ratio=1)
    
    # Determine grid dimensions and center indices for slices
    nx, ny = size(h)
    i_center = Int(round(nx / 2))
    j_center = Int(round(ny / 2))
    
    # Extract slices through the center of h
    # Horizontal slice: row at the center (constant y)
    slice_horizontal = u[:,j_center]
    # Vertical slice: column at the center (constant x)
    slice_vertical = h[:, j_center]
    
    p3 = plot(xc, slice_horizontal, lw=2, marker=:circle,
              xlabel="x", ylabel="h", 
              title="Horizontal Slice at y = $(round(yc[j_center], digits=2))")
    p4 = plot(yc, slice_vertical, lw=2, marker=:circle,
              xlabel="y", ylabel="h",
              title="Vertical Slice at x = $(round(xc[i_center], digits=2))")
    
    # Combine the four plots into a 2x2 layout
    plot(p1, p2, p3, p4, layout=(2,2), size=(1200,800),
         title="Results at t = $final_time")
end

# Example usage:
# plot_four_subplots(xc, yc, h, u, v, lx, ly, timeout)
