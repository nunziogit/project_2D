function plot_results_3D(xc, yc, h, final_time)
    anim = @animate for theta in 0:5:355
        surface(xc, yc, h',
            title = "Final Water Depth at t = $final_time",
            xlabel = "x", ylabel = "y", zlabel = "Water Depth",
            aspect_ratio = :equal,
            margin = 2mm,           # Reduce margin to use space better
            color = :blues,
            camera = (theta, 30))
    end
    return anim
end
