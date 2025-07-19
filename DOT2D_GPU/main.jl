# DOT 2D for solving the 2D Shallow Water Equations
using Printf, CUDA, BechmarckTools, IJulia, Revise, Plots, LinearAlgebra

##default(size = (1200, 400), framestyle = :box, label = false, grid = false, margin = 10mm, lw = 6, labelfontsize = 20, tickfontsize = 20, titlefontsize = 24)
#
## Include the external file with the plotting function
include("functions.jl")

const PARAMETERS = (
    gravit = 9.81
)

@views function DOT_2D()
	# physics
	lx, ly  = 40.0, 40.0
	gravit = PARAMETERS.gravit 
	R       = 2.5 #Radius of the initial circular dam
	hins    = 2.5
	hout    = 1.0
	timeout = 2.0
	# numerics
	# nx, ny = 101, 101 # ORIGINAL
	nx = ny = 8192
    nt = 10000
	ngp = 2  #number of gaussian points
	sgp, wgp = gaussian_points(ngp)

	# GPU settings
	threads = (32, 32)
	blocks = (nx÷threads[1], ny÷threads[2])

	#nvis = 20
	cfl = 0.49
	# derived numerics
	dx, dy = lx / nx, ly / ny
	xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)

	# array initialisation
    h_cpu  = zeros(Float64, nx, ny)
    qx_cpu = zeros(Float64, nx, ny)
    qy_cpu = zeros(Float64, nx, ny)
    u_cpu  = zeros(Float64, nx, ny)
    v_cpu  = zeros(Float64, nx, ny)

	DLx     = CUDA.zeros(Float64, 3, nx - 1, ny)
	DRx     = CUDA.zeros(Float64, 3, nx - 1, ny)

    h  = CUDA.zeros(Float64, nx, ny)
	u  = CUDA.zeros(Float64, nx, ny)
	v  = CUDA.zeros(Float64, nx, ny)
    qx = CUDA.zeros(Float64, nx, ny)
    qy = CUDA.zeros(Float64, nx, ny)
   
	DLy     = CUDA.zeros(Float64, 3, nx, ny - 1)
	DRy     = CUDA.zeros(Float64, 3, nx, ny - 1)


	## Create a Boolean mask for points inside the square
	#in_square = (abs.(xc_col .- cx) .<= R / 2) .& (abs.(yc_row .- cy) .<= R / 2)
	#
	# Set h: use hins for points inside the square, and hout otherwise
	cx = 0.5lx
	cy = 0.5ly
	#h .= ifelse.(in_square, hins, hout)
	D = [sqrt((xi - cx)^2 + (yj - cy)^2) for xi in xc, yj in yc]
	h .= ifelse.(D .<= R, hins, hout)
	for i ∈ 1:nx
		for j ∈ 1:ny
			D = sqrt((xc[i] - cx)^2 + (yc[j] - cy)^2)
			h[i, j] = D <= R ? hins : hout
		end
	end


	#set_dambreak!(h, hins, hout; direction = :y)
	qx = @. (u * h)
	qy = @. (v * h)
	# h_i, u_i, v_i = copy(h), copy(u), copy(v)
	# time loop
	time = 0.0
	timeout_reached = false
	@time for it ∈ 1:nt 
		# set CFL condition
		λx = @. (abs(u) + sqrt(gravit * h))
		λy = @. (abs(v) + sqrt(gravit * h))
		dtlx = dx ./ λx
		dtly = dy ./ λy
		dtl = min(minimum(dtlx), minimum(dtly))
		dt = cfl * dtl
		#dt = 0.01
		if time + dt > timeout
			dt = timeout - time
			timeout_reached = true
		end
		time += dt

		# Precompute constant factors
		dtdx = dt / dx
		dtdy = dt / dy

        @views for igp ∈ 1:ngp
			# Allocate vectors ----------- X -------------
            hgpx    = cuda.zeros(Float64, 3, nx-1, ny)
            ugpx    = cuda.zeros(Float64, 3, nx-1, ny)
            vgpx    = cuda.zeros(Float64, 3, nx-1, ny)
			dpsidsx = CUDA.zeros(Float64, 3, nx - 1, ny)
			Axgp    = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
			Axgpabs = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
			
			# Compute vectors
			hgpx  = @. h[1:end-1,:] + sgp[igp] * (h[2:end,:] - h[1:end-1,:]) # come se R - L
			ugpx  = @. u[1:end-1,:] + sgp[igp] * (u[2:end,:] - u[1:end-1,:])
			vgpx  = @. v[1:end-1,:] + sgp[igp] * (v[2:end,:] - v[2:end-1,:])            
			@cuda blocks=blocks threads=threads dpsisdx!(dpsidsx, h, u, v); synchronize()
    		@cuda blocks=blocks threads=threads Axgp!(Axgp, hgp, ugp, vgp); synchronize()
    		@cuda blocks=blocks threads=threads Axgpabs!(Axgpabs, hgp, ugp, vgp); synchronize()
    		@cuda blocks=blocks threads=threads Dx!(DLx, DRx, dpsidsx, Axgp, Axgpabs, wgp); synchronize()
			
			# Deallocate vectors
			CUDA.unsafe_free(hgpx)            
            CUDA.unsafe_free(ugpx)            
            CUDA.unsafe_free(vgpx)        
			CUDA.unsafe_free(dpsidsx)
			CUDA.unsafe_free(Axgp)
			CUDA.unsafe_free(Axgpabs)

			# Allocate vectors ----------- Y -------------
            hgpy    = cuda.zeros(Float64, 3, nx, ny-1)
            ugpy    = cuda.zeros(Float64, 3, nx, ny-1)
            vgpy    = cuda.zeros(Float64, 3, nx, ny-1)
			dpsidsy = CUDA.zeros(Float64, 3, nx, ny - 1)
			Aygp    = CUDA.zeros(Float64, 3, 3, nx, ny - 1)
			Aygpabs = CUDA.zeros(Float64, 3, 3, nx, ny - 1)
            
			# Compute vectors
			hgpy = @. h[:, 1:end-1] + sgp[igp] * (h[:, 2:end] - h[:, 1:end-1])
			ugpy = @. u[:, 1:end-1] + sgp[igp] * (u[:, 2:end] - u[:, 1:end-1])
			vgpy = @. v[:, 1:end-1] + sgp[igp] * (v[:, 2:end] - v[:, 1:end-1])
			@cuda blocks=blocks threads=threads dpsisdy!(dpsidsy, h, u, v); synchronize()
    		@cuda blocks=blocks threads=threads Aygp!(Aygp, hgp, ugp, vgp); synchronize()
    		@cuda blocks=blocks threads=threads Aygpabs!(Aygpabs, hgp, ugp, vgp); synchronize()
    		@cuda blocks=blocks threads=threads Dy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp); synchronize()

			# Deallocate vectors
            CUDA.unsafe_free(hgpy)
            CUDA.unsafe_free(ugpy)
            CUDA.unsafe_free(vgpy)
			CUDA.unsafe_free(dpsidsy)
			CUDA.unsafe_free(Aygp)
			CUDA.unsafe_free(Aygpabs)
        end

        # update!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy, threads, blocks)
		@cuda blocks=blocks threads=threads hqxqy!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy); synchronize()


        #= Bullshit
		#Set fluctuation to zero
		DLx .= 0.0
		DRx .= 0.0
		DLy .= 0.0
		DRy .= 0.0
        =#

		# Update velocity safely to avoid division by zero
		#u .= @. ifelse(h > 1e-12, qx / h, 0.0)
		#v .= @. ifelse(h > 1e-12, qy / h, 0.0)
		# Update velocities from momentum and depth, making sure to avoid division by zero.
		if (it%100 == 0)
            h_cpu  = Array(h)
            qx_cpu = Array(qx)
            qy_cpu = Array(qy)
            for i in 1:(nx)
		    	for j in 1:(ny)
		    		if  h_cpu[i, j] > 1e-12
		    			u_cpu[i, j] = qx_cpu[i, j] / h_cpu[i, j]
		    			v_cpu[i, j] = qy_cpu[i, j] / h_cpu[i, j]
		    		else
		    			u_cpu[i, j] = 0.0
		    			v_cpu[i, j] = 0.0
		    		end
		    	end
		    end
        end
		# Exit the loop if timeout_reached is true
		if timeout_reached
			println("Timeout reached")
			break
		end
        
	end
	# Call the plot_results function after the loop
	#@show (u)
	plot_results(xc, yc, h_cpu, u_cpu, v_cpu, lx, ly, timeout)
	# Call the plot_results_3D function to visualize the results in 3D
	#plot_results_3D(xc, yc, h, timeout)
	#anim = plot_results_3D(xc, yc, h, timeout)
	#gif(anim, "results.gif", fps = 15)


end

DOT_2D()

# ---------------------- FUNCTIONS --------------------

function gaussian_points(ngp::Int)
	if ngp == 1
		sgp = [0.5]
		wgp = [1.0]
	elseif ngp == 2
		sgp = [0.5 - sqrt(3) / 6, 0.5 + sqrt(3) / 6]
		wgp = [0.5, 0.5]
	elseif ngp == 3
		sgp = [0.5 - sqrt(15) / 10, 0.5, 0.5 + sqrt(15) / 10]
		wgp = [5 / 18, 8 / 18, 5 / 18]
	else
		error("Unsupported number of Gaussian points: $ngp")
	end
	return sgp, wgp
end



function set_dambreak!(h, hins, hout; direction = :x)
	nx, ny = size(h)
	if direction == :x
		igate = nx ÷ 2
		for i in 1:nx
			for j in 1:ny
				h[i, j] = i <= igate ? hins : hout
			end
		end
	elseif direction == :y
		jgate = ny ÷ 2
		for i in 1:nx
			for j in 1:ny
				h[i, j] = j <= jgate ? hins : hout
			end
		end
	else
		error("Unknown direction: choose :x or :y")
	end
end

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
    plot(p1, p5, p2, p6, p3, p7, p4, p8, layout=(4,2), size=(1400,1000),
     title="Results at t = $final_time")
end

# Example usage:
# plot_results(xc, yc, h, u, v, lx, ly, final_time)


## --------------------------- ##

# Example usage:
# Create a dam-break along the x-direction
#set_dambreak!(h, hins, hout; direction=:x)

# Create a dam-break along the y-direction
#set_dambreak!(h, hins, hout; direction=:y)

#using ProfileView
#@profview DOT_2D()