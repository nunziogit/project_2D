# DOT 2D for solving the 2D Shallow Water Equations
using Printf, CUDA, IJulia, Revise, Plots, LinearAlgebra, Infiltrator

##default(size = (1200, 400), framestyle = :box, label = false, grid = false, margin = 10mm, lw = 6, labelfontsize = 20, tickfontsize = 20, titlefontsize = 24)
#
## Include the external file with the plotting function
include("functions.jl")

const PARAMETERS = (
    gravit = 9.81,
	useless = 1.0
)

@views function DOT_2D()
	# physics
	lx, ly  = 40.0, 40.0
	gravit  = PARAMETERS.gravit 
	R       = 2.5 #Radius of the initial circular dam
	hins    = 2.5
	hout    = 1.0
	timeout = 2.0
	# numerics
	nx = ny = 512
    nt = 10000
	ngp = 2  #number of gaussian points
	sgp, wgp = gaussian_points(ngp)

	# GPU settings
	threads = (32, 16)
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
	h_cpu .= ifelse.(D .<= R, hins, hout)
	for i ∈ 1:nx
		for j ∈ 1:ny
			D = sqrt((xc[i] - cx)^2 + (yc[j] - cy)^2)
			h_cpu[i, j] = D <= R ? hins : hout
		end
	end

	h = CuArray(h_cpu)
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
            hgpx    = CUDA.zeros(Float64, nx-1, ny)
            ugpx    = CUDA.zeros(Float64, nx-1, ny)
            vgpx    = CUDA.zeros(Float64, nx-1, ny)
			dpsidsx = CUDA.zeros(Float64, 3, nx - 1, ny)
			Axgp    = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
			Axgpabs = CUDA.zeros(Float64, 3, 3, nx - 1, ny)

			# Compute vectors
			hgpx  = @. h[1:end-1,:] + sgp[igp] * (h[2:end,:] - h[1:end-1,:]) # come se R - L
			ugpx  = @. u[1:end-1,:] + sgp[igp] * (u[2:end,:] - u[1:end-1,:])
			vgpx  = @. v[1:end-1,:] + sgp[igp] * (v[2:end,:] - v[1:end-1,:])            
			@cuda blocks=blocks threads=threads dpsidsx!(dpsidsx, h, u, v); synchronize()
			@cuda blocks=blocks threads=threads Axgp!(Axgp, hgpx, ugpx, vgpx); synchronize()
			@cuda blocks=blocks threads=threads Axgpabs!(Axgpabs, hgpx, ugpx, vgpx); synchronize()
			@cuda blocks=blocks threads=threads Dx!(DLx, DRx, dpsidsx, Axgp, Axgpabs, wgp[igp]); synchronize()

			# Deallocate vectors
			CUDA.unsafe_free!(hgpx)            
            CUDA.unsafe_free!(ugpx)            
            CUDA.unsafe_free!(vgpx)        
			CUDA.unsafe_free!(dpsidsx)
			CUDA.unsafe_free!(Axgp)
			CUDA.unsafe_free!(Axgpabs)

			# Allocate vectors ----------- Y -------------
            hgpy    = CUDA.zeros(Float64, nx, ny-1)
            ugpy    = CUDA.zeros(Float64, nx, ny-1)
            vgpy    = CUDA.zeros(Float64, nx, ny-1)
			dpsidsy = CUDA.zeros(Float64, 3, nx, ny - 1)
			Aygp    = CUDA.zeros(Float64, 3, 3, nx, ny - 1)
			Aygpabs = CUDA.zeros(Float64, 3, 3, nx, ny - 1)
            
			# Compute vectors
			hgpy = @. h[:, 1:end-1] + sgp[igp] * (h[:, 2:end] - h[:, 1:end-1])
			ugpy = @. u[:, 1:end-1] + sgp[igp] * (u[:, 2:end] - u[:, 1:end-1])
			vgpy = @. v[:, 1:end-1] + sgp[igp] * (v[:, 2:end] - v[:, 1:end-1])
			@cuda blocks=blocks threads=threads dpsidsy!(dpsidsy, h, u, v); synchronize()
    		@cuda blocks=blocks threads=threads Aygp!(Aygp, hgpy, ugpy, vgpy); synchronize()
    		@cuda blocks=blocks threads=threads Aygpabs!(Aygpabs, hgpy, ugpy, vgpy); synchronize()
    		@cuda blocks=blocks threads=threads Dy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp[igp]); synchronize()

			# Deallocate vectors
            CUDA.unsafe_free!(hgpy)
            CUDA.unsafe_free!(ugpy)
            CUDA.unsafe_free!(vgpy)
			CUDA.unsafe_free!(dpsidsy)
			CUDA.unsafe_free!(Aygp)
			CUDA.unsafe_free!(Aygpabs)
			@infiltrate false
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

		it%1000 == 0 && @printf("Iteration = %d, time = %.4f s \n", it, time)
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