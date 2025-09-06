using Plots, Plots.Measures, Printf
using BenchmarkTools
using StaticArrays, LoopVectorization

function gaussian_points(ngp::Int) #=  =#
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

function apply_reflective_boundaries!(h, qx, qy)
	# Left wall (x = 0)
	h[1, :]  .= h[2, :]
	qx[1, :] .= -qx[2, :]
	qy[1, :] .= qy[2, :]

	# Right wall (x = lx)
	h[end, :]  .= h[end-1, :]
	qx[end, :] .= -qx[end-1, :]
	qy[end, :] .= qy[end-1, :]

	# Bottom wall (y = 0)
	h[:, 1]  .= h[:, 2]
	qx[:, 1] .= qx[:, 2]
	qy[:, 1] .= -qy[:, 2]

	# Top wall (y = ly)
	h[:, end]  .= h[:, end-1]
	qx[:, end] .= qx[:, end-1]
	qy[:, end] .= -qy[:, end-1]

	return nothing
end

function compute_dt(nx, ny, _dx, _dy, cfl, u, v, h, gravit)
	nthreads = Threads.nthreads()
	# Create an array to hold the maximum from each thread.
	local_maxes = zeros(Float64, nthreads)

	Threads.@threads for j in 1:ny
		tid = Threads.threadid()
		local_max = 0.0
		@inbounds for i in 1:nx
			# Compute the local wave speed for cell (i, j)
			local_speed = abs(u[i, j]) + sqrt(gravit * h[i, j])
			r1 = local_speed * _dx
			r2 = local_speed * _dy
			local_max = max(local_max, r1, r2)
		end
		# Update the thread-local maximum.
		local_maxes[tid] = max(local_maxes[tid], local_max)
	end

	# Reduce the thread-local max values to a global maximum.
	global_max = maximum(local_maxes)
	dt = cfl / global_max
	return dt
end


function process_fluxes_x_merged!(
	dpsidsx, Axgp, Axgpabs,
	DLx, DRx,
	hl, hr, ul, ur, vl, vr,
	ugp, vgp, c2gp,
	λ1, λ2, λ3,
	w::Float64, nx::Int, ny::Int,
)
	@inbounds Threads.@threads for i in 1:(nx-1)
		for j in 1:ny
			# --- dpsidsx computation ---
			hrij = hr[i, j]
			hlij = hl[i, j]
			urij = ur[i, j]
			ulij = ul[i, j]
			vrij = vr[i, j]
			vlij = vl[i, j]
			ugpij = ugp[i, j]
			vgpij = vgp[i, j]

			# Compute the difference vector and store as a StaticVector
			local_dps = @SVector [hrij - hlij,
				urij * hrij - ulij * hlij,
				vrij * hrij - vlij * hlij]
			# Optionally store into the full array if needed:
			dpsidsx[:, i, j] .= local_dps

			# --- Axgp computation ---
			# Build the 3x3 flux matrix using StaticArrays
			localAx = @SMatrix [                                 0.0                     1.0                     0.0;
				 c2gp[i, j]-ugpij*ugpij   2.0*ugpij          0.0;
				-ugpij*vgpij       vgpij              ugpij]
			Axgp[:, :, i, j] .= localAx

			# --- Axgpabs computation ---
			λ1ij = λ1[i, j]
			λ2ij = λ2[i, j]
			λ3ij = λ3[i, j]
			absλ1ij = abs(λ1ij)
			absλ2ij = abs(λ2ij)
			absλ3ij = abs(λ3ij)
			den = λ1ij - λ3ij
			invden = 1.0 / den          # reciprocal of den
			absden = absλ1ij - absλ3ij

			localAxabs = if abs(den) > 1e-12
				@SMatrix [            (-absλ1ij*λ3ij+absλ3ij*λ1ij)*invden            absden*invden                 0.0;
					-λ1ij*λ3ij*absden*invden                         (λ1ij*absλ1ij-λ3ij*absλ3ij)*invden           0.0;
					-((den * absλ2ij - absλ3ij * λ1ij + absλ1ij * λ3ij) * vgpij)*invden    vgpij*absden*invden     absλ2ij]
			else
				@SMatrix zeros(3, 3)
			end
			Axgpabs[:, :, i, j] .= localAxabs

			# --- Flux accumulation ---
			# Compute the contribution: (Ax ± Axabs) * dps
			local_flux_left  = w * (localAx - localAxabs) * local_dps
			local_flux_right = w * (localAx + localAxabs) * local_dps

			DLx[:, i, j] .+= local_flux_left
			DRx[:, i, j] .+= local_flux_right
		end
	end
	return nothing
end

function process_fluxes_y_merged!(
	dpsidsy, Aygp, Aygpabs,  # output arrays: dpsidsy (3×nx×(ny-1)), Aygp and Aygpabs (3×3×nx×(ny-1))
	DLy, DRy,               # accumulation arrays: DLy, DRy (3×nx×(ny-1))
	hb, ht, ub, ut, vb, vt,   # cell-face arrays for bottom/top (size: nx×(ny-1))
	ugp, vgp, c2gp,         # interpolated values (size: nx×(ny-1))
	λ1, λ2, λ3,             # eigenvalue arrays (size: nx×(ny-1))
	w::Float64,             # weight for the current Gaussian point
	nx::Int, ny::Int,        # grid dimensions (with ny as the number of cells in y, so faces are ny-1)
)
	@inbounds Threads.@threads for j in 1:(ny-1)
		for i in 1:nx
			# --- dpsidsy computation ---
			local_ht = ht[i, j]
			local_hb = hb[i, j]
			local_ut = ut[i, j]
			local_ub = ub[i, j]
			local_vt = vt[i, j]
			local_vb = vb[i, j]

			# Compute the difference vector (as a static vector)
			local_dps = @SVector [local_ht - local_hb,
				local_ut * local_ht - local_ub * local_hb,
				local_vt * local_ht - local_vb * local_hb]
			dpsidsy[:, i, j] .= local_dps

			# --- Aygp computation ---
			# Cache interpolated values
			local_ugp = ugp[i, j]
			local_vgp = vgp[i, j]
			local_c2gp = c2gp[i, j]
			# Build the 3×3 flux matrix for y-direction using StaticArrays.
			# Note: For y-direction, the structure is different:
			#   Aygp[1, :] corresponds to the "momentum" in the y direction.
			localAy = @SMatrix [                 0.0                  0.0            1.0;
				-local_ugp*local_vgp   local_vgp     local_ugp;
				 local_c2gp-local_vgp^2    0.0         2.0*local_vgp]
			Aygp[:, :, i, j] .= localAy

			# --- Aygpabs computation ---
			# Extract eigenvalue scalars from the y arrays
			local_λ1 = λ1[i, j]
			local_λ2 = λ2[i, j]
			local_λ3 = λ3[i, j]
			local_absλ1 = abs(local_λ1)
			local_absλ2 = abs(local_λ2)
			local_absλ3 = abs(local_λ3)
			local_den = local_λ1 - local_λ3
			local_absden = local_absλ1 - local_absλ3

			localAyabs = if abs(local_den) > 1e-12
				@SMatrix [      (-local_absλ1*local_λ3+local_absλ3*local_λ1)/local_den      0.0                   local_absden/local_den;
					-((local_den * local_absλ2 - local_absλ3 * local_λ1 + local_absλ1 * local_λ3) * local_ugp)/local_den abs(local_λ2)   local_ugp*local_absden/local_den;
					-local_λ1*local_λ3*local_absden/local_den                           0.0                   (local_λ1*local_absλ1-local_λ3*local_absλ3)/local_den]
			else
				@SMatrix zeros(3, 3)
			end
			Aygpabs[:, :, i, j] .= localAyabs

			# --- Flux accumulation ---
			# Compute the flux contribution: (Aygp ± Aygpabs) * dps
			local_flux_left = w * (localAy - localAyabs) * local_dps
			local_flux_right = w * (localAy + localAyabs) * local_dps

			DLy[:, i, j] .+= local_flux_left
			DRy[:, i, j] .+= local_flux_right
		end
	end

	return nothing
end

#function update_avx!(h, qx, qy, DLx, DRx, DLy, DRy, dtdx, dtdy, nx, ny)
#    @avx for j in 2:(ny-1), i in 2:(nx-1)
#        # Compute divergence contributions
#        div_x = dtdx * (DRx[1, i-1, j] + DLx[1, i, j])
#        div_y = dtdy * (DRy[1, i, j-1] + DLy[1, i, j])
#        h[i, j] -= div_x + div_y
#
#        div_x_qx = dtdx * (DRx[2, i-1, j] + DLx[2, i, j])
#        div_y_qx = dtdy * (DRy[2, i, j-1] + DLy[2, i, j])
#        qx[i, j] -= div_x_qx + div_y_qx
#
#        div_x_qy = dtdx * (DRx[3, i-1, j] + DLx[3, i, j])
#        div_y_qy = dtdy * (DRy[3, i, j-1] + DLy[3, i, j])
#        qy[i, j] -= div_x_qy + div_y_qy
#    end
#    return nothing
#end


function update!(h, qx, qy, DLx, DRx, DLy, DRy, dtdx, dtdy, nx::Int, ny::Int)
	@inbounds Threads.@threads for j in 2:(ny-1)
		for i in 2:(nx-1)
			# Precompute divergence terms for h:
			div_x = dtdx * (DRx[1, i-1, j] + DLx[1, i, j])
			div_y = dtdy * (DRy[1, i, j-1] + DLy[1, i, j])
			h[i, j] -= div_x + div_y

			# For qx:
			div_x_qx = dtdx * (DRx[2, i-1, j] + DLx[2, i, j])
			div_y_qx = dtdy * (DRy[2, i, j-1] + DLy[2, i, j])
			qx[i, j] -= div_x_qx + div_y_qx

			# For qy:
			div_x_qy = dtdx * (DRx[3, i-1, j] + DLx[3, i, j])
			div_y_qy = dtdy * (DRy[3, i, j-1] + DLy[3, i, j])
			qy[i, j] -= div_x_qy + div_y_qy
		end
	end
	return nothing
end

function update_velocity!(u, v, qx, qy, h, nx, ny)
	@inbounds Threads.@threads for i in 1:nx
		for j in 1:ny
			if h[i, j] > 1e-12
				inv_h = 1.0 / h[i, j]  # Compute reciprocal once.
				u[i, j] = qx[i, j] * inv_h
				v[i, j] = qy[i, j] * inv_h
			else
				u[i, j] = 0.0
				v[i, j] = 0.0
			end
		end
	end
	return nothing
end


@views function DOT_2D(; do_check = false)

	println("Number of threads: ", Threads.nthreads())

	# physics
	lx, ly  = 40.0, 40.0
	gravit  = 9.81
	R       = 2.5 #Radius of the initial circular dam
	hins    = 2.5
	hout    = 1.0
	timeout = 4.0
	# numerics
	nx, ny = 201, 201
	maxiter_t = 10000
	ngp = 1  #number of gaussian points
	sgp, wgp = gaussian_points(ngp)
	cfl = 0.49
	#iteration loop
	nvis = 20
	it = 1

	# derived numeri]cs
	dx, dy = lx / nx, ly / ny
	xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)

	#performance scalars
	_dx = 1.0 / dx
	_dy = 1.0 / dy

	# array initialisation
	h = zeros(Float64, nx, ny)
	u = zeros(Float64, nx, ny)
	v = zeros(Float64, nx, ny)

	Axgp    = zeros(Float64, 3, 3, nx - 1, ny)
	Axgpabs = zeros(Float64, 3, 3, nx - 1, ny)
	dpsidsx = zeros(Float64, 3, nx - 1, ny)
	DLx     = zeros(Float64, 3, nx - 1, ny)
	DRx     = zeros(Float64, 3, nx - 1, ny)

	Aygp    = zeros(Float64, 3, 3, nx, ny - 1)
	Aygpabs = zeros(Float64, 3, 3, nx, ny - 1)
	dpsidsy = zeros(Float64, 3, nx, ny - 1)
	DLy     = zeros(Float64, 3, nx, ny - 1)
	DRy     = zeros(Float64, 3, nx, ny - 1)

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
	#h_i, u_i, v_i = copy(h), copy(u), copy(v)
	# time loop
	time = 0.0
	timeout_reached = false
	t_tic = Base.time()
	while it <= maxiter_t && !timeout_reached
		#t_tic = Base.time()

		# --- Time step ---
		# Compute the time step based on the CFL condition
		dt = compute_dt(nx, ny, _dx, _dy, cfl, u, v, h, gravit)
		#dt = 0.01
		if time + dt > timeout
			dt = timeout - time
			timeout_reached = true
		end
		time += dt

		# Precompute constant factors
		dtdx = dt / dx
		dtdy = dt / dy

		# --- X-direction fluxes (vertical faces) ---
		# Left/right states on cell faces
		@views begin
			# Create views for left/right cell values
			hl = h[1:end-1, :]
			hr = h[2:end, :]
			ul = u[1:end-1, :]
			ur = u[2:end, :]
			vl = v[1:end-1, :]
			vr = v[2:end, :]

			for igp in 1:ngp

				hgp  = @. hl + sgp[igp] * (hr - hl)
				ugp  = @. ul + sgp[igp] * (ur - ul)
				vgp  = @. vl + sgp[igp] * (vr - vl)
				c2gp = @. gravit * hgp

				λ1 = @. (ugp - sqrt(c2gp))
				λ2 = ugp
				λ3 = @. (ugp + sqrt(c2gp))
				process_fluxes_x_merged!(
					dpsidsx, Axgp, Axgpabs,   # output arrays (for differences and matrices)
					DLx, DRx,                # accumulation arrays for flux contributions
					hl, hr, ul, ur, vl, vr,    # views for left/right cell values
					ugp, vgp, c2gp,          # computed cell-centered/interpolated values
					λ1, λ2, λ3,              # eigenvalue arrays
					wgp[igp],                # weight for the current Gaussian point
					nx, ny,                   # grid dimensions
				)

			end

			DLx = @. 0.5 * DLx
			DRx = @. 0.5 * DRx
		end

		@views begin
			# Create views for bottom/top cell values.
			hb = h[:, 1:end-1]
			ht = h[:, 2:end]
			ub = u[:, 1:end-1]
			ut = u[:, 2:end]
			vb = v[:, 1:end-1]
			vt = v[:, 2:end]

			for igp in 1:ngp
				hgp  = @. hb + sgp[igp] * (ht - hb)
				ugp  = @. ub + sgp[igp] * (ut - ub)
				vgp  = @. vb + sgp[igp] * (vt - vb)
				c2gp = @. gravit * hgp

				λ1 = @. vgp - sqrt(c2gp)
				λ2 = vgp
				λ3 = @. vgp + sqrt(c2gp)

				process_fluxes_y_merged!(
					dpsidsy, Aygp, Aygpabs,
					DLy, DRy,
					hb, ht, ub, ut, vb, vt,
					ugp, vgp, c2gp,
					λ1, λ2, λ3,
					wgp[igp], nx, ny,
				)
			end

			DLy .= @. 0.5 * DLy
			DRy .= @. 0.5 * DRy
		end


		# --- Update Step ---
		# Call the update function.
		update!(h, qx, qy, DLx, DRx, DLy, DRy, dtdx, dtdy, nx, ny)
		#update_avx!(h, qx, qy, DLx, DRx, DLy, DRy, dtdx, dtdy, nx, ny)
		#BoundaryConditions

		# Reflective boundary conditions on all 4 walls
		apply_reflective_boundaries!(h, qx, qy)

		#Set fluctuation to zero
		DLx .= 0.0
		DRx .= 0.0
		DLy .= 0.0
		DRy .= 0.0

		# Update velocity safely to avoid division by zero
		# Update velocities from momentum and depth, making sure to avoid division by zero.
		update_velocity!(u, v, qx, qy, h, nx, ny)

		if do_check && it % nvis == 0
			@printf("  time=%.3f, dt=%.3f\n", time, dt)
			display(heatmap(xc, yc, h'; xlims = (xc[1], xc[end]), ylims = (yc[1], yc[end]), aspect_ratio = 1, c = :redsblues, clim = (0.5, 1)))
			#plot_results(xc, yc, h, u, v, lx, ly, time)
		end

		it += 1

	end
	t_toc = Base.time() - t_tic

	A_eff = 3 * 2 * nx * ny * sizeof(Float64) / 1e9
	t_it = t_toc / it
	T_eff = A_eff / t_it
	@printf("Time: %1.3f s, T_eff = %1.3e GB/s, it = %d\n", t_toc, T_eff, it)
	return
end

DOT_2D(; do_check = false)
