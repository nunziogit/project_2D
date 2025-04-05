using Plots, Plots.Measures, Printf
using BenchmarkTools

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

function apply_reflective_boundaries!(h::AbstractArray{Float64},
	qx::AbstractArray{Float64},
	qy::AbstractArray{Float64})
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


@views function compute_dt(nx, ny, _dx, _dy, cfl, u, v, h, gravit)
	max_ratio = 0.0
	# Loop over grid points
	for j in 1:ny
		for i in 1:nx
			# Compute the local wave speed once
			local_speed = abs(u[i, j]) + sqrt(gravit * h[i, j])
			# Multiply by precomputed reciprocals instead of dividing at each point
			r1 = local_speed * _dx
			r2 = local_speed * _dy
			# Update the maximum ratio
			max_ratio = max(max_ratio, r1, r2)
		end
	end
	dt = cfl / max_ratio
	return dt
end

# Define an inline function to compute dpsidsx for a given cell.
function compute_dpsidsx!(dps, hl, hr, ul, ur, vl, vr)
	dps[1] = hr - hl
	dps[2] = ur * hr - ul * hl
	dps[3] = vr * hr - vl * hl
end

function compute_Axgp!(Ax, ugp, vgp, c2gp)
	Ax[1, 1] = 0.0
	Ax[1, 2] = 1.0
	Ax[1, 3] = 0.0
	Ax[2, 1] = c2gp - ugp^2
	Ax[2, 2] = 2.0 * ugp
	Ax[2, 3] = 0.0
	Ax[3, 1] = -ugp * vgp
	Ax[3, 2] = vgp
	Ax[3, 3] = ugp
end

# Define an inline function to compute Axgpabs.
function compute_Axgpabs!(Axabs, λ1, λ2, λ3, vgp)
	absλ1 = abs(λ1)
	absλ3 = abs(λ3)
	den = λ1 - λ3
	invden = 1.0 / den         # Compute reciprocal once
	absden = absλ1 - absλ3

	Axabs[1, 1] = (-absλ1 * λ3 + absλ3 * λ1) * invden
	Axabs[1, 2] = absden * invden
	Axabs[1, 3] = 0.0

	Axabs[2, 1] = -λ1 * λ3 * absden * invden
	Axabs[2, 2] = (λ1 * absλ1 - λ3 * absλ3) * invden
	Axabs[2, 3] = 0.0

	Axabs[3, 1] = -((den * abs(λ2) - absλ3 * λ1 + absλ1 * λ3) * vgp) * invden
	Axabs[3, 2] = vgp * absden * invden
	Axabs[3, 3] = abs(λ2)
end



function compute_fluct_DLx_DRx!(DLx, DRx, dpsidsx, Axgp, Axgpabs, wgp, igp, nx, ny)
	@inbounds for j in 1:ny
		for i in 1:(nx-1)
			# Cache the current slice for dpsidsx and the matrices
			dps   = dpsidsx[:, i, j]    # 3-element vector
			Ax    = Axgp[:, :, i, j]      # 3x3 matrix
			Axabs = Axgpabs[:, :, i, j]   # 3x3 matrix

			# Compute matrix-vector products and accumulate in DLx and DRx
			DLx[:, i, j] .+= wgp[igp] * (Ax - Axabs) * dps
			DRx[:, i, j] .+= wgp[igp] * (Ax + Axabs) * dps
		end
	end
	return nothing
end

function compute_dpsidsy!(dps, hb, ht, ub, ut, vb, vt)
	# dps is a 3-element vector that will be updated in place.
	dps[1] = ht - hb
	dps[2] = ut * ht - ub * hb
	dps[3] = vt * ht - vb * hb
end

function compute_Aygp!(Aygp, ugp, vgp, c2gp)
	# Aygp is a 3x3 matrix (updated in place) for the given cell.
	Aygp[1, 1] = 0.0
	Aygp[1, 2] = 0.0
	Aygp[1, 3] = 1.0
	Aygp[2, 1] = -ugp * vgp
	Aygp[2, 2] = vgp
	Aygp[2, 3] = ugp
	Aygp[3, 1] = c2gp - vgp^2
	Aygp[3, 2] = 0.0
	Aygp[3, 3] = 2.0 * vgp
end

function compute_Aygpabs!(Aygpabs, λ1, λ2, λ3, ugp)
	# Compute common subexpressions.
	absλ1 = abs(λ1)
	absλ3 = abs(λ3)
	den = λ1 - λ3
	invden = 1.0 / den          # reciprocal of den
	absden = absλ1 - absλ3

	# Row 1.
	Aygpabs[1, 1] = (-absλ1 * λ3 + absλ3 * λ1) * invden
	Aygpabs[1, 2] = 0.0
	Aygpabs[1, 3] = absden * invden

	# Row 2.
	Aygpabs[2, 1] = -((den * abs(λ2) - absλ3 * λ1 + absλ1 * λ3) * ugp) * invden
	Aygpabs[2, 2] = abs(λ2)
	Aygpabs[2, 3] = ugp * absden * invden

	# Row 3.
	Aygpabs[3, 1] = -λ1 * λ3 * absden * invden
	Aygpabs[3, 2] = 0.0
	Aygpabs[3, 3] = (λ1 * absλ1 - λ3 * absλ3) * invden
end

function compute_fluct_DLy_DRy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp, igp, nx, ny)
	@inbounds for j in 1:(ny-1)
		for i in 1:nx
			# dps is a 3-element vector for the current cell interface.
			dps = dpsidsy[:, i, j]
			# Ay and Ayabs are the 3x3 matrices.
			Ay    = Aygp[:, :, i, j]
			Ayabs = Aygpabs[:, :, i, j]

			DLy[:, i, j] .+= wgp[igp] * (Ay - Ayabs) * dps
			DRy[:, i, j] .+= wgp[igp] * (Ay + Ayabs) * dps
		end
	end
	return nothing
end

function update!(h, qx, qy, DLx, DRx, DLy, DRy, dtdx, dtdy, nx, ny)
	@inbounds  Threads.@threads for i in 2:(nx-1)
		for j in 2:(ny-1)
			# Compute divergence in x-direction:
			h[i, j]  -= dtdx * (DRx[1, i-1, j] + DLx[1, i, j]) + dtdy * (DRy[1, i, j-1] + DLy[1, i, j])
			qx[i, j] -= dtdx * (DRx[2, i-1, j] + DLx[2, i, j]) + dtdy * (DRy[2, i, j-1] + DLy[2, i, j])
			qy[i, j] -= dtdx * (DRx[3, i-1, j] + DLx[3, i, j]) + dtdy * (DRy[3, i, j-1] + DLy[3, i, j])
		end
	end
end

function update_velocity!(u, v, qx, qy, h, nx, ny)
	@inbounds for i in 1:nx
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
	return nothing #=  =#
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

	# derived numerics
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
	t_tic = 0.0
	while it <= maxiter_t && !timeout_reached
		t_tic = Base.time()

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



				@inbounds for j in 1:ny
					for i in 1:(nx-1)
						# --- dpsidsx computation ---
						compute_dpsidsx!(dpsidsx[:, i, j],
							hl[i, j], hr[i, j],
							ul[i, j], ur[i, j],
							vl[i, j], vr[i, j])
						# Compute Axgp for the current cell.
						compute_Axgp!(Axgp[:, :, i, j], ugp[i, j], vgp[i, j], c2gp[i, j])
						# --- Axgpabs computation ---
						compute_Axgpabs!(Axgpabs[:, :, i, j], λ1[i, j], λ2[i, j], λ3[i, j], vgp[i, j])
					end
				end
				compute_fluct_DLx_DRx!(DLx, DRx, dpsidsx, Axgp, Axgpabs, wgp, igp, nx, ny)
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

				@inbounds for j in 1:(ny-1)
					for i in 1:nx
						# Compute differences for the stencil in y-direction.
						compute_dpsidsy!(dpsidsy[:, i, j],
							hb[i, j], ht[i, j],
							ub[i, j], ut[i, j],
							vb[i, j], vt[i, j])
						# Compute the flux matrix.
						compute_Aygp!(Aygp[:, :, i, j],
							ugp[i, j], vgp[i, j], c2gp[i, j])
						# Compute the absolute flux matrix.
						compute_Aygpabs!(Aygpabs[:, :, i, j],
							λ1[i, j], λ2[i, j], λ3[i, j], ugp[i, j])
					end
				end

				compute_fluct_DLy_DRy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp, igp, nx, ny)
			end

			DLy .= @. 0.5 * DLy
			DRy .= @. 0.5 * DRy
		end


		# --- Update Step ---
		# Call the update function.
		update!(h, qx, qy, DLx, DRx, DLy, DRy, dtdx, dtdy, nx, ny)
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