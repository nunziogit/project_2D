using Plots, Plots.Measures
##default(size = (1200, 400), framestyle = :box, label = false, grid = false, margin = 10mm, lw = 6, labelfontsize = 20, tickfontsize = 20, titlefontsize = 24)
#
## Include the external file with the plotting function
#include("plot_results.jl")
##include("plot_results_3D.jl")


@views function DOT_2D()
	# physics
	lx, ly  = 40.0, 40.0
	gravit  = 9.81
	R       = 2.5 #Radius of the initial circular dam
	hins    = 2.5
	hout    = 1.0
	timeout = 4.0
	# numerics
	nx, ny = 101, 101
	nt = 10000
	ngp = 1  #number of gaussian points
	sgp, wgp = gaussian_points(ngp)

	#nvis = 20
	cfl = 0.49
	# derived numerics
	dx, dy = lx / nx, ly / ny
	xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)

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
	h_i, u_i, v_i = copy(h), copy(u), copy(v)
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
						hrij = hr[i, j]
						hlij = hl[i, j]
						urij = ur[i, j]
						ulij = ul[i, j]
						vrij = vr[i, j]
						vlij = vl[i, j]

						dpsidsx[1, i, j] = hrij - hlij
						dpsidsx[2, i, j] = urij * hrij - ulij * hlij
						dpsidsx[3, i, j] = vrij * hrij - vlij * hlij

						# --- Axgp computation ---
						ugpij  = ugp[i, j]
						vgpij  = vgp[i, j]
						c2gpij = c2gp[i, j]

						Axgp[1, 1, i, j] = 0.0
						Axgp[1, 2, i, j] = 1.0
						Axgp[1, 3, i, j] = 0.0
						Axgp[2, 1, i, j] = c2gpij - ugpij^2
						Axgp[2, 2, i, j] = 2.0 * ugpij
						Axgp[2, 3, i, j] = 0.0
						Axgp[3, 1, i, j] = -ugpij * vgpij
						Axgp[3, 2, i, j] = vgpij
						Axgp[3, 3, i, j] = ugpij

						# --- Axgpabs computation ---
						λ1ij    = λ1[i, j]
						λ3ij    = λ3[i, j]
						λ2ij    = λ2[i, j]
						absλ1ij = abs(λ1ij)
						absλ3ij = abs(λ3ij)
						den     = λ1ij - λ3ij
						absden  = absλ1ij - absλ3ij

						Axgpabs[1, 1, i, j] = (-absλ1ij * λ3ij + absλ3ij * λ1ij) / den
						Axgpabs[1, 2, i, j] = absden / den
						Axgpabs[1, 3, i, j] = 0.0

						Axgpabs[2, 1, i, j] = -λ1ij * λ3ij * absden / den
						Axgpabs[2, 2, i, j] = (λ1ij * absλ1ij - λ3ij * absλ3ij) / den
						Axgpabs[2, 3, i, j] = 0.0

						Axgpabs[3, 1, i, j] = -((den * abs(λ2ij) - absλ3ij * λ1ij + absλ1ij * λ3ij) * vgpij) / den
						Axgpabs[3, 2, i, j] = vgpij * absden / den
						Axgpabs[3, 3, i, j] = abs(λ2ij)
					end
				end

				@inbounds for j in 1:ny
					for i in 1:(nx-1)
						# Cache the current slice for dpsidsx and the matrices
						dps   = dpsidsx[:, i, j]            # 3-element vector
						Ax    = Axgp[:, :, i, j]              # 3x3 matrix
						Axabs = Axgpabs[:, :, i, j]          # 3x3 matrix

						# Compute matrix-vector products and accumulate in DLx and DRx
						DLx[:, i, j] .+= wgp[igp] * (Ax - Axabs) * dps
						DRx[:, i, j] .+= wgp[igp] * (Ax + Axabs) * dps
					end
				end
			end

			DLx = @. 0.5 * DLx
			DRx = @. 0.5 * DRx
		end

		@views begin
			# --- Y-direction fluxes (horizontal faces) ---
			# Bottom/top states on cell faces
			hb = h[:, 1:end-1]
			ht = h[:, 2:end]
			ub = u[:, 1:end-1]
			ut = u[:, 2:end]
			vb = v[:, 1:end-1]
			vt = v[:, 2:end]

			for igp in 1:ngp

				hgp = @. hb + sgp[igp] * (ht - hb)
				ugp = @. ub + sgp[igp] * (ut - ub)
				vgp = @. vb + sgp[igp] * (vt - vb)
				c2gp = @. gravit * hgp

				λ1 = @. (vgp - sqrt(c2gp))
				λ2 = vgp
				λ3 = @. (vgp + sqrt(c2gp))

				@inbounds for j in 1:(ny-1)
					for i in 1:nx

						htij = ht[i, j]
						hbij = hb[i, j]
						utij = ut[i, j]
						ubij = ub[i, j]
						vtij = vt[i, j]
						vbij = vb[i, j]

						dpsidsy[1, i, j] = htij - hbij
						dpsidsy[2, i, j] = utij * htij - ubij * hbij
						dpsidsy[3, i, j] = vtij * htij - vbij * hbij

						# Cache values for Aygp
						ugpij = ugp[i, j]
						vgpij = vgp[i, j]
						c2gpij = c2gp[i, j]

						Aygp[1, 1, i, j] = 0.0
						Aygp[1, 2, i, j] = 0.0
						Aygp[1, 3, i, j] = 1.0
						Aygp[2, 1, i, j] = -ugpij * vgpij
						Aygp[2, 2, i, j] = vgpij
						Aygp[2, 3, i, j] = ugpij
						Aygp[3, 1, i, j] = c2gpij - vgpij^2
						Aygp[3, 2, i, j] = 0.0
						Aygp[3, 3, i, j] = 2.0 * vgpij

						# Precompute common subexpressions for Aygpabs
						λ1ij    = λ1[i, j]
						λ3ij    = λ3[i, j]
						absλ1ij = abs(λ1ij)
						absλ3ij = abs(λ3ij)
						den     = λ1ij - λ3ij
						absden  = absλ1ij - absλ3ij
						λ2ij    = λ2[i, j]

						Aygpabs[1, 1, i, j] = (-absλ1ij * λ3ij + absλ3ij * λ1ij) / den
						Aygpabs[1, 2, i, j] = 0.0
						Aygpabs[1, 3, i, j] = absden / den

						Aygpabs[2, 1, i, j] = -((den * abs(λ2ij) - absλ3ij * λ1ij + absλ1ij * λ3ij) * ugpij) / den
						Aygpabs[2, 2, i, j] = abs(λ2ij)
						Aygpabs[2, 3, i, j] = ugpij * absden / den

						Aygpabs[3, 1, i, j] = -λ1ij * λ3ij * absden / den
						Aygpabs[3, 2, i, j] = 0.0
						Aygpabs[3, 3, i, j] = (λ1ij * absλ1ij - λ3ij * absλ3ij) / den
					end
				end

				@inbounds for j in 1:(ny-1)
					for i in 1:nx
						# Cache the current slices for efficiency
						dps = dpsidsy[:, i, j]          # 3-element vector
						Ay = Aygp[:, :, i, j]           # 3x3 matrix
						Ayabs = Aygpabs[:, :, i, j]       # 3x3 matrix

						# Compute the matrix-vector products and accumulate into DLy and DRy
						DLy[:, i, j] .+= wgp[igp] * (Ay - Ayabs) * dps
						DRy[:, i, j] .+= wgp[igp] * (Ay + Ayabs) * dps
					end
				end
			end

			DLy = @. 0.5 * DLy
			DRy = @. 0.5 * DRy
		end



		# --- Update Step ---
		# Update the interior cells with the divergence of the fluxes.
		# For the x-direction, compute differences along dimension 1,
		# for the y-direction, differences along dimension 2.
		@inbounds for i in 2:(nx-1)
			for j in 2:(ny-1)
				# Compute divergence in x-direction:
				h[i, j]  -= dtdx * (DRx[1, i-1, j] + DLx[1, i, j]) + dtdy * (DRy[1, i, j-1] + DLy[1, i, j])
				qx[i, j] -= dtdx * (DRx[2, i-1, j] + DLx[2, i, j]) + dtdy * (DRy[2, i, j-1] + DLy[2, i, j])
				qy[i, j] -= dtdx * (DRx[3, i-1, j] + DLx[3, i, j]) + dtdy * (DRy[3, i, j-1] + DLy[3, i, j])
			end
		end
		#BoundaryConditions

		# Reflective boundary conditions on all 4 walls

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

		#Set fluctuation to zero
		DLx .= 0.0
		DRx .= 0.0
		DLy .= 0.0
		DRy .= 0.0

		# Update velocity safely to avoid division by zero
		#u .= @. ifelse(h > 1e-12, qx / h, 0.0)
		#v .= @. ifelse(h > 1e-12, qy / h, 0.0)
		# Update velocities from momentum and depth, making sure to avoid division by zero.
		for i in 1:(nx)
			for j in 1:(ny)
				if h[i, j] > 1e-12
					u[i, j] = qx[i, j] / h[i, j]
					v[i, j] = qy[i, j] / h[i, j]
				else
					u[i, j] = 0.0
					v[i, j] = 0.0
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
	plot_results(xc, yc, h, u, v, lx, ly, timeout)
	# Call the plot_results_3D function to visualize the results in 3D
	#plot_results_3D(xc, yc, h, timeout)
	#anim = plot_results_3D(xc, yc, h, timeout)
	#gif(anim, "results.gif", fps = 15)


end


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

DOT_2D()

# Example usage:
# Create a dam-break along the x-direction
#set_dambreak!(h, hins, hout; direction=:x)

# Create a dam-break along the y-direction
#set_dambreak!(h, hins, hout; direction=:y)

#using ProfileView
#@profview DOT_2D()