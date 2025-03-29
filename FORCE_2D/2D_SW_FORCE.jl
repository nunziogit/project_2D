using Plots, Plots.Measures
##default(size = (1200, 400), framestyle = :box, label = false, grid = false, margin = 10mm, lw = 6, labelfontsize = 20, tickfontsize = 20, titlefontsize = 24)
#
## Include the external file with the plotting function
include("plot_results.jl")
##include("plot_results_3D.jl")


@views function FORCE_2D()
	# physics
	lx, ly = 40, 40.0
	gravit = 9.81
	R      = 2.5 #Radius of the initial circular dam
	hins   = 2.5
	hout   = 0.5
	#uins    = 0.0
	#vins    = 0.0
	#uout    = 0.0
	#vout    = 0.0
	timeout = 1.8
	# numerics
	nx, ny = 101, 101
	nt = 10000
	#nvis = 20
	cfl = 0.45
	# derived numerics
	dx, dy = lx / nx, ly / ny
	xc, yc = LinRange(dx / 2, lx - dx / 2, nx), LinRange(dy / 2, ly - dy / 2, ny)

	g05 = 0.5gravit
	# array initialisation
	h = zeros(Float64, nx, ny)
	u = zeros(Float64, nx, ny)
	v = zeros(Float64, nx, ny)
	# For x-direction fluxes (vertical faces), we need a flux defined on a grid of size (nx-1, ny)
	Fx1 = zeros(Float64, nx - 1, ny)
	Fx2 = zeros(Float64, nx - 1, ny)
	Fx3 = zeros(Float64, nx - 1, ny)

	Fx1LF = zeros(Float64, nx - 1, ny)
	Fx2LF = zeros(Float64, nx - 1, ny)
	Fx3LF = zeros(Float64, nx - 1, ny)
	Fx1LW = zeros(Float64, nx - 1, ny)
	Fx2LW = zeros(Float64, nx - 1, ny)
	Fx3LW = zeros(Float64, nx - 1, ny)


	# For y-direction fluxes (horizontal faces), we need a flux defined on a grid of size (nx, ny-1)
	Fy1 = zeros(Float64, nx, ny - 1)
	Fy2 = zeros(Float64, nx, ny - 1)
	Fy3 = zeros(Float64, nx, ny - 1)

	Fy1LF = zeros(Float64, nx, ny - 1)
	Fy2LF = zeros(Float64, nx, ny - 1)
	Fy3LF = zeros(Float64, nx, ny - 1)
	Fy1LW = zeros(Float64, nx, ny - 1)
	Fy2LW = zeros(Float64, nx, ny - 1)
	Fy3LW = zeros(Float64, nx, ny - 1)


	cx = 0.5lx
	cy = 0.5ly

	# Reshape xc and yc to form a grid
	#xc_col = reshape(xc, :, 1)   # column vector
	#yc_row = reshape(yc, 1, :)   # row vector
	#
	## Create a Boolean mask for points inside the square
	#in_square = (abs.(xc_col .- cx) .<= R / 2) .& (abs.(yc_row .- cy) .<= R / 2)
	#
	## Set h: use hins for points inside the square, and hout otherwise
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


	#@views (h)

	qx = @. (u * h)
	qy = @. (v * h)
	h_i, u_i, v_i = copy(h), copy(u), copy(v)
	# time loop
	time = 0.0
	timeout_reached = false
	@views for it ∈ 1:nt
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
		dxdt025 = 0.25(dx / dt)  # assuming dx_dt = dx/dt, similarly for dy
		dydt025 = 0.25(dy / dt)


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

			# Compute flux-related intermediate quantities
			qxl  = hl .* ul
			qxr  = hr .* ur
			qyl  = hl .* vl
			qyr  = hr .* vr
			qxyl = hl .* ul .* vl
			qxyr = hr .* ur .* vr

			# Compute Lax-Friedrichs (LF) flux components on x-faces
			Fx1LF = @. 0.5 * (qxl + qxr) - dxdt025 * (hr - hl)
			Fx2LF = @. 0.5 * ((ul^2) * hl + g05 * (hl^2) + (ur^2) * hr + g05 * (hr^2)) - dxdt025 * (qxr - qxl)
			Fx3LF = @. 0.5 * (qxyl + qxyr) - dxdt025 * (qyr - qyl)

			# Compute Lax-Wendroff (LW) flux components on x-faces
			hLW  = @. 0.5 * (hl + hr) - dtdx * (qxr - qxl)
			qxLW = @. 0.5 * (ul * hl + ur * hr) - dtdx * (((ur^2) * hr + g05 * (hr^2)) - ((ul^2) * hl + g05 * (hl^2)))
			qyLW = @. 0.5 * (vl * hl + vr * hr) - dtdx * (ur * vr * hr - ul * vl * hl)

			uLW = @. qxLW / hLW
			vLW = @. qyLW / hLW

			Fx1LW = @. hLW * uLW
			Fx2LW = @. (uLW^2) * hLW + g05 * (hLW^2)
			Fx3LW = @. uLW * vLW * hLW

			# Average the LF and LW fluxes for a blended scheme
			Fx1 = @. 0.5 * (Fx1LF + Fx1LW)
			Fx2 = @. 0.5 * (Fx2LF + Fx2LW)
			Fx3 = @. 0.5 * (Fx3LF + Fx3LW)
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

			# Compute intermediate quantities for flux calculations
			qxb  = hb .* ub
			qxt  = ht .* ut
			qyb  = hb .* vb
			qyt  = ht .* vt
			qxyb = hb .* ub .* vb
			qxyt = ht .* ut .* vt

			# Lax-Friedrichs (LF) flux components on y-faces
			Fy1LF = @. 0.5 * (qyb + qyt) - dydt025 * (ht - hb)
			Fy2LF = @. 0.5 * (qxyb + qxyt) - dydt025 * (qxt - qxb)
			Fy3LF = @. 0.5 * ((vb^2) * hb + g05 * (hb^2) + (vt^2) * ht + g05 * (ht^2)) - dydt025 * (qyt - qyb)

			# Lax-Wendroff (LW) flux components on y-faces
			hLW  = @. 0.5 * (hb + ht) - dtdy * (qyt - qyb)
			qxLW = @. 0.5 * (ub * hb + ut * ht) - dtdy * (qxyt - qxyb)
			qyLW = @. 0.5 * (vb * hb + vt * ht) - dtdy * ((vt^2) * ht + g05 * (ht^2) - (vb^2) * hb - g05 * (hb^2))

			uLW = @. qxLW / hLW
			vLW = @. qyLW / hLW

			Fy1LW = @. hLW * vLW
			Fy2LW = @. uLW * vLW * hLW
			Fy3LW = @. (vLW^2) * hLW + g05 * (hLW^2)

			# Blend LF and LW fluxes for a combined scheme
			Fy1 = @. 0.5 * (Fy1LF + Fy1LW)
			Fy2 = @. 0.5 * (Fy2LF + Fy2LW)
			Fy3 = @. 0.5 * (Fy3LF + Fy3LW)
		end
		# --- Update Step ---
		# Update the interior cells with the divergence of the fluxes.
		# For the x-direction, compute differences along dimension 1,
		# for the y-direction, differences along dimension 2.
		@inbounds for i in 2:(nx-1)
			for j in 2:(ny-1)
				# Compute divergence in x-direction:
				# For cell (i,j) the right face is at Fx[ i, j ] and the left face is at Fx[ i-1, j ].
				flux_div_x_h = (Fx1[i, j] - Fx1[i-1, j]) / dx
				flux_div_y_h = (Fy1[i, j] - Fy1[i, j-1]) / dy
				h[i, j] -= dt * (flux_div_x_h + flux_div_y_h)

				flux_div_x_qx = (Fx2[i, j] - Fx2[i-1, j]) / dx
				flux_div_y_qx = (Fy2[i, j] - Fy2[i, j-1]) / dy
				qx[i, j] -= dt * (flux_div_x_qx + flux_div_y_qx)

				flux_div_x_qy = (Fx3[i, j] - Fx3[i-1, j]) / dx
				flux_div_y_qy = (Fy3[i, j] - Fy3[i, j-1]) / dy
				qy[i, j] -= dt * (flux_div_x_qy + flux_div_y_qy)
			end
		end
		#BoundaryConditions
		#h[:, 1] .= h[:, 2]
		#h[:, end] .= h[:, end-1]
		#qx[:, 1] .= qx[:, 2]
		#qx[:, end] .= qx[:, end-1]
		#qy[:, 1] .= qy[:, 2]
		#qy[:, end] .= qy[:, end-1]

		#h[1, :] .= h[2, :]
		#h[end, :] .= h[end-1, :]
		#qx[1, :] .= qx[2, :]
		#qx[end, :] .= qx[end-1, :]
		#qy[1, :] .= qy[2, :]
		#qy[end, :] .= qy[end-1, :]

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




		#		h[2:end-1, 2:end-1]  .-= (
		#		dtdx .* (Fx1[2:end-1, 2:end-1] .- Fx1[1:end-2, 2:end-1]) .+
		#		dtdy .* (Fy1[2:end-1, 2:end-1] .- Fy1[2:end-1, 1:end-2])
		#)
		#		qx[2:end-1, 2:end-1] .-= (
		#		dtdx .* (Fx2[2:end-1, 2:end-1] .- Fx2[1:end-2, 2:end-1]) .+
		#		dtdy .* (Fy2[2:end-1, 2:end-1] .- Fy2[2:end-1, 1:end-2])
		#)
		#		qy[2:end-1, 2:end-1] .-= (
		#		dtdx .* (Fx3[2:end-1, 2:end-1] .- Fx3[1:end-2, 2:end-1]) .+
		#		dtdy .* (Fy3[2:end-1, 2:end-1] .- Fy3[2:end-1, 1:end-2])
		#)

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

# Example usage:
# Create a dam-break along the x-direction
#set_dambreak!(h, hins, hout; direction=:x)

# Create a dam-break along the y-direction
#set_dambreak!(h, hins, hout; direction=:y)


FORCE_2D()
