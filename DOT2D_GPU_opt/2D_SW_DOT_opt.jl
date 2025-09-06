using CUDA, Revise

#CUDA.device()         # stampa il device attivo
#CUDA.versioninfo()    # stampa info su CUDA, driver e GPU

using Plots, Plots.Measures, Printf#, ProfileView

#default(size = (1200, 400), framestyle = :box, label = false, grid = false, margin = 10mm, lw = 2, labelfontsize = 6, tickfontsize = 6, titlefontsize = 8)
#
## Include the external file with the plotting function
include("plot_results.jl")

function parameters(nx, ny)
    # Physics
    lx, ly  = 40.0, 40.0 # dominio fisco
    gravit  = 9.81
    R       = 2.5 # Radius of the initial circular dam → dambreak circolare
    hins    = 2.5 # profondità dentro la diga
    hout    = 1.0 # profondità fuori la diga
    timeout = 2.0 # tempo di uscita

    # Numerics
    nt  = 1000 # numero di step temporali
    ngp = 2  #number of gaussian points
    cfl = 0.49 # Courant number per il 2D è max 0.5

    # Derived numerics
    dx, dy   = lx / nx, ly / ny
    _dx, _dy = 1.0 / dx, 1.0 / dy
    return (; nx, ny, lx, ly, gravit, R, hins, hout, timeout, nt, ngp, cfl, dx, dy, _dx, _dy)
end

@inbounds @views function IC!(h, u, v, qx, qy, xc, yc, p, nx, ny)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    # Initial conditions
    # Set h: use hins for points inside the square, and hout otherwise
	cx = 0.5 * p.lx
	cy = 0.5 * p.ly
	if i>=1 && i <= nx && j>=1 && j <= ny
		D = sqrt((xc[i] - cx)^2 + (yc[j] -cy)^2)
		h[i, j] = D <= p.R ? p.hins : p.hout
		#set_dambreak!(h, p.hins, p.hout; direction = :y)
		qx[i, j] = u[i, j] * h[i, j]
		qy[i, j] = v[i, j] * h[i, j]
	end
    return nothing
end

@inbounds @views function compute_dtl(dtlx, dtly, h, u, v, p, nx, ny)
	gravit = p.gravit
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	if i>=1 && i <= nx && j>=1 && j <= ny
        h_ij = max(h[i, j], 1e-12)
		λx   = abs(u[i, j]) + sqrt(gravit * h_ij)
		λy   = abs(v[i, j]) + sqrt(gravit * h_ij)
		dtlx[i, j] = p.dx / λx
		dtly[i, j] = p.dy / λy
	end
	return nothing
end

# Moltiplicazione matrice vettore (3x3 * 3x1)
@inbounds @views function matvec(A11,A12,A13,A21,A22,A23,A31,A32,A33, v)
    return (
        A11*v[1] + A12*v[2] + A13*v[3],
        A21*v[1] + A22*v[2] + A23*v[3],
        A31*v[1] + A32*v[2] + A33*v[3],
    )
end

@inbounds @views function compute_xfluxes!(DLx, DRx, h, u, v, sgp, wgp, p, nx, ny)
	gravit=p.gravit;	ngp=p.ngp
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= nx-1 && j >= 1 && j <= ny
        for igp in 1:ngp
            # Calcolo valori hl, hr, ul, ur, vl, vr (a sinistra/destra)
            hl = h[i, j];       hr = h[i+1, j]
            ul = u[i, j];       ur = u[i+1, j]
            vl = v[i, j];       vr = v[i+1, j]

            # Calcolo hgp, ugp, vgp nel punto di Gauss
            hgp  = hl + sgp[igp] * (hr - hl)
            ugp  = ul + sgp[igp] * (ur - ul)
            vgp  = vl + sgp[igp] * (vr - vl)
            c2gp = gravit * hgp

            # Autovalori
            λ1 = ugp - sqrt(c2gp);    λ2 = ugp;    λ3 = ugp + sqrt(c2gp)

            # Salti
            dps1 = hr - hl
            dps2 = ur * hr - ul * hl
            dps3 = vr * hr - vl * hl

            # Matrice Axgp (3x3)
            Axgp_11 = 0.0
            Axgp_12 = 1.0
            Axgp_13 = 0.0

            Axgp_21 = c2gp - ugp^2
            Axgp_22 = 2.0 * ugp
            Axgp_23 = 0.0

            Axgp_31 = -ugp * vgp
            Axgp_32 = vgp
            Axgp_33 = ugp

            # Matrice Axgpabs (3x3)
            absλ1 = abs(λ1);    absλ2 = abs(λ2);    absλ3 = abs(λ3)
            den = λ1 - λ3
            _den = 1.0 / den
            absden = absλ1 - absλ3

            Axgpabs_11 = (-absλ1 * λ3 + absλ3 * λ1) * _den
            Axgpabs_12 = absden * _den
            Axgpabs_13 = 0.0

            Axgpabs_21 = -λ1 * λ3 * absden * _den
            Axgpabs_22 = (λ1 * absλ1 - λ3 * absλ3) * _den
            Axgpabs_23 = 0.0

            Axgpabs_31 = -((den * absλ2 - absλ3 * λ1 + absλ1 * λ3) * vgp) * _den
            Axgpabs_32 = vgp * absden * _den
            Axgpabs_33 = absλ2

            # Calcolo prodotti matriciali (Ax - Axabs)*dps e (Ax + Axabs)*dps
            # dps vettore 3x1
            dps = (dps1, dps2, dps3)

            Ax_minus_Axabs = (
                Axgp_11 - Axgpabs_11, Axgp_12 - Axgpabs_12, Axgp_13 - Axgpabs_13,
                Axgp_21 - Axgpabs_21, Axgp_22 - Axgpabs_22, Axgp_23 - Axgpabs_23,
                Axgp_31 - Axgpabs_31, Axgp_32 - Axgpabs_32, Axgp_33 - Axgpabs_33,
            )

            Ax_plus_Axabs = (
                Axgp_11 + Axgpabs_11, Axgp_12 + Axgpabs_12, Axgp_13 + Axgpabs_13,
                Axgp_21 + Axgpabs_21, Axgp_22 + Axgpabs_22, Axgp_23 + Axgpabs_23,
                Axgp_31 + Axgpabs_31, Axgp_32 + Axgpabs_32, Axgp_33 + Axgpabs_33,
            )

            # Moltiplico e accumulo in DLx e DRx (in place)
            resL = matvec(Ax_minus_Axabs[1],Ax_minus_Axabs[2],Ax_minus_Axabs[3],
                          Ax_minus_Axabs[4],Ax_minus_Axabs[5],Ax_minus_Axabs[6],
                          Ax_minus_Axabs[7],Ax_minus_Axabs[8],Ax_minus_Axabs[9], dps)
            resR = matvec(Ax_plus_Axabs[1],Ax_plus_Axabs[2],Ax_plus_Axabs[3],
                          Ax_plus_Axabs[4],Ax_plus_Axabs[5],Ax_plus_Axabs[6],
                          Ax_plus_Axabs[7],Ax_plus_Axabs[8],Ax_plus_Axabs[9], dps)

            @inbounds for k in 1:3
                DLx[k, i, j] += 0.5 * wgp[igp] * resL[k]
                DRx[k, i, j] += 0.5 * wgp[igp] * resR[k]
            end
        end
    end
    return nothing
end

@inbounds @views function compute_yfluxes!(DLy, DRy, h, u, v, sgp, wgp, p, nx, ny)
    gravit = p.gravit;      ngp = p.ngp
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i >= 1 && i <= nx && j >= 1 && j <= ny-1
        for igp in 1:ngp
            # Valori inferiori e superiori lungo y (facce orizzontali)
            hb = h[i, j];       ht = h[i, j+1]
            ub = u[i, j];       ut = u[i, j+1]
            vb = v[i, j];       vt = v[i, j+1]

            # Valori interpolati al punto di Gauss
            hgp  = hb + sgp[igp] * (ht - hb)
            ugp  = ub + sgp[igp] * (ut - ub)
            vgp  = vb + sgp[igp] * (vt - vb)
            c2gp = gravit * hgp

            # Autovalori (in y)
            λ1 = vgp - sqrt(c2gp)
            λ2 = vgp
            λ3 = vgp + sqrt(c2gp)

            # Salti
            dps1 = ht - hb
            dps2 = ut * ht - ub * hb
            dps3 = vt * ht - vb * hb

            # Matrice Aygp (3x3)
            Aygp_11 = 0.0
            Aygp_12 = 0.0
            Aygp_13 = 1.0

            Aygp_21 = -ugp * vgp
            Aygp_22 = vgp
            Aygp_23 = ugp

            Aygp_31 = c2gp - vgp^2
            Aygp_32 = 0.0
            Aygp_33 = 2.0 * vgp

            # Matrice Aygpabs (3x3)
            absλ1 = abs(λ1)
            absλ3 = abs(λ3)
            den = λ1 - λ3
            _den = 1.0 / den
            absden = absλ1 - absλ3

            Aygpabs_11 = (-absλ1 * λ3 + absλ3 * λ1) * _den
            Aygpabs_12 = 0.0
            Aygpabs_13 = absden * _den

            Aygpabs_21 = -((den * abs(λ2) - absλ3 * λ1 + absλ1 * λ3) * ugp) * _den
            Aygpabs_22 = abs(λ2)
            Aygpabs_23 = ugp * absden * _den

            Aygpabs_31 = -λ1 * λ3 * absden * _den
            Aygpabs_32 = 0.0
            Aygpabs_33 = (λ1 * absλ1 - λ3 * absλ3) * _den

            # Vettore salti
            dps = (dps1, dps2, dps3)

            Ay_minus_Axabs = (
                Aygp_11 - Aygpabs_11, Aygp_12 - Aygpabs_12, Aygp_13 - Aygpabs_13,
                Aygp_21 - Aygpabs_21, Aygp_22 - Aygpabs_22, Aygp_23 - Aygpabs_23,
                Aygp_31 - Aygpabs_31, Aygp_32 - Aygpabs_32, Aygp_33 - Aygpabs_33,
            )

            Ay_plus_Axabs = (
                Aygp_11 + Aygpabs_11, Aygp_12 + Aygpabs_12, Aygp_13 + Aygpabs_13,
                Aygp_21 + Aygpabs_21, Aygp_22 + Aygpabs_22, Aygp_23 + Aygpabs_23,
                Aygp_31 + Aygpabs_31, Aygp_32 + Aygpabs_32, Aygp_33 + Aygpabs_33,
            )

            # Moltiplicazione matrice-vettore (riutilizzo funzione matvec)
            resL = matvec(Ay_minus_Axabs[1],Ay_minus_Axabs[2],Ay_minus_Axabs[3],
                          Ay_minus_Axabs[4],Ay_minus_Axabs[5],Ay_minus_Axabs[6],
                          Ay_minus_Axabs[7],Ay_minus_Axabs[8],Ay_minus_Axabs[9], dps)
            resR = matvec(Ay_plus_Axabs[1],Ay_plus_Axabs[2],Ay_plus_Axabs[3],
                          Ay_plus_Axabs[4],Ay_plus_Axabs[5],Ay_plus_Axabs[6],
                          Ay_plus_Axabs[7],Ay_plus_Axabs[8],Ay_plus_Axabs[9], dps)

            @inbounds for k in 1:3
                DLy[k, i, j] += 0.5 * wgp[igp] * resL[k]
                DRy[k, i, j] += 0.5 * wgp[igp] * resR[k]
            end
        end
    end

    return nothing
end

@inbounds @views function update_h_qx_qy!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy, p, nx, ny)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	if i >= 2 && i <= nx-1 && j >= 2 && j <= ny-1
		@inbounds begin
			# Compute divergence in x-direction:
			h[i, j]  -= dtdx * (DRx[1, i-1, j] + DLx[1, i, j]) + dtdy * (DRy[1, i, j-1] + DLy[1, i, j])
			qx[i, j] -= dtdx * (DRx[2, i-1, j] + DLx[2, i, j]) + dtdy * (DRy[2, i, j-1] + DLy[2, i, j])
			qy[i, j] -= dtdx * (DRx[3, i-1, j] + DLx[3, i, j]) + dtdy * (DRy[3, i, j-1] + DLy[3, i, j])
		end
	end
	return nothing
end

@inbounds @views function BC!(h, qx, qy, p, nx, ny)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	# Reflective boundary conditions on all 4 walls
	
	@inbounds begin
        # Left wall (x = 1)
        if i == 1 && 1 <= j <= ny
            h[1, j]  =  h[2, j]
            qx[1, j] = -qx[2, j]
            qy[1, j] =  qy[2, j]
        end

        # Right wall (x = nx)
        if i == nx && 1 <= j <= ny
            h[nx, j]  =  h[nx-1, j]
            qx[nx, j] = -qx[nx-1, j]
            qy[nx, j] =  qy[nx-1, j]
        end

        # Bottom wall (y = 1)
        if j == 1 && 1 <= i <= nx
            h[i, 1]  =  h[i, 2]
            qx[i, 1] =  qx[i, 2]
            qy[i, 1] = -qy[i, 2]
        end

        # Top wall (y = ny)
        if j == ny && 1 <= i <= nx
            h[i, ny]  =  h[i, ny-1]
            qx[i, ny] =  qx[i, ny-1]
            qy[i, ny] = -qy[i, ny-1]
        end
    end
	return nothing
end

@inbounds @views function update_uv!(h, u, v, qx, qy, p, nx, ny)
	i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
	j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
	if i >= 1 && i <= nx && j >= 1 && j <= ny
		@inbounds begin
			if h[i, j] > 1e-12 # evito divisioni per zero
				_hij    = 1.0/h[i, j]
				u[i, j] = qx[i, j] * _hij
				v[i, j] = qy[i, j] * _hij
			else
				u[i, j] = 0.0
				v[i, j] = 0.0
			end
		end
	end
	return nothing
end

@inbounds @views function DOT_2D()
	# Parameters
    nx = ny = 128 # → numero di celle
    threads = (32, 16)
	blocks = (nx÷threads[1], ny÷threads[2])
    # configurazioni ottimali (32, 1)
    p = parameters(nx, ny)
	xc, yc = LinRange(0.5 * p.dx, p.lx - 0.5 * p.dx, nx), LinRange(0.5 * p.dy, p.ly - 0.5 * p.dy, ny) # baricentri delle celle
	sgp, wgp = gaussian_points(p.ngp) # funzione per i punti di Gauss e i pesi

	# Array initialisation
	h  = CUDA.zeros(Float64, nx, ny)
	u  = CUDA.zeros(Float64, nx, ny)
	v  = CUDA.zeros(Float64, nx, ny)
	qx = CUDA.zeros(Float64, nx, ny)
	qy = CUDA.zeros(Float64, nx, ny)

	# Initial conditions
    CUDA.@sync @cuda blocks=blocks threads=threads IC!(h, u, v, qx, qy, xc, yc, p, nx, ny)
	# Array initialisation
	# termini eq. 15 per x
	DLx     = CUDA.zeros(Float64, 3, nx - 1, ny)
	DRx     = CUDA.zeros(Float64, 3, nx - 1, ny)
	# termini eq. 15 per y
	DLy     = CUDA.zeros(Float64, 3, nx, ny - 1)
	DRy     = CUDA.zeros(Float64, 3, nx, ny - 1)

	# time loop
	time = 0.0
	timeout_reached = false
	dtlx, dtly = CUDA.zeros(Float64, nx, ny), CUDA.zeros(Float64, nx, ny)
	@time for it ∈ 1:p.nt 
		# set CFL condition
		CUDA.@sync @cuda blocks=blocks threads=threads compute_dtl(dtlx, dtly, h, u, v, p, nx, ny)
		dtl = min(CUDA.reduce(min, dtlx), CUDA.reduce(min, dtly))
		dt  = p.cfl * dtl

		# controllo dt all'ultimo passo
		if time + dt > p.timeout
			dt = p.timeout - time
			timeout_reached = true
		end
		time += dt

		# Precompute constant factors
		dtdx = dt * p._dx
		dtdy = dt * p._dy

		# --- X-direction fluxes (vertical faces) ---
		# Left/right states on cell faces
		CUDA.@sync @cuda blocks=blocks threads=threads compute_xfluxes!(DLx, DRx, h, u, v, CuArray(sgp), CuArray(wgp), p, nx, ny)
		CUDA.@sync @cuda blocks=blocks threads=threads compute_yfluxes!(DLy, DRy, h, u, v, CuArray(sgp), CuArray(wgp), p, nx, ny)

		# --- Update Step ---
		# Update the interior cells with the divergence of the fluxes.
		# For the x-direction, compute differences along dimension 1,
		# for the y-direction, differences along dimension 2.

		CUDA.@sync @cuda blocks=blocks threads=threads update_h_qx_qy!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy, p, nx, ny)
        #Set fluctuation to zero
		DLx .= 0.0
		DRx .= 0.0
		DLy .= 0.0
		DRy .= 0.0
		#BoundaryConditions

		CUDA.@sync @cuda blocks=blocks threads=threads BC!(h, qx, qy, p, nx, ny)

		# Update velocities from momentum and depth, making sure to avoid division by zero.
		CUDA.@sync @cuda blocks=blocks threads=threads update_uv!(h, u, v, qx, qy, p, nx, ny)
		# Exit the loop if timeout_reached is true
		if timeout_reached
			println("Timeout reached")
			break
		end
	end
	# Call the plot_results function after the loop
	#@show (u)

	plot_results(xc, yc, Array(h), Array(u), Array(v), p.lx, p.ly, p.timeout)
	# Call the plot_results_3D function to visualize the results in 3D
	#plot_results_3D(xc, yc, h, timeout)
	#anim = plot_results_3D(xc, yc, h, timeout)
	#gif(anim, "results.gif", fps = 15)


end

# punti di gauss
function gaussian_points(ngp::Int)
	if ngp == 1
		sgp = [0.5]
		wgp = [1.0]
	elseif ngp == 2
		sgp = [0.5 - sqrt(3) / 6, 0.5 + sqrt(3) / 6]
		wgp = [0.5, 0.5]
	elseif ngp == 3
		sgp = [0.5 - sqrt(15) * 0.1, 0.5, 0.5 + sqrt(15) * 0.1]
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
# @profview DOT_2D()