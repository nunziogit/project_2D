using IJulia, Revise, Printf, Infiltrator 

function xflux!(dpsidsx, Axgp, Axgpabs, DLx, DRx, h, u, v, hgp, ugp, vgp, wgp, threads, blocks)
    @cuda blocks=blocks threads=threads dpsidsx!(dpsidsx, h, u, v); synchronize()
    @cuda blocks=blocks threads=threads Axgp!(Axgp, hgp, ugp, vgp); synchronize()
    @cuda blocks=blocks threads=threads Axgpabs!(Axgpabs, hgp, ugp, vgp); synchronize()
    @cuda blocks=blocks threads=threads Dx!(DLx, DRx, dpsidsx, Axgp, Axgpabs, wgp); synchronize()
end

function dpsidsx!(dpsidsx, h, u, v)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (ix≥1 && ix<size(h,1))
        dpsidsx[1, ix, iy] = h[ix+1, iy] - h[ix,   iy]
		dpsidsx[2, ix, iy] = u[ix+1, iy] * h[ix+1, iy] - u[ix, iy] * h[ix, iy]
		dpsidsx[3, ix, iy] = v[ix+1, iy] * h[ix+1, iy] - v[ix, iy] * h[ix, iy]
    end
    return
end

function Axgp!(Axgp, hgp, ugp, vgp)
    gravit = PARAMETERS.gravit
    # abbiamo messo gp per ricordarci che non ci sono tutti i punti ma 
    # ne manca uno in x
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    # c2gp = @. gravit * hgp
    if (ix≥1 && ix≤size(hgp,1))
        Axgp[1, 1, ix, iy] = 0.0
	    Axgp[1, 2, ix, iy] = 1.0
	    Axgp[1, 3, ix, iy] = 0.0
	    Axgp[2, 1, ix, iy] = gravit*hgp[ix, iy] - ugp[ix, iy]^2
	    Axgp[2, 2, ix, iy] = 2.0 * ugp[ix, iy]
	    Axgp[2, 3, ix, iy] = 0.0
	    Axgp[3, 1, ix, iy] = -ugp[ix, iy] * vgp[ix, iy]
	    Axgp[3, 2, ix, iy] = vgp[ix, iy]
	    Axgp[3, 3, ix, iy] = ugp[ix, iy]
    end
    return
end

function Axgpabs!(Axgpabs, hgp, ugp, vgp)
    gravit = PARAMETERS.gravit
    # abbiamo messo gp per ricordarci che non ci sono tutti i punti ma 
    # ne manca uno in x
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if (ix≥1 && ix≤size(hgp,1))
        den     = (ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
    	_den    = 1/den
        absden  = abs(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - abs(ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
        
        Axgpabs[1, 1, ix, iy] = (-abs(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])) + 
                                abs(ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * (ugp[ix, iy] - sqrt(gravit*hgp[ix, iy]))) * _den
		Axgpabs[1, 2, ix, iy] = absden * _den
		Axgpabs[1, 3, ix, iy] = 0.0

		Axgpabs[2, 1, ix, iy] = -(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * absden * _den
		Axgpabs[2, 2, ix, iy] = ((ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * abs(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - 
                                (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * abs(ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))) * _den
		Axgpabs[2, 3, ix, iy] = 0.0

		Axgpabs[3, 1, ix, iy] = -((den * abs(ugp[ix, iy]) - abs(ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * 
                                (ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) + abs(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * 
                                (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))) * vgp[ix, iy]) * _den
		Axgpabs[3, 2, ix, iy] = vgp[ix, iy] * absden * _den
		Axgpabs[3, 3, ix, iy] = abs(ugp[ix, iy])
    end
    return
end

function Dx!(DLx, DRx, dpsidsx, Axgp, Axgpabs, wgp)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (ix≥1 && ix≤size(dpsidsx,2))
        DLx[1, ix, iy] = 0.5* wgp*( ((Axgp[1, 1, ix, iy] - Axgpabs[1, 1, ix, iy]) * dpsidsx[1, ix, iy]) + 
                                    ((Axgp[1, 2, ix, iy] - Axgpabs[1, 2, ix, iy]) * dpsidsx[2, ix, iy]) +
                                    ((Axgp[1, 3, ix, iy] - Axgpabs[1, 3, ix, iy]) * dpsidsx[3, ix, iy]) )
        DLx[2, ix, iy] = 0.5* wgp*( ((Axgp[2, 1, ix, iy] - Axgpabs[2, 1, ix, iy]) * dpsidsx[1, ix, iy]) + 
                                    ((Axgp[2, 2, ix, iy] - Axgpabs[2, 2, ix, iy]) * dpsidsx[2, ix, iy]) +
                                    ((Axgp[2, 3, ix, iy] - Axgpabs[2, 3, ix, iy]) * dpsidsx[3, ix, iy]) )
        DLx[3, ix, iy] = 0.5* wgp*( ((Axgp[3, 1, ix, iy] - Axgpabs[3, 1, ix, iy]) * dpsidsx[1, ix, iy]) + 
                                    ((Axgp[3, 2, ix, iy] - Axgpabs[3, 2, ix, iy]) * dpsidsx[2, ix, iy]) +
                                    ((Axgp[3, 3, ix, iy] - Axgpabs[3, 3, ix, iy]) * dpsidsx[3, ix, iy]) )

        DRx[1, ix, iy] = 0.5* wgp*( ((Axgp[1, 1, ix, iy] + Axgpabs[1, 1, ix, iy]) * dpsidsx[1, ix, iy]) + 
                                    ((Axgp[1, 2, ix, iy] + Axgpabs[1, 2, ix, iy]) * dpsidsx[2, ix, iy]) +
                                    ((Axgp[1, 3, ix, iy] + Axgpabs[1, 3, ix, iy]) * dpsidsx[3, ix, iy]) )
        DRx[2, ix, iy] = 0.5* wgp*( ((Axgp[2, 1, ix, iy] + Axgpabs[2, 1, ix, iy]) * dpsidsx[1, ix, iy]) + 
                                    ((Axgp[2, 2, ix, iy] + Axgpabs[2, 2, ix, iy]) * dpsidsx[2, ix, iy]) +
                                    ((Axgp[2, 3, ix, iy] + Axgpabs[2, 3, ix, iy]) * dpsidsx[3, ix, iy]) )
        DRx[3, ix, iy] = 0.5* wgp*( ((Axgp[3, 1, ix, iy] + Axgpabs[3, 1, ix, iy]) * dpsidsx[1, ix, iy]) + 
                                    ((Axgp[3, 2, ix, iy] + Axgpabs[3, 2, ix, iy]) * dpsidsx[2, ix, iy]) +
                                    ((Axgp[3, 3, ix, iy] + Axgpabs[3, 3, ix, iy]) * dpsidsx[3, ix, iy]) )
    end
    return
end

# TOP == RIGHT
function yflux!(dpsidsy, Aygp, Aygpabs, DLy, DRy, h, u, v, hgp, ugp, vgp, wgp, threads, blocks)
    @cuda blocks=blocks threads=threads dpsidsy!(dpsidsy, h, u, v); synchronize()
    @cuda blocks=blocks threads=threads Aygp!(Aygp, hgp, ugp, vgp); synchronize()
    @cuda blocks=blocks threads=threads Aygpabs!(Aygpabs, hgp, ugp, vgp); synchronize()
    @cuda blocks=blocks threads=threads Dy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp); synchronize()
end

function dpsidsy!(dpsidsy, h, u, v)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (iy≥1 && iy<size(h,2))
        dpsidsy[1, ix, iy] = h[ix, iy+1] - h[ix,   iy]
        dpsidsy[2, ix, iy] = u[ix, iy+1] * h[ix, iy+1] - u[ix, iy] * h[ix, iy]
        dpsidsy[3, ix, iy] = v[ix, iy+1] * h[ix, iy+1] - v[ix, iy] * h[ix, iy]
    end
    return
end

function Aygp!(Aygp, hgp, ugp, vgp)
    gravit = PARAMETERS.gravit
    # abbiamo messo gp per ricordarci che non ci sono tutti i punti ma 
    # ne manca uno in x
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    # c2gp = @. gravit * hgp
    if (iy≥1 && iy≤size(hgp,2))
        Aygp[1, 1, ix, iy] = 0.0
	    Aygp[1, 2, ix, iy] = 0.0
	    Aygp[1, 3, ix, iy] = 1.0
	    Aygp[2, 1, ix, iy] = -ugp[ix, iy] * vgp[ix, iy]
	    Aygp[2, 2, ix, iy] = vgp[ix, iy]
	    Aygp[2, 3, ix, iy] = ugp[ix, iy]
	    Aygp[3, 1, ix, iy] = gravit*hgp[ix, iy] - vgp[ix, iy]^2
	    Aygp[3, 2, ix, iy] = 0.0
	    Aygp[3, 3, ix, iy] = 2.0 * vgp[ix, iy]
    end
    return
end

function Aygpabs!(Aygpabs, hgp, ugp, vgp)
    gravit = PARAMETERS.gravit
    # abbiamo messo gp per ricordarci che non ci sono tutti i punti ma 
    # ne manca uno in x
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
						
    if (iy≥1 && iy≤size(hgp,2))
    	den     = (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
        _den    = 1/den
	    absden  = abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))

        Aygpabs[1, 1, ix, iy] = (-abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
                            + abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy]))) * _den
		Aygpabs[1, 2, ix, iy] = 0.0
		Aygpabs[1, 3, ix, iy] = absden * _den

		Aygpabs[2, 1, ix, iy] = -((den * abs(vgp[ix, iy]) - abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) 
                            + abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * 
                            (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))) * ugp[ix, iy]) * _den
		Aygpabs[2, 2, ix, iy] = abs(vgp[ix, iy])
		Aygpabs[2, 3, ix, iy] = ugp[ix, iy] * absden * _den

		Aygpabs[3, 1, ix, iy] = -(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * absden * _den
		Aygpabs[3, 2, ix, iy] = 0.0
		Aygpabs[3, 3, ix, iy] = ((vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * 
                            abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))) * _den
    end
    return
end

function Dy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (iy≥1 && iy≤size(dpsidsy,3))
        # DLy[:, ix, iy] = 0.5*(wgp * (Aygp[:, :, ix, iy] - Aygpabs[:, :, ix, iy]) * dpsidsy[:, ix, iy])
		# DRy[:, ix, iy] = 0.5*(wgp * (Aygp[:, :, ix, iy] + Aygpabs[:, :, ix, iy]) * dpsidsy[:, ix, iy])
        DLy[1, ix, iy] = 0.5* wgp*( ((Aygp[1, 1, ix, iy] - Aygpabs[1, 1, ix, iy]) * dpsidsy[1, ix, iy]) + 
                                    ((Aygp[1, 2, ix, iy] - Aygpabs[1, 2, ix, iy]) * dpsidsy[2, ix, iy]) +
                                    ((Aygp[1, 3, ix, iy] - Aygpabs[1, 3, ix, iy]) * dpsidsy[3, ix, iy]) )
        DLy[2, ix, iy] = 0.5* wgp*( ((Aygp[2, 1, ix, iy] - Aygpabs[2, 1, ix, iy]) * dpsidsy[1, ix, iy]) + 
                                    ((Aygp[2, 2, ix, iy] - Aygpabs[2, 2, ix, iy]) * dpsidsy[2, ix, iy]) +
                                    ((Aygp[2, 3, ix, iy] - Aygpabs[2, 3, ix, iy]) * dpsidsy[3, ix, iy]) )
        DLy[3, ix, iy] = 0.5* wgp*( ((Aygp[3, 1, ix, iy] - Aygpabs[3, 1, ix, iy]) * dpsidsy[1, ix, iy]) + 
                                    ((Aygp[3, 2, ix, iy] - Aygpabs[3, 2, ix, iy]) * dpsidsy[2, ix, iy]) +
                                    ((Aygp[3, 3, ix, iy] - Aygpabs[3, 3, ix, iy]) * dpsidsy[3, ix, iy]) )

        DRy[1, ix, iy] = 0.5* wgp*( ((Aygp[1, 1, ix, iy] + Aygpabs[1, 1, ix, iy]) * dpsidsy[1, ix, iy]) + 
                                    ((Aygp[1, 2, ix, iy] + Aygpabs[1, 2, ix, iy]) * dpsidsy[2, ix, iy]) +
                                    ((Aygp[1, 3, ix, iy] + Aygpabs[1, 3, ix, iy]) * dpsidsy[3, ix, iy]) )
        DRy[2, ix, iy] = 0.5* wgp*( ((Aygp[2, 1, ix, iy] + Aygpabs[2, 1, ix, iy]) * dpsidsy[1, ix, iy]) + 
                                    ((Aygp[2, 2, ix, iy] + Aygpabs[2, 2, ix, iy]) * dpsidsy[2, ix, iy]) +
                                    ((Aygp[2, 3, ix, iy] + Aygpabs[2, 3, ix, iy]) * dpsidsy[3, ix, iy]) )
        DRy[3, ix, iy] = 0.5* wgp*( ((Aygp[3, 1, ix, iy] + Aygpabs[3, 1, ix, iy]) * dpsidsy[1, ix, iy]) + 
                                    ((Aygp[3, 2, ix, iy] + Aygpabs[3, 2, ix, iy]) * dpsidsy[2, ix, iy]) +
                                    ((Aygp[3, 3, ix, iy] + Aygpabs[3, 3, ix, iy]) * dpsidsy[3, ix, iy]) )
    end
    return
end

function update!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy, threads, blocks)
    @cuda blocks=blocks threads=threads hqxqy!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy); synchronize()
end

function hqxqy!(h, qx, qy, DRx, DRy, DLx, DLy, dtdx, dtdy)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (ix>1 && ix < size(h,1) && iy>1 && iy< size(h,2))
         h[ix, iy] =  h[ix, iy] - dtdx * (DRx[1, ix-1, iy] + DLx[1, ix, iy]) + dtdy * (DRy[1, ix, iy-1] + DLy[1, ix, iy])
		qx[ix, iy] = qx[ix, iy] - dtdx * (DRx[2, ix-1, iy] + DLx[2, ix, iy]) + dtdy * (DRy[2, ix, iy-1] + DLy[2, ix, iy])
		qy[ix, iy] = qy[ix, iy] - dtdx * (DRx[3, ix-1, iy] + DLx[3, ix, iy]) + dtdy * (DRy[3, ix, iy-1] + DLy[3, ix, iy])
    end
    # Left wall
     h[1, iy] =   h[2, iy]
	qx[1, iy] = -qx[2, iy]
	qy[1, iy] =  qy[2, iy]
    # Right wall
     h[end, iy] =   h[end-1, iy]
	qx[end, iy] = -qx[end-1, iy]
	qy[end, iy] =  qy[end-1, iy]
    # Bottom wall
     h[ix, 1] =   h[ix, 2]
	qx[ix, 1] =  qx[ix, 2]
	qy[ix, 1] = -qy[ix, 2]
    # Top wall
     h[ix, end] =   h[ix, end-1]
	qx[ix, end] =  qx[ix, end-1]
	qy[ix, end] = -qy[ix, end-1]
    return
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
    
    savefig("results.png") 
end