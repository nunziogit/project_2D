using IJulia, BenchmarkTools, Revise, Printf 

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
    # c2gp = @. gravit * hgp

    #=
	λ1ij    = ugp[ix, iy] - sqrt(gravit*hgp[ix, iy]) # λ1[i, j]
  	λ2ij    = ugp[ix, iy]  #λ2[i, j]
	λ3ij    = ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])   #λ3[i, j]
	absλ1ij = abs(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy]))
    absλ2ij = abs(ugp[ix, iy])
	absλ3ij = abs(ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))

    den     = (ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
	_den    = 1/((ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (ugp[ix, iy] + sqrt(gravit*hgp[ix, iy])))
	absden  = abs(ugp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - abs(ugp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
    =#


    if (ix≥1 && ix≤size(h,1))
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
    if (ix≥1 && ix≤size(h,1))
        DLx[:, ix, iy] = 0.5*(wgp * (Axgp[:, :, ix, iy] - Axgpabs[:, :, ix, iy]) * dpsidsx[:, ix, iy])
		DRx[:, ix, iy] = 0.5*(wgp * (Axgp[:, :, ix, iy] + Axgpabs[:, :, ix, iy]) * dpsidsx[:, ix, iy])
    end
end

# TOP == RIGHT
function yflux!(dpsidsy, Aygp, Aygpabs, DLy, DRy, h, u, v, hgp, ugp, vgp, wgp, threads, blocks)
    @cuda blocks=blocks threads=threads dpsisdy!(dpsidsy, h, u, v); synchronize()
    @cuda blocks=blocks threads=threads Aygp!(Aygp, hgp, ugp, vgp); synchronize()
    @cuda blocks=blocks threads=threads Aygpabs!(Aygpabs, hgp, ugp, vgp); synchronize()
    @cuda blocks=blocks threads=threads Dy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp); synchronize()
end

function dpsidsy!(dpsidsy, h, u, v)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (iy≥1 && iy<size(h,2))
        dpsisdy[1, ix, iy] = h[ix, iy+1] - h[ix,   iy]
        dpsisdy[2, ix, iy] = u[ix, iy+1] * h[ix, iy+1] - u[ix, iy] * h[ix, iy]
        dpsisdy[3, ix, iy] = v[iX, iy+1] * h[ix, iy+1] - v[ix, iy] * h[ix, iy]
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
        Axgp[1, 1, ix, iy] = 0.0
	    Axgp[1, 2, ix, iy] = 0.0
	    Axgp[1, 3, ix, iy] = 1.0
	    Axgp[2, 1, ix, iy] = -ugp[ix, iy] * vgp[ix, iy]
	    Axgp[2, 2, ix, iy] = vgp[ix, iy]
	    Axgp[2, 3, ix, iy] = ugp[ix, iy]
	    Axgp[3, 1, ix, iy] = gravit*hgp[ix, iy] - vgp[ix, iy]^2
	    Axgp[3, 2, ix, iy] = 0.0
	    Axgp[3, 3, ix, iy] = 2.0 * vgp[ix, iy]
    end
    return
end

function Aygpabs!(Axgpabs, hgp, ugp, vgp)
    gravit = PARAMETERS.gravit
    # abbiamo messo gp per ricordarci che non ci sono tutti i punti ma 
    # ne manca uno in x
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    # c2gp = @. gravit * hgp


    #=
	λ1ij    = (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy]))
  	λ2ij    = vgp[ix, iy]  
	λ3ij    = (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
	absλ1ij = abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy]))
    absλ2ij = abs(vgp[ix, iy])
	absλ3ij = abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
    =#
						
    if (iy≥1 && iy≤size(h,2))
    	den     = (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
        _den    = 1/den
	    absden  = abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))

        Aygpabs[1, 1, i, j] = (-abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))
                            + abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy]))) * _den
		Aygpabs[1, 2, i, j] = 0.0
		Aygpabs[1, 3, i, j] = absden * _den

		Aygpabs[2, 1, i, j] = -((den * abs(vgp[ix, iy]) - abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) 
                            + abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * 
                            (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))) * ugpij) * _den
		Aygpabs[2, 2, i, j] = abs(vgp[ix, iy])
		Aygpabs[2, 3, i, j] = ugpij * absden * _den

		Aygpabs[3, 1, i, j] = -(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * absden * _den
		Aygpabs[3, 2, i, j] = 0.0
		Aygpabs[3, 3, i, j] = ((vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) * abs(vgp[ix, iy] - sqrt(gravit*hgp[ix, iy])) - (vgp[ix, iy] + sqrt(gravit*hgp[ix, iy])) * 
                            abs(vgp[ix, iy] + sqrt(gravit*hgp[ix, iy]))) * _den
    end
    return
end

function Dy!(DLy, DRy, dpsidsy, Aygp, Aygpabs, wgp)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (iy≥1 && iy≤size(h,2))
        DLy[:, ix, iy] = 0.5*(wgp * (Aygp[:, :, ix, iy] - Aygpabs[:, :, ix, iy]) * dpsidsy[:, ix, iy])
		DRy[:, ix, iy] = 0.5*(wgp * (Aygp[:, :, ix, iy] + Aygpabs[:, :, ix, iy]) * dpsidsy[:, ix, iy])
    end
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