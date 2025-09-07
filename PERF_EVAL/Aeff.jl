# This code is for evaluating the effective memory throughput of the GPU kernels
using BenchmarkTools, CUDA, Printf, Plots

include("functions_GPU.jl")
include("functions_GPU_opt.jl")

# GPU code
@inbounds @views function GPU_code(; do_cicle=false)
    if do_cicle
        # pow ∈ 0:4  if needed for the threds (2^pow)
        # pow ∈ 7:10 if needed for the array size (2^pow)
        for pow ∈ 0:5
            nx = ny = 2^10
            threads = (8, 2^pow)
            blocks = (nx ÷ threads[1], ny ÷ threads[2])

            # Initialization of the necessary arrays
            h       = CUDA.rand(Float64, nx, ny)
            u       = CUDA.rand(Float64, nx, ny)
            v       = CUDA.rand(Float64, nx, ny)
            # qx      = CUDA.zeros(Float64, nx, ny)
            # qy      = CUDA.zeros(Float64, nx, ny)
            hgpx    = CUDA.zeros(Float64, nx-1, ny)
            ugpx    = CUDA.zeros(Float64, nx-1, ny)
            vgpx    = CUDA.zeros(Float64, nx-1, ny)
            dpsidsx = CUDA.zeros(Float64, 3, nx - 1, ny)
            Axgp    = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
            Axgpabs = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
            DLx     = CUDA.zeros(Float64, 3, nx - 1, ny)
            DRx     = CUDA.zeros(Float64, 3, nx - 1, ny)

            t_it = @belapsed begin 
                @cuda blocks=$blocks threads=$threads dpsidsx!($dpsidsx, $h, $u, $v); synchronize()
                @cuda blocks=$blocks threads=$threads Axgp!($Axgp, $hgpx, $ugpx, $vgpx); synchronize()
                @cuda blocks=$blocks threads=$threads Axgpabs!($Axgpabs, $hgpx, $ugpx, $vgpx); synchronize()
                @cuda blocks=$blocks threads=$threads Dx!($DLx, $DRx, $dpsidsx, $Axgp, $Axgpabs, 1.0); synchronize()
            end
            t_tot = 99/1.e9*nx*ny*sizeof(Float64)/t_it
            @printf("GPU: nx = ny = %d and Threads = (%d, %d) T_tot = %.4f GB/s \n", nx, threads[1], threads[2], t_tot)

            CUDA.unsafe_free!(h)
            CUDA.unsafe_free!(u)
            CUDA.unsafe_free!(v)
            CUDA.unsafe_free!(hgpx)            
            CUDA.unsafe_free!(ugpx)            
            CUDA.unsafe_free!(vgpx)        
            CUDA.unsafe_free!(dpsidsx)
            CUDA.unsafe_free!(Axgp)
            CUDA.unsafe_free!(Axgpabs)
            CUDA.unsafe_free!(DLx)
            CUDA.unsafe_free!(DRx)
        end
    else
        nx = ny = 2^8 # (256)
        threads = (32, 1)
        blocks = (nx ÷ threads[1], ny ÷ threads[2])

        # Initialization of the necessary arrays
        h       = CUDA.rand(Float64, nx, ny)
        u       = CUDA.rand(Float64, nx, ny)
        v       = CUDA.rand(Float64, nx, ny)
        # qx      = CUDA.zeros(Float64, nx, ny)
        # qy      = CUDA.zeros(Float64, nx, ny)
        hgpx    = CUDA.zeros(Float64, nx-1, ny)
        ugpx    = CUDA.zeros(Float64, nx-1, ny)
        vgpx    = CUDA.zeros(Float64, nx-1, ny)
        dpsidsx = CUDA.zeros(Float64, 3, nx - 1, ny)
        Axgp    = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
        Axgpabs = CUDA.zeros(Float64, 3, 3, nx - 1, ny)
        DLx     = CUDA.zeros(Float64, 3, nx - 1, ny)
        DRx     = CUDA.zeros(Float64, 3, nx - 1, ny)

        t_it = @belapsed begin 
            @cuda blocks=$blocks threads=$threads dpsidsx!($dpsidsx, $h, $u, $v); synchronize()
            @cuda blocks=$blocks threads=$threads Axgp!($Axgp, $hgpx, $ugpx, $vgpx); synchronize()
            @cuda blocks=$blocks threads=$threads Axgpabs!($Axgpabs, $hgpx, $ugpx, $vgpx); synchronize()
            @cuda blocks=$blocks threads=$threads Dx!($DLx, $DRx, $dpsidsx, $Axgp, $Axgpabs, 1.0); synchronize()
        end
        t_tot = 99/1.e9*nx*ny*sizeof(Float64)/t_it
        @printf("GPU: nx = ny = %d and Threads = (%d, %d) T_tot = %.4f GB/s \n", nx, threads[1], threads[2], t_tot)
        
        CUDA.unsafe_free!(h)
        CUDA.unsafe_free!(u)
        CUDA.unsafe_free!(v)
        CUDA.unsafe_free!(hgpx)            
        CUDA.unsafe_free!(ugpx)            
        CUDA.unsafe_free!(vgpx)        
        CUDA.unsafe_free!(dpsidsx)
        CUDA.unsafe_free!(Axgp)
        CUDA.unsafe_free!(Axgpabs)
        CUDA.unsafe_free!(DLx)
        CUDA.unsafe_free!(DRx)
    end
end

GPU_code(do_cicle=true)


# GPU optimized code
@inbounds @views function GPU_code_opt(; do_cicle=false)
    if do_cicle
        # pow ∈ 0:4  if needed for the threds (2^pow)
        # pow ∈ 7:10 if needed for the array size (2^pow)
        for pow ∈ 0:5
            nx = ny = 2^10
            threads = (8, 2^pow)
            blocks = (nx ÷ threads[1], ny ÷ threads[2])

            # Initialization of the necessary arrays
            h       = CUDA.rand(Float64, nx, ny)
            u       = CUDA.rand(Float64, nx, ny)
            v       = CUDA.rand(Float64, nx, ny)
            DLx     = CUDA.zeros(Float64, 3, nx - 1, ny)
            DRx     = CUDA.zeros(Float64, 3, nx - 1, ny)
            sgp     = CUDA.rand(Float64, 3)
            wgp     = CUDA.rand(Float64, 3) 
            p       = parameters(nx, ny)

            t_it = @belapsed begin 
                @cuda blocks=$blocks threads=$threads compute_xfluxes!($DLx, $DRx, $h, $u, $v, $sgp, $wgp, $p, $nx, $ny); synchronize()
            end

            t_tot = 48/1.e9*nx*ny*sizeof(Float64)/t_it
            @printf("GPU_opt: nx = ny = %d and Threads = (%d, %d) T_tot = %.4f GB/s \n", nx, threads[1], threads[2], t_tot)

            CUDA.unsafe_free!(h)
            CUDA.unsafe_free!(u)
            CUDA.unsafe_free!(v)
            CUDA.unsafe_free!(DLx)
            CUDA.unsafe_free!(DRx)
            CUDA.unsafe_free!(sgp)
            CUDA.unsafe_free!(wgp)
        end
    else
        nx = ny = 2^8
        threads = (32, 1)
        blocks = (nx ÷ threads[1], ny ÷ threads[2])

        # Initialization of the necessary arrays
        h       = CUDA.rand(Float64, nx, ny)
        u       = CUDA.rand(Float64, nx, ny)
        v       = CUDA.rand(Float64, nx, ny)
        DLx     = CUDA.zeros(Float64, 3, nx - 1, ny)
        DRx     = CUDA.zeros(Float64, 3, nx - 1, ny)
        sgp     = CUDA.rand(Float64, 3)
        wgp     = CUDA.rand(Float64, 3) 
        p       = parameters(nx, ny)

        t_it = @belapsed begin 
            @cuda blocks=$blocks threads=$threads compute_xfluxes!($DLx, $DRx, $h, $u, $v, $sgp, $wgp, $p, $nx, $ny); synchronize()
        end

        t_tot = 48/1.e9*nx*ny*sizeof(Float64)/t_it
        @printf("GPU_opt: nx = ny = %d and Threads = (%d, %d) T_tot = %.4f GB/s \n", nx, threads[1], threads[2], t_tot)

        CUDA.unsafe_free!(h)
        CUDA.unsafe_free!(u)
        CUDA.unsafe_free!(v)
        CUDA.unsafe_free!(DLx)
        CUDA.unsafe_free!(DRx)
        CUDA.unsafe_free!(sgp)
        CUDA.unsafe_free!(wgp)
    end
end

GPU_code_opt(; do_cicle=true)
