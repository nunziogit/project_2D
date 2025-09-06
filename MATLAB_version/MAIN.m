%% This code implement the DOT scheme applied to the 2D SWE
% the scheme is exactly copied from the julia one, it is needed just to
% compare the performances between MATLAB and Julia
clc
clear
close all

%% Physics
lx = 40.0; ly = 40.0;
gravit  = 9.81;
R       = 2.5;
hins    = 2.5;
hout    = 1.0;
timeout = 0.2;

%% Numerics
nx = 64; ny = nx;
nt = 10000;
ngp = 2;
[sgp, wgp] = gaussian_points(ngp);

nvis = 1000;
cfl = 0.49;

dx = lx/nx; dy = ly/ny;
xc = linspace(0.5*dx, lx-0.5*dx, nx);
yc = linspace(0.5*dy, ly-0.5*dy, ny);

%% Array initialization
h = zeros(nx, ny);
u = zeros(nx, ny);
v = zeros(nx, ny);

Axgp    = zeros(3, 3, nx-1, ny);
Axgpabs = zeros(3, 3, nx-1, ny);
dpsidsx = zeros(3, nx-1, ny);
DLx     = zeros(3, nx-1, ny);
DRx     = zeros(3, nx-1, ny);

Aygp    = zeros(3, 3, nx, ny-1);
Aygpabs = zeros(3, 3, nx, ny-1);
dpsidsy = zeros(3, nx, ny-1);
DLy     = zeros(3, nx, ny-1);
DRy     = zeros(3, nx, ny-1);

%% Initial conditions
cx = 0.5*lx; cy = 0.5*ly;
[X, Y] = ndgrid(xc, yc);
D = sqrt((X - cx).^2 + (Y - cy).^2);
h = zeros(size(D)); 
h(D <= R) = hins;
h(D > R)  = hout;

qx = u .* h;
qy = v .* h;

%% Main loop in time
time = 0.0;
tic
for it = 1:nt
    %% dt evaluation
    lambdax = abs(u) + sqrt(gravit*h);
    lambday = abs(v) + sqrt(gravit*h);
    dtlx = dx ./ lambdax;
    dtly = dy ./ lambday;
    dtl  = min(min(min(dtlx)), min(min(dtly)));
    dt   = cfl * dtl;

    % time control
    if (time+dt>timeout)
        dt = timeout - time;
    end
    if (time >= timeout)
        fprintf("Timeout riched \n")
        break
    end

    %% D evaluations
    dtdx = dt/dx; dtdy = dt/dy;

    % X direction
    % --- Create views for left/right cell values ---
    hl = h(1:end-1, :);
    hr = h(2:end, :);
    ul = u(1:end-1, :);
    ur = u(2:end, :);
    vl = v(1:end-1, :);
    vr = v(2:end, :);
    
    for igp = 1:ngp
        % Gauss points interpolations
        hgp  = hl + sgp(igp) * (hr - hl);
        ugp  = ul + sgp(igp) * (ur - ul);
        vgp  = vl + sgp(igp) * (vr - vl);
        c2gp = gravit * hgp;
    
        lambda1 = ugp - sqrt(c2gp);
        lambda2 = ugp;
        lambda3 = ugp + sqrt(c2gp);
    
        % --- First loop: compute dpsidsx, Axgp, Axgpabs ---
        for j = 1:ny
            for i = 1:(nx-1)
                hrij = hr(i, j);
                hlij = hl(i, j);
                urij = ur(i, j);
                ulij = ul(i, j);
                vrij = vr(i, j);
                vlij = vl(i, j);
    
                dpsidsx(1, i, j) = hrij - hlij;
                dpsidsx(2, i, j) = urij * hrij - ulij * hlij;
                dpsidsx(3, i, j) = vrij * hrij - vlij * hlij;
    
                ugpij  = ugp(i, j);
                vgpij  = vgp(i, j);
                c2gpij = c2gp(i, j);
    
                % --- Axgp matrix ---
                Axgp(1, 1, i, j) = 0.0;
                Axgp(1, 2, i, j) = 1.0;
                Axgp(1, 3, i, j) = 0.0;
                Axgp(2, 1, i, j) = c2gpij - ugpij^2;
                Axgp(2, 2, i, j) = 2.0 * ugpij;
                Axgp(2, 3, i, j) = 0.0;
                Axgp(3, 1, i, j) = -ugpij * vgpij;
                Axgp(3, 2, i, j) = vgpij;
                Axgp(3, 3, i, j) = ugpij;
    
                % --- Axgpabs matrix ---
                lambda1ij = lambda1(i, j);
                lambda2ij = lambda2(i, j);
                lambda3ij = lambda3(i, j);
    
                absl1ij = abs(lambda1ij);
                absl3ij = abs(lambda3ij);
                den     = lambda1ij - lambda3ij;
                absden  = absl1ij - absl3ij;
    
                Axgpabs(1, 1, i, j) = (-absl1ij * lambda3ij + absl3ij * lambda1ij) / den;
                Axgpabs(1, 2, i, j) = absden / den;
                Axgpabs(1, 3, i, j) = 0.0;
    
                Axgpabs(2, 1, i, j) = -lambda1ij * lambda3ij * absden / den;
                Axgpabs(2, 2, i, j) = (lambda1ij * absl1ij - lambda3ij * absl3ij) / den;
                Axgpabs(2, 3, i, j) = 0.0;
    
                Axgpabs(3, 1, i, j) = -((den * abs(lambda2ij) - absl3ij * lambda1ij + absl1ij * lambda3ij) * vgpij) / den;
                Axgpabs(3, 2, i, j) = vgpij * absden / den;
                Axgpabs(3, 3, i, j) = abs(lambda2ij);
            end
        end
    
        % --- Second loop: accumulate DLx and DRx ---
        for j = 1:ny
            for i = 1:(nx-1)
                dps   = dpsidsx(:, i, j);      % 3x1 vector
                Ax    = Axgp(:, :, i, j);      % 3x3 matrix
                Axabs = Axgpabs(:, :, i, j);   % 3x3 matrix
    
                DLx(:, i, j) = DLx(:, i, j) + wgp(igp) * (Ax - Axabs) * dps;
                DRx(:, i, j) = DRx(:, i, j) + wgp(igp) * (Ax + Axabs) * dps;
            end
        end
    end

    % Final scaling
    DLx = 0.5 * DLx;
    DRx = 0.5 * DRx;
    % END of X direction

    % Y direction
    % --- Y-direction fluxes (horizontal faces) ---
    % Bottom/top states on cell faces
    hb = h(:, 1:end-1);
    ht = h(:, 2:end);
    ub = u(:, 1:end-1);
    ut = u(:, 2:end);
    vb = v(:, 1:end-1);
    vt = v(:, 2:end);
    
    for igp = 1:ngp
        % Gauss point interpolations
        hgp  = hb + sgp(igp) * (ht - hb);
        ugp  = ub + sgp(igp) * (ut - ub);
        vgp  = vb + sgp(igp) * (vt - vb);
        c2gp = gravit * hgp;
    
        lambda1 = vgp - sqrt(c2gp);
        lambda2 = vgp;
        lambda3 = vgp + sqrt(c2gp);
    
        % --- First loop: compute dpsidsy, Aygp, Aygpabs ---
        for j = 1:(ny-1)
            for i = 1:nx
                htij = ht(i, j);
                hbij = hb(i, j);
                utij = ut(i, j);
                ubij = ub(i, j);
                vtij = vt(i, j);
                vbij = vb(i, j);
    
                dpsidsy(1, i, j) = htij - hbij;
                dpsidsy(2, i, j) = utij * htij - ubij * hbij;
                dpsidsy(3, i, j) = vtij * htij - vbij * hbij;
    
                ugpij  = ugp(i, j);
                vgpij  = vgp(i, j);
                c2gpij = c2gp(i, j);
    
                % --- Aygp matrix ---
                Aygp(1, 1, i, j) = 0.0;
                Aygp(1, 2, i, j) = 0.0;
                Aygp(1, 3, i, j) = 1.0;
                Aygp(2, 1, i, j) = -ugpij * vgpij;
                Aygp(2, 2, i, j) = vgpij;
                Aygp(2, 3, i, j) = ugpij;
                Aygp(3, 1, i, j) = c2gpij - vgpij^2;
                Aygp(3, 2, i, j) = 0.0;
                Aygp(3, 3, i, j) = 2.0 * vgpij;
    
                % --- Aygpabs matrix ---
                lambda1ij = lambda1(i, j);
                lambda3ij = lambda3(i, j);
                lambda2ij = lambda2(i, j);
    
                absl1ij = abs(lambda1ij);
                absl3ij = abs(lambda3ij);
                den     = lambda1ij - lambda3ij;
                absden  = absl1ij - absl3ij;
    
                Aygpabs(1, 1, i, j) = (-absl1ij * lambda3ij + absl3ij * lambda1ij) / den;
                Aygpabs(1, 2, i, j) = 0.0;
                Aygpabs(1, 3, i, j) = absden / den;
    
                Aygpabs(2, 1, i, j) = -((den * abs(lambda2ij) - absl3ij * lambda1ij + absl1ij * lambda3ij) * ugpij) / den;
                Aygpabs(2, 2, i, j) = abs(lambda2ij);
                Aygpabs(2, 3, i, j) = ugpij * absden / den;
    
                Aygpabs(3, 1, i, j) = -lambda1ij * lambda3ij * absden / den;
                Aygpabs(3, 2, i, j) = 0.0;
                Aygpabs(3, 3, i, j) = (lambda1ij * absl1ij - lambda3ij * absl3ij) / den;
            end
        end
    
        % --- Second loop: accumulate DLy and DRy ---
        for j = 1:(ny-1)
            for i = 1:nx
                dps   = dpsidsy(:, i, j);    % 3x1 vector
                Ay    = Aygp(:, :, i, j);    % 3x3 matrix
                Ayabs = Aygpabs(:, :, i, j); % 3x3 matrix
    
                DLy(:, i, j) = DLy(:, i, j) + wgp(igp) * (Ay - Ayabs) * dps;
                DRy(:, i, j) = DRy(:, i, j) + wgp(igp) * (Ay + Ayabs) * dps;
            end
        end
    end
    
    % Final scaling
    DLy = 0.5 * DLy;
    DRy = 0.5 * DRy;
    % END of Y direction

    %% Update the solution
    for i = 2:(nx-1)
        for j = 2:(ny-1)
            % --- Compute divergence in x-direction ---
            h(i, j)  = h(i, j)  - dtdx * (DRx(1, i-1, j) + DLx(1, i, j)) ...
                                 - dtdy * (DRy(1, i, j-1) + DLy(1, i, j));
    
            qx(i, j) = qx(i, j) - dtdx * (DRx(2, i-1, j) + DLx(2, i, j)) ...
                                 - dtdy * (DRy(2, i, j-1) + DLy(2, i, j));
    
            qy(i, j) = qy(i, j) - dtdx * (DRx(3, i-1, j) + DLx(3, i, j)) ...
                                 - dtdy * (DRy(3, i, j-1) + DLy(3, i, j));
        end
    end

    % BC
    h(1,:)  = h(2,:);
    qx(1,:) = -qx(2,:);
    qy(1,:) = qy(2,:);

    h(end,:) = h(end-1,:);
    qx(end,:) = -qx(end-1,:);
    qy(end,:) = qy(end-1,:);

    h(:,1) = h(:,2);
    qx(:,1) = qx(:,2);
    qy(:,1) = -qy(:,2);

    h(:,end) = h(:,end-1);
    qx(:,end) = qx(:,end-1);
    qy(:,end) = -qy(:,end-1);

    % set D to zero
    DLx(:) = 0.0;
    DRx(:) = 0.0;
    DLy(:) = 0.0;
    DRy(:) = 0.0;


    for i = 1:nx
        for j = 1:ny
            if h(i, j) > 1e-12
                u(i, j) = qx(i, j) / h(i, j);
                v(i, j) = qy(i, j) / h(i, j);
            else
                u(i, j) = 0.0;
                v(i, j) = 0.0;
            end
        end
    end

    time = time + dt;

end % main loop in time
elapsed_time = toc;
fprintf("Elapsed time: %.4f seconds \n", elapsed_time)

% --- Compute velocity magnitude ---
vel = sqrt(u.^2 + v.^2);

% --- Heatmaps for h, u, v, vel ---
figure;

subplot(4,2,1);
imagesc(xc, yc, h'); axis equal; colorbar;
xlabel('x'); ylabel('y'); title('Water Depth (h)');

subplot(4,2,3);
imagesc(xc, yc, u'); axis equal; colorbar;
xlabel('x'); ylabel('y'); title('Velocity u');

subplot(4,2,5);
imagesc(xc, yc, v'); axis equal; colorbar;
xlabel('x'); ylabel('y'); title('Velocity v');

subplot(4,2,7);
imagesc(xc, yc, vel'); axis equal; colorbar;
xlabel('x'); ylabel('y'); title('Velocity Magnitude');

% --- Extract diagonal slices (45° slice through the domain) ---
[nx, ny] = size(h);
n_diag = min(nx, ny);

slice_h   = zeros(1, n_diag);
slice_u   = zeros(1, n_diag);
slice_v   = zeros(1, n_diag);
slice_vel = zeros(1, n_diag);
s         = zeros(1, n_diag);

for i = 1:n_diag
    slice_h(i)   = h(i, i);
    slice_u(i)   = u(i, i);
    slice_v(i)   = v(i, i);
    slice_vel(i) = vel(i, i);
    s(i)         = sqrt((xc(i) - xc(1))^2 + (yc(i) - yc(1))^2);
end

% --- Slice plots ---
subplot(4,2,2);
plot(s, slice_h, 'o-', 'LineWidth', 2);
xlabel('Distance along 45° slice'); ylabel('h'); title('Diagonal Slice of h');

subplot(4,2,4);
plot(s, slice_u, 'o-', 'LineWidth', 2);
xlabel('Distance along 45° slice'); ylabel('u'); title('Diagonal Slice of u');

subplot(4,2,6);
plot(s, slice_v, 'o-', 'LineWidth', 2);
xlabel('Distance along 45° slice'); ylabel('v'); title('Diagonal Slice of v');

subplot(4,2,8);
plot(s, slice_vel, 'o-', 'LineWidth', 2);
xlabel('Distance along 45° slice'); ylabel('vel'); title('Diagonal Slice of vel');

sgtitle(['Results at t = ', num2str(timeout)]);




