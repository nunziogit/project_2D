function [s, w] = gaussian_points(n)
    if (n==1)
        s = [0.5];
        w = [1.0];
    elseif (n==2)
        s = [0.5 - sqrt(3) / 6; 0.5 + sqrt(3) / 6];
        w = [0.5; 0.5];
    elseif (n==3)
        s = [0.5 - sqrt(15) / 10; 0.5; 0.5 + sqrt(15) / 10];
        w = [5/18; 8/18; 5/18];
    end
end