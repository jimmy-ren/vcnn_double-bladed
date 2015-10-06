function [vin, vout] = EdgeExtract(img, imgf)
    S = im2double(img);
    vin = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
    img_filtered = im2double(imgf);
    S = im2double(img_filtered);
    vout = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
end


