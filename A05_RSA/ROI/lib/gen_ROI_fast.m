
function Sphere_Index = gen_ROI_fast(Vref,mmCoords,Center_vox, Radium)

%
%        Center:
%               Center coordinates of the ROIs (# of ROIs*3 array).
%        Radium:
%               Radium for Sphere;
%        Ref_img:
%               Directory & filename of the reference image.
%

%Vref = spm_vol(Ref_img);
%[Data, mmCoords] = spm_read_vols(Vref);
%[~, VoxelQuantity] = size(mmCoords);
%Data_vector = reshape(Data, 1, VoxelQuantity);

[rowsQuantity, ~] = size(Center_vox);
if rowsQuantity ~= 3
    Center_vox = Center_vox';
end
Center_mm = Vref.mat * [Center_vox; 1];

xs = mmCoords(1,:) - Center_mm(1);
ys = mmCoords(2,:) - Center_mm(2);
zs = mmCoords(3,:) - Center_mm(3);

radii = sqrt(xs.^2+ys.^2+zs.^2);  %与中心点的距离
Sphere_Index = find(radii <= Radium);  %返回序号吗？一个行向量
