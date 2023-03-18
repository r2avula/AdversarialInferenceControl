function [points_out] = roundOff(points_in,precision)
if (precision==0)
    error("here in roundOff")
end
residue = points_in - precision.*floor(points_in./precision);
% residue = points_in - precision.*ceil(points_in./precision);
points_out = points_in - residue;
fix_flag = residue>=(precision/2);
points_out(fix_flag) = points_out(fix_flag) + precision;
end