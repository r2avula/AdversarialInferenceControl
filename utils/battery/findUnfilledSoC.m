function [unfilled_soc_bin_idx] = findUnfilledSoC(samples)
soc_bins = size(samples,1);
valid = zeros(1,soc_bins);
for j = 1:soc_bins
    temp_idx = find(isnan(samples(j,:,:)), 1);
    if(~isempty(temp_idx))
       valid(j) = 1;
    end
end
valid_indices = find(valid == 1);
new_num = length(valid_indices);
if(new_num>0)
    unfilled_soc_bin_idx = valid_indices(unidrnd(new_num));
else
    unfilled_soc_bin_idx = [];
end
end

