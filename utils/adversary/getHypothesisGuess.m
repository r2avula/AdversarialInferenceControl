function [h_hat_opt_idx] = getHypothesisGuess(belief_in,DRs_in_Rn)
tf = contains(DRs_in_Rn,belief_in, true);
if(any(tf))
    h_hat_opt_idx = find(tf,1);
else
    h_hat_opt_idx = 1;
end
end

