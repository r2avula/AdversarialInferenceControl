function [sm_data, gt_data,P_H0,h0_idxs] = generateSyntheticData(params)
h_num = params.h_num;
numHorizons = params.numHorizons;
k_num = params.k_num;

sm_data = zeros(k_num,numHorizons);
gt_data = zeros(k_num,numHorizons);
h0_idxs = zeros(1,numHorizons);

[progressData, progressDataQueue] = ProgressData('\t\t\tGenerating data : ');
incPercent = (1/numHorizons)*100;
internal_routine_fn = @internal_routine;
[~,p_pool] = evalc('gcp(''nocreate'');');
if isempty(p_pool)
    for dayIdx=1:numHorizons
        [sm_data(:, dayIdx), gt_data(:, dayIdx), h0_idxs(dayIdx)] = feval(internal_routine_fn, params); %#ok<*FVAL> 
        send(progressDataQueue, incPercent);
    end
else
    params = parallel.pool.Constant(params);
    parfor dayIdx=1:numHorizons
        params_ = params.Value;
        [sm_data(:, dayIdx), gt_data(:, dayIdx), h0_idxs(dayIdx)] = feval(internal_routine_fn, params_);
        send(progressDataQueue, incPercent);
    end
end
progressData.terminate();

P_H0 = zeros(h_num,1);
for h_idx = 1:h_num
    P_H0(h_idx) = sum(h0_idxs(:)==h_idx);
end
P_H0 = P_H0/sum(P_H0);


    function [sm_data_, h_k_idx_, h_0_idx] = internal_routine(params)
        P_HgHn1 = params.P_HgHn1;
        P_XgH = params.P_XgH;
        k_num_ = params.k_num;
        x_p_pu = params.x_p_pu;
        x_offset = params.x_offset;

        h_k_idx_ = zeros(k_num_,1);
        sm_data_ = zeros(k_num_,1);

        %     rng('shuffle')
        this_step_distribution = params.P_H0;
        cumulative_distribution = cumsum(this_step_distribution);
        previous_step = find(cumulative_distribution>=rand(),1);
        h_0_idx = previous_step;
        for k_idx=1:k_num_
            this_step_distribution = P_HgHn1(:,previous_step)';
            cumulative_distribution = cumsum(this_step_distribution);
            previous_step = find(cumulative_distribution>=rand(),1);
            h_k_idx_(k_idx) = previous_step;

            x_distribution = P_XgH(:,previous_step)';
            cumulative_distribution = cumsum(x_distribution);
            sm_data_(k_idx) = x_p_pu*(find(cumulative_distribution>=rand(),1)+x_offset);
        end
    end


end