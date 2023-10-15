function [DRs_in_Rn] = getDecisionRegionPolyhedrons(params,display_flag)
if(nargin==1)
    display_flag = true;
end
h_num = params.h_num;

%% Decision regions
if(display_flag)
    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing decision regions : ');
    incPercent = (1/h_num)*100;
else
    progressData = [];
    progressDataQueue = [];
    incPercent = [];
end

DRs_in_Rn = Polyhedron();
internal_routine_fn = @internal_routine;
% [~,p_pool] = evalc('gcp(''nocreate'');');
% if isempty(p_pool)
    for region_idx = 1:h_num
        DRs_in_Rn(region_idx) = internal_routine_fn(region_idx, params);
        if(display_flag)
            send(progressDataQueue, incPercent);
        end
    end
% else
%     params = parallel.pool.Constant(params);
%     parfor region_idx = 1:h_num
%         params_ = params.Value;
%         DRs_in_Rn(region_idx) = feval(internal_routine_fn, region_idx, params_);  %#ok<*FVAL> 
%         if(display_flag)
%             send(progressDataQueue, incPercent);
%         end
%     end
% end

if(display_flag)
    progressData.terminate();
end

drs_num = length(DRs_in_Rn);
if(drs_num~= h_num)
    error('Something is wrong!');
end

    function drs_full_dim_t = internal_routine(region_idx, params)
        h_num_ = params.h_num;
        C_HgHh_design = params.C_HgHh_design;
        dr_facets_Ae_T = zeros(h_num_-1,h_num_);
        con_idx = 1;
        for j_idx = 1:h_num_
            if(j_idx ~=region_idx)
                dr_facets_Ae_T(con_idx,:) = (C_HgHh_design(j_idx,:)-C_HgHh_design(region_idx,:));
                con_idx = con_idx + 1;
            end
        end
        drs_full_dim_t = Polyhedron('A',[-eye(h_num_);dr_facets_Ae_T],'b',zeros(2*h_num_-1,1));
        drs_full_dim_t.minHRep();
    end
end

