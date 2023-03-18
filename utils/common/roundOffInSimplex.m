function [belief_out,belief_out_idxs] = roundOffInSimplex(belief_in,beliefSpacePrecision,dbs_data,anchor_idxs)
if(size(belief_in,1) == 1 && size(belief_in,2)>1)
    belief_in = belief_in';
end
points_num = size(belief_in,2);
a_num = size(belief_in,1);
if(nargin<3)
    dbs_data = [];
end
if(nargin<2)
    error('beliefSpacePrecision = ?');
end
belief_in = belief_in./sum(belief_in);
belief_out = zeros(a_num,points_num);
belief_out_idxs = zeros(points_num,1);
if(~isempty(dbs_data))
    if(isstruct(dbs_data))
        dbs = dbs_data.dbs;
        nn_cells = dbs_data.nn_cells;
        for point_idx = 1:points_num
            tf = contains(nn_cells,belief_in(:,point_idx), true);
            if(any(tf))
                belief_idx = find(tf,1);
                belief_out(:,point_idx) = dbs(:,belief_idx);
                belief_out_idxs(point_idx) = belief_idx;
            else
                error('~constraints_satisfied')
            end
        end
    elseif(isa(dbs_data,'Polyhedron'))
        for point_idx = 1:points_num
            tf = contains(dbs_data,belief_in(:,point_idx), true);
            if(any(tf))
                belief_idx = find(tf,1);
                belief_out(:,point_idx) = dbs_data(belief_idx).Data.randomInteriorPoint;
                belief_out_idxs(point_idx) = belief_idx;
            else
                dbs_data_data = [dbs_data.Data];
                belief_space_T = [dbs_data_data.randomInteriorPoint]';
                rounded_belief_idx = knnsearch(belief_space_T,belief_in(:,point_idx)');
                belief_out(:,point_idx) = belief_space_T(rounded_belief_idx,:)';
                belief_out_idxs(point_idx) = rounded_belief_idx;
            end
        end
    else
        belief_space_T = dbs_data;
        for point_idx = 1:points_num
            rounded_belief_idx = knnsearch(belief_space_T,belief_in(:,point_idx)');
            belief_out(:,point_idx) = belief_space_T(rounded_belief_idx,:)';
            belief_out_idxs(point_idx) = rounded_belief_idx;
        end
    end
else
    if(nargin<4)
        [~,anchor_idxs] = max(belief_in,[],1);
    end
    for point_idx = 1:points_num
        anchor_idx = anchor_idxs(point_idx);
        this_point = belief_in(:,point_idx);
        this_point = roundOff(this_point,beliefSpacePrecision);
        sum_this_point = sum(this_point);
        while sum_this_point>1
            this_point = max(0,this_point - beliefSpacePrecision);
            sum_this_point = sum(this_point);
            if sum_this_point==0
                this_point = 0*this_point;
                this_point(anchor_idx) = 1;
                sum_this_point = 1;
            end
        end
        if sum_this_point<1
            this_point(anchor_idx) = 1 - sum(this_point(setdiff(1:a_num,anchor_idx)));
        end
        belief_out(:,point_idx) = this_point;
    end
end
end