function  [nn_cells] = get_NN_ClassifyingConstraints(dbs,beliefSpacePrecision, display_flag) % euclidean distance
if(nargin==2)
    display_flag = false;
end
a_num = size(dbs,1);
dbs_count = size(dbs,2);
dbs_norm = sum(dbs.*dbs,1);

precisionDigits = 6;
nn_cells = Polyhedron();

if(display_flag)
    [progressData, progressDataQueue] = ProgressData('\t\t\tComputing nearest neighbour partitioning constraints : ');
    incPercent = (1/dbs_count)*100;
else
    progressData = [];
    progressDataQueue = [];
    incPercent = [];
end

for dbs_idx = 1:dbs_count
    dbs_t = dbs;
    seed_point_norm = dbs_norm(dbs_idx);
    seed_point_t = dbs_t(:,dbs_idx);
    other_dbs_idxs = setdiff(1:dbs_count,dbs_idx);
    other_dbs_diff = round(sum(abs(dbs_t(:,other_dbs_idxs)-seed_point_t)),precisionDigits);
    seed_point_neighbours = dbs_t(:,other_dbs_idxs(other_dbs_diff==2*beliefSpacePrecision));
    seed_point_neighbours_num = size(seed_point_neighbours,2);
    seed_point_neighbours_norm = sum(seed_point_neighbours.*seed_point_neighbours,1);

    cell_constraints_lhs = zeros(seed_point_neighbours_num, a_num);
    cell_constraints_rhs = zeros(seed_point_neighbours_num, 1);
    for con_idx = 1:seed_point_neighbours_num
        cell_constraints_lhs(con_idx, :) = 2*(seed_point_neighbours(:, con_idx)-seed_point_t)';
        cell_constraints_rhs(con_idx) =  seed_point_neighbours_norm(con_idx) - seed_point_norm;
    end
    cell_constraints_lhs = normr(cell_constraints_lhs - cell_constraints_rhs*ones(1,a_num));
    cell_constraints_rhs = 0*cell_constraints_rhs;
    orig_cell = Polyhedron('A',[-eye(a_num);cell_constraints_lhs],'b',[zeros(a_num,1);cell_constraints_rhs]);
    nn_cells(dbs_idx) = orig_cell.copy();
    nn_cells(dbs_idx).Data.SeedPoint = seed_point_t;
    nn_cells(dbs_idx).Data.randomInteriorPoint = seed_point_t;
    if(display_flag)
        send(progressDataQueue, incPercent);
    end
end
if(display_flag)
    progressData.terminate();
end
end

