function [gamma_out,gamma_out_flag] = prune_POMDP(gamma_in,polyhedralCone)
a_num = size(gamma_in,1);
num_vecs = size(gamma_in,2);
Ae_cons = [ones(1,a_num),0];

gurobi_var_dim = a_num+1;
gurobi_model.modelsense = 'min';
gurobi_model.vtype = repmat('C', gurobi_var_dim, 1);
gurobi_model.lb    = [zeros(a_num, 1);-inf];
gurobi_model.ub   = [ones(a_num, 1);+inf];
gurobi_model.obj  = [zeros(1,a_num),1];
gurobi_model_params.outputflag = 0;
gurobi_model_params.DualReductions = 0;

poly_AConstraints = polyhedralCone.A;
polyhedron = polyhedralCone&Polyhedron(eye(a_num));
sol = interiorPoint(polyhedron);
test_point = sol.x;
[~,best_vec_idx] = min(test_point'*gamma_in);

gamma_out_flag = false(num_vecs,1);
gamma_out_flag(best_vec_idx) = true;

poly_ACons = [poly_AConstraints, zeros(size(poly_AConstraints,1),1)];
for test_vec_idx = 1:num_vecs
    if(~gamma_out_flag(test_vec_idx))
        test_vec = gamma_in(:,test_vec_idx);
        test_complete = false;
        while ~test_complete
            temp_mat = -transpose(test_vec - gamma_in(:,gamma_out_flag));
            A_cons = [poly_ACons;[temp_mat,ones(size(temp_mat,1),1)]];
            b_cons = zeros(size(poly_ACons,1)+size(temp_mat,1),1);

            gurobi_model.A     = sparse([Ae_cons;A_cons]);
            gurobi_model.rhs   = [1;b_cons];
            gurobi_model.sense = ['=',repelem('<',1,size(A_cons,1))];
            optimizerResult = gurobi(gurobi_model, gurobi_model_params);
            if strcmp(optimizerResult.status, 'OPTIMAL') && (optimizerResult.objval>=0)
                test_point = optimizerResult.x(1:a_num);
                test_value = test_point'*test_vec;
                test_verify_flag = ~gamma_out_flag;
                test_verify_flag(1:test_vec_idx) = false;
                values = test_point'*gamma_in(:,test_verify_flag);
                if(isempty(values)||test_value>=max(values))
                    test_complete = true;
                else
                    [~,best_vec_idx] = min(values);
                    gamma_out_flag(best_vec_idx) = true;
                    if(best_vec_idx==test_vec_idx)
                        test_complete = true;
                    end
                end
            else
                gamma_out_flag(test_vec_idx) = true;
                test_complete = true;
            end
        end
    end
end
gamma_out = gamma_in(:,gamma_out_flag);
end