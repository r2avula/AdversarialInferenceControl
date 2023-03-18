function [gurobi_model,gurobi_model_params,Aeq_cons,beq_cons] = get_gurobi_model_FDC(params,function_handles,valid_yc_idxs_flag)
h_num = params.h_num;
x_num = params.x_num;
z_num = params.z_num;
s_num = params.s_num;
u_num = params.u_num;
valid_XgYZn1 = params.valid_XgYZn1;

y_control_num = params.y_control_num;
y_control_range = 1:y_control_num;

s_range = 1:s_num;
h_range = 1:h_num;
x_range = 1:x_num;

if(nargin==2)
    valid_yc_idxs_flag = true(y_control_num,1);
elseif(nargin==1)
    error("require function_handles")
end

HsZ2A = function_handles.HsZ2A;
XHsAn1s_2S = function_handles.XHsAn1s_2S;
YcSs_2U = function_handles.YcSs_2U;
YcsS_2U = function_handles.YcsS_2U;

gurobi_var_dim = u_num;
gurobi_model.modelsense = 'min';
gurobi_model.vtype = repmat('C', gurobi_var_dim, 1);
gurobi_model.lb    = zeros(gurobi_var_dim, 1);
gurobi_model.ub   = ones(gurobi_var_dim, 1);
gurobi_model_params.outputflag = 0;

Aeq_cons = zeros(s_num,gurobi_var_dim);
for s_k_idx = s_range
    Aeq_cons(s_k_idx,YcsS_2U(y_control_range,s_k_idx)) = 1;
end
beq_cons = ones(s_num,1);

Aeq_cons_2 = zeros(1,gurobi_var_dim);
for z_kn1_idx = 1:z_num
    a_kn1_idxs = HsZ2A(h_range,z_kn1_idx);
    for yc_idx = 1:y_control_num
        invalid_x_idxs = setdiff(x_range, valid_XgYZn1{yc_idx, z_kn1_idx});
        for x_k_idx = invalid_x_idxs
            s_k_idxs = XHsAn1s_2S(x_k_idx,h_range',a_kn1_idxs');
            Aeq_cons_2(YcSs_2U(yc_idx,s_k_idxs)) = 1;
        end
    end
end

if(any(~valid_yc_idxs_flag))
    invalid_yc_idxs = find(~valid_yc_idxs_flag);
    for s_k_idx = s_range
        Aeq_cons_2(s_k_idx,YcsS_2U(invalid_yc_idxs,s_k_idx)) = 1; %#ok<FNDSB> 
    end
end

if(any(Aeq_cons_2>0))
    Aeq_cons = [Aeq_cons;Aeq_cons_2];
    beq_cons = [beq_cons;0];
end

gurobi_model.A = sparse(Aeq_cons);
gurobi_model.rhs  = beq_cons;
gurobi_model.sense = repmat('=',[1,length(beq_cons)]);
end

