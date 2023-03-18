function [P_out,P_out_num,P_changed] = cutPolyhedronWithHyperPlane(P_in,hp_Ae_T)
P_changed = false;
% a_num = P_in.Dim;
% simplex_a_dim = Polyhedron(eye(a_num));

intersection_part_pos_hs = Polyhedron('A',[P_in.A;hp_Ae_T],'b',[P_in.b;zeros(size(hp_Ae_T,1),1)]);
intersection_part_neg_hs = Polyhedron('A',[P_in.A;-hp_Ae_T],'b',[P_in.b;zeros(size(hp_Ae_T,1),1)]);
isValidPolyhedron_fn = @(x) isValidPolyhedron(x);
if(isValidPolyhedron_fn(intersection_part_pos_hs) && isValidPolyhedron_fn(intersection_part_neg_hs))
    % add code which compares normalized P_in.A and hp_Ae_T
    P_out = [intersection_part_pos_hs;intersection_part_neg_hs];
    P_out_num = 2;
    P_changed = true;
end

if(~P_changed)
    P_out = P_in;
    P_out_num = 1;
end

    function [isValid] = isValidPolyhedron(P_in)
        isValid = isFullDim(P_in);
        %         if(isValid)
        %             P_in_in_simplex = P_in&simplex_a_dim;
        %             isValid = size(P_in_in_simplex.V,1) >= P_in.Dim;
        %         end
    end
end

