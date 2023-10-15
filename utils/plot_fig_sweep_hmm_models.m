function plot_fig_sweep_hmm_models(plot_inputs)
NC= plot_inputs.NC;
inst_opt_FDC = plot_inputs.inst_opt_FDC;
subopt_det_subpolicy_FDC = plot_inputs.subopt_det_subpolicy_FDC;
subopt_DBS_FDC = plot_inputs.subopt_DBS_FDC;
subopt_DBS_FDC_UA = plot_inputs.subopt_DBS_FDC_UA;
best_effort_moderation= plot_inputs.best_effort_moderation;
differential_privacy = plot_inputs.differential_privacy;
AMDPG = plot_inputs.AMDPG;


valid_model_flag = plot_inputs.valid_model_flag;
P_HgHn1_elem_range = plot_inputs.P_HgHn1_elem_range;
P_HgHn1_elem_num = length(P_HgHn1_elem_range);

if(NC)
    reward_NC = plot_inputs.reward_NC;
    precision_NC = plot_inputs.precision_NC;
end

if(inst_opt_FDC)
    reward_inst_opt_FDC = plot_inputs.reward_inst_opt_FDC;
    precision_inst_opt_FDC = plot_inputs.precision_inst_opt_FDC;
end

if(subopt_det_subpolicy_FDC)
    reward_subopt_det_SP_FDC = plot_inputs.reward_subopt_det_SP_FDC;
    precision_subopt_det_SP_FDC = plot_inputs.precision_subopt_det_SP_FDC;
end

if(subopt_DBS_FDC)
    reward_subopt_DBS_FDC = plot_inputs.reward_subopt_DBS_FDC;
    precision_subopt_DBS_FDC = plot_inputs.precision_subopt_DBS_FDC;
end

if(subopt_DBS_FDC_UA)
    reward_UA_subopt_DBS_FDC = plot_inputs.reward_UA_subopt_DBS_FDC;
    precision_UA_subopt_DBS_FDC = plot_inputs.precision_UA_subopt_DBS_FDC;
    reward_UA_subopt_DBS_FDC_UA = plot_inputs.reward_UA_subopt_DBS_FDC_UA;
end

if(best_effort_moderation)
    reward_BEM = plot_inputs.reward_BEM;
    precision_BEM = plot_inputs.precision_BEM;
end

if(differential_privacy)
    min_reward_DP = plot_inputs.min_reward_DP;
    min_precision_DP = plot_inputs.min_precision_DP;
end

if(AMDPG)
    reward_AMDPG = plot_inputs.reward_AMDPG;
    precision_AMDPG = plot_inputs.precision_AMDPG;
end

plot_ua = true;
plot_reward = true;
plot_precision = true;

font_size = 20;
line_width = 4;
marker_size = 16;

set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% set(0,'DefaultFigureWindowStyle','docked')
set(0,'DefaultFigureWindowStyle','normal')
figure_size = [1,1,1370,900];
num_columns = 4;
tiles_size_1 = 2;
tiles_size_2 = 2;
TileSpacing = 'loose';
Padding = 'loose';

if plot_ua
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','AA \& UA with original data');
        end

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
            'DisplayName','AA with modified data','LineStyle',':');

        plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC_UA(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
            'DisplayName','UA with modified data','LineStyle',':');

        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        hold off
    end
    linkaxes(H,'xy');
    ylim(H, [0 2]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');
    ylabel(t,'Average Bayesian reward','FontSize',font_size+2,'Interpreter','latex')
    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_reward
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end
        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end
        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete AA beliefs','LineStyle',':');
        end
        if(subopt_DBS_FDC_UA)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
                'DisplayName','Discrete UA beliefs','LineStyle',':');
        end
        if(best_effort_moderation)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_BEM(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','BEM','Marker','diamond','MarkerSize',marker_size);
        end
        if(differential_privacy)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),min_reward_DP(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','min DP');
        end
        if(AMDPG)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),reward_AMDPG(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG algorithm');
        end
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        hold off
    end
    linkaxes(H,'xy');
    ylim(H, [0 2]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');
    ylabel(t,'Average Bayesian reward','FontSize',font_size+2,'Interpreter','latex')
    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

if plot_precision
    figure_ = figure('Color',[1 1 1]);
    figure_.Position = figure_size;
    t = tiledlayout(tiles_size_1, tiles_size_2);
    t.TileSpacing = TileSpacing;
    t.Padding = Padding;
    H = gobjects(1,P_HgHn1_elem_num-2);
    for idx = 2:P_HgHn1_elem_num-1
        H(idx-1) = nexttile;
        hold on

        if(NC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_NC(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','Original data');
        end
        if(inst_opt_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_inst_opt_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','*','MarkerSize',marker_size,...
                'DisplayName','Instantaneously-optimal','LineStyle',':');
        end
        if(subopt_det_subpolicy_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_subopt_det_SP_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','o','MarkerSize',marker_size,...
                'DisplayName','Degenerate sub-policies','LineStyle',':');
        end
        if(subopt_DBS_FDC)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','+','MarkerSize',marker_size,...
                'DisplayName','Discrete AA beliefs','LineStyle',':');
        end
        if(subopt_DBS_FDC_UA)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_UA_subopt_DBS_FDC(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','x','MarkerSize',marker_size,...
                'DisplayName','Discrete UA beliefs','LineStyle',':');
        end
        if(best_effort_moderation)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_BEM(idx,valid_model_flag(idx,:)),'LineWidth',line_width, 'DisplayName','BEM','Marker','diamond','MarkerSize',marker_size);
        end
        if(differential_privacy)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),min_precision_DP(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','min DP');
        end
        if(AMDPG)
            plot(P_HgHn1_elem_range(valid_model_flag(idx,:)),precision_AMDPG(idx,valid_model_flag(idx,:)),'LineWidth',line_width,'Marker','square','MarkerSize',marker_size,...
                'DisplayName','AMDPG algorithm');
        end
        xlabel(['Varying $\lambda_1$ by fixing $\lambda_0$ = ',num2str(P_HgHn1_elem_range(idx))],'Interpreter','latex')
        H(idx-1).FontSize = font_size;
        hold off
    end
    linkaxes(H,'xy');
    ylim(H, [0 1]);
    xlim(H, [P_HgHn1_elem_range(2) P_HgHn1_elem_range(end-1)]);
    xticks(H,P_HgHn1_elem_range(2):0.2:P_HgHn1_elem_range(end-1));
    box(H,'on');
    grid(H,'on');
    ylabel(t,'Precision','FontSize',font_size+2,'Interpreter','latex')
    lg  = legend(H(1),'Orientation','Horizontal','NumColumns',num_columns,'FontSize',font_size-2);
    lg.Layout.Tile = 'north';
    set(figure_,'SelectionHighlight','off');
end

end