function [function_handles] = get_vectorizing_function_handles(params)
h_num = params.h_num;
z_num = params.z_num;
x_num = params.x_num;
a_num = params.a_num;
s_num = x_num*h_num*a_num;
y_control_num = params.y_control_num;

HZ2A= @(h,z)sub2ind([h_num,z_num],h,z);
A2HZ= @(a)ind2sub([h_num,z_num],a);
XHAn1_2S= @(x,h,an1)sub2ind([x_num,h_num,a_num],x,h,an1);
S2XHAn1= @(s)ind2sub([x_num,h_num,a_num],s);
YcS_2U= @(yc,s)sub2ind([y_control_num,s_num],yc,s);
U2YcS= @(u)sub2ind([y_control_num,s_num],u);

HZs2A= @(h,zs)HZ2A(h*ones(size(zs)),zs);
HsZ2A= @(hs,z)HZ2A(hs,z*ones(size(hs)));
XsHAn1_2S= @(xs,h,an1)XHAn1_2S(xs,h*ones(size(xs)),an1*ones(size(xs)));
XHsAn1s_2S= @(x,hs,an1s)reshape(XHAn1_2S(x*ones(length(hs),length(an1s)),hs(:)*ones(size(an1s(:)))', ones(size(hs(:)))*an1s(:)'),...
    iscolumn(hs)*[length(hs)*length(an1s),1]+isrow(hs)*[1,length(hs)*length(an1s)]);
XsHAn1s_2S= @(xs,h,an1s)reshape(XHAn1_2S(xs(:)*ones(size(an1s(:)))', h*ones(length(xs),length(an1s)), ones(size(xs(:)))*an1s(:)'),...
    iscolumn(xs)*[length(xs)*length(an1s),1]+isrow(xs)*[1,length(xs)*length(an1s)]);
YcSs_2U= @(yc,ss)YcS_2U(yc*ones(size(ss)),ss);
YcsS_2U= @(ycs,s)YcS_2U(ycs,s*ones(size(ycs)));
XHAn1s_2S= @(x,h,an1s)XHAn1_2S(x*ones(size(an1s)),h*ones(size(an1s)),an1s);

YcH_2V= @(yc,h)sub2ind([y_control_num,h_num],yc,h);
YcsH_2V= @(ycs,h)YcH_2V(ycs,h*ones(size(ycs)));
YcHs_2V= @(yc,hs)YcH_2V(yc*ones(size(hs)),hs);
V2YcH= @(v)ind2sub([y_control_num,h_num],v);

function_handles = struct;
function_handles.HZ2A = HZ2A;
function_handles.A2HZ = A2HZ;
function_handles.XHAn1_2S = XHAn1_2S;
function_handles.S2XHAn1 = S2XHAn1;
function_handles.YcS_2U = YcS_2U;
function_handles.HZs2A = HZs2A;
function_handles.HsZ2A = HsZ2A;
function_handles.XsHAn1_2S = XsHAn1_2S;
function_handles.XHAn1s_2S = XHAn1s_2S;
function_handles.XHsAn1s_2S = XHsAn1s_2S;
function_handles.XsHAn1s_2S = XsHAn1s_2S;
function_handles.YcSs_2U = YcSs_2U;
function_handles.YcsS_2U = YcsS_2U;
function_handles.U2YcS = U2YcS;
function_handles.YcH_2V = YcH_2V;
function_handles.YcsH_2V = YcsH_2V;
function_handles.YcHs_2V = YcHs_2V;
function_handles.V2YcH = V2YcH;

if(isfield(params,'l_num'))
    l_num = params.l_num;
    t_num = params.t_num;
    b_num = params.b_num;

    HL2B = @(h,l)sub2ind([h_num,l_num],h,l);
    B2HL = @(b)ind2sub([h_num,l_num],b);
    HLs2B = @(h,ls)HL2B(h*ones(size(ls)),ls);
    HsL2B = @(hs,l)HL2B(hs,l*ones(size(hs)));
    
    XHBn1_2T = @(x,h,b)sub2ind([x_num,h_num, b_num],x,h,b);
    T2XHBn1 = @(t)ind2sub([x_num,h_num, b_num],t);
    % XBs_2T = @(x,ls)XHBn1_2T(x*ones(size(ls)),ls);
    XsHBn1_2T = @(xs,h, b)XHBn1_2T(xs,h*ones(size(xs)), b*ones(size(xs)));
    XHBn1s_2T = @(x,h, bn1s)XHBn1_2T(x*ones(size(bn1s)),h*ones(size(bn1s)), bn1s);

    YT_2W = @(y,t)sub2ind([y_control_num,t_num],y,t);
    YTs_2W = @(y,ts)YT_2W(y*ones(size(ts)),ts);
    YsT_2W = @(ys,t)YT_2W(ys,t*ones(size(ys)));


    function_handles.HL2B = HL2B;
    function_handles.B2HL = B2HL;
    function_handles.HsL2B = HsL2B;
    function_handles.HLs2B = HLs2B;

    function_handles.XHBn1_2T = XHBn1_2T;
    function_handles.XsHBn1_2T = XsHBn1_2T;
    function_handles.XHBn1s_2T = XHBn1s_2T;
    function_handles.T2XHBn1 = T2XHBn1;

    function_handles.YT_2W = YT_2W;
    function_handles.YTs_2W = YTs_2W;
    function_handles.YsT_2W = YsT_2W;
end
end

