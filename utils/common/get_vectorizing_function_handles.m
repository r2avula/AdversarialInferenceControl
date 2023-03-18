function [function_handles] = get_vectorizing_function_handles(params)
h_num = params.h_num;
z_num = params.z_num;
x_num = params.x_num;
a_num = params.a_num;
s_num = x_num*h_num*a_num;
y_control_num = params.y_control_num;

AYc2V= @(a,yc)sub2ind([a_num, y_control_num],a,yc);
HZ2A= @(h,z)sub2ind([h_num,z_num],h,z);
A2HZ= @(a)ind2sub([h_num,z_num],a);
XHAn1_2S= @(x,h,an1)sub2ind([x_num,h_num,a_num],x,h,an1);
S2XHAn1= @(s)ind2sub([x_num,h_num,a_num],s);
YcS_2U= @(yc,s)sub2ind([y_control_num,s_num],yc,s);
U2YcS= @(u)sub2ind([y_control_num,s_num],u);

HZs2A= @(h,zs)HZ2A(h*ones(size(zs)),zs);
AsYc2V= @(as,yc)AYc2V(as,yc*ones(size(as)));
HsZ2A= @(hs,z)HZ2A(hs,z*ones(size(hs)));
XsHAn1_2S= @(xs,h,an1)XHAn1_2S(xs,h*ones(size(xs)),an1*ones(size(xs)));
XHsAn1s_2S= @(x,hs,an1s)reshape(XHAn1_2S(x*ones(length(hs),length(an1s)),hs(:)*ones(size(an1s(:)))', ones(size(hs(:)))*an1s(:)'),...
    iscolumn(hs)*[length(hs)*length(an1s),1]+isrow(hs)*[1,length(hs)*length(an1s)]);
XsHAn1s_2S= @(xs,h,an1s)reshape(XHAn1_2S(xs(:)*ones(size(an1s(:)))', h*ones(length(xs),length(an1s)), ones(size(xs(:)))*an1s(:)'),...
    iscolumn(xs)*[length(xs)*length(an1s),1]+isrow(xs)*[1,length(xs)*length(an1s)]);
YcSs_2U= @(yc,ss)YcS_2U(yc*ones(size(ss)),ss);
YcsS_2U= @(ycs,s)YcS_2U(ycs,s*ones(size(ycs)));
XHAn1s_2S= @(x,h,an1s)XHAn1_2S(x*ones(size(an1s)),h*ones(size(an1s)),an1s);

c_num = x_num*h_num*h_num;

XHHn1_2C= @(x,h,hn1)sub2ind([x_num,h_num,h_num],x,h,hn1);
XsHHn1_2C= @(xs,h,hn1)XHHn1_2C(xs,h*ones(size(xs)),hn1*ones(size(xs)));
YcC_2G= @(yc,c)sub2ind([y_control_num,c_num],yc,c);
YcCs_2G= @(yc,cs)YcC_2G(yc*ones(size(cs)),cs);

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
function_handles.AYc2V = AYc2V;
function_handles.AsYc2V = AsYc2V;

function_handles.XHHn1_2C = XHHn1_2C;
function_handles.XsHHn1_2C = XsHHn1_2C;
function_handles.YcC_2G = YcC_2G;
function_handles.YcCs_2G = YcCs_2G;


HY_2V = @(b,y)sub2ind([h_num,y_control_num],b,y);
HYs_2V = @(b,ys)HY_2V(b*ones(size(ys)),ys);
HsY_2V = @(bs,y)HY_2V(bs,y*ones(size(bs)));

function_handles.HY_2V = HY_2V;
function_handles.HYs_2V = HYs_2V;
function_handles.HsY_2V = HsY_2V;

if(isfield(params,'l_num'))
    l_num = params.l_num;
    t_num = params.t_num;
    b_num = params.b_num;

    HL2B = @(h,l)sub2ind([h_num,l_num],h,l);
    B2HL = @(b)ind2sub([h_num,l_num],b);
    HLs2B = @(h,ls)HL2B(h*ones(size(ls)),ls);
    HsL2B = @(hs,l)HL2B(hs,l*ones(size(hs)));
    
    XB_2T = @(x,l)sub2ind([x_num,b_num],x,l);
    T2XB = @(t)ind2sub([x_num,b_num],t);
    XBs_2T = @(x,ls)XB_2T(x*ones(size(ls)),ls);
    XsB_2T = @(xs,l)XB_2T(xs,l*ones(size(xs)));

    YT_2W = @(y,t)sub2ind([y_control_num,t_num],y,t);
    YTs_2W = @(y,ts)YT_2W(y*ones(size(ts)),ts);
    YsT_2W = @(ys,t)YT_2W(ys,t*ones(size(ys)));


    function_handles.HL2B = HL2B;
    function_handles.B2HL = B2HL;
    function_handles.HsL2B = HsL2B;
    function_handles.HLs2B = HLs2B;

    function_handles.XB_2T = XB_2T;
    function_handles.T2XB = T2XB;
    function_handles.XBs_2T = XBs_2T;
    function_handles.XsB_2T = XsB_2T;

    function_handles.YT_2W = YT_2W;
    function_handles.YTs_2W = YTs_2W;
    function_handles.YsT_2W = YsT_2W;
end
end

