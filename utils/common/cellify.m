function val = cellify(val)
% CELLIFY

% Revised: 7-8-2021
% Copyright 2021 The MathWorks, Inc.

if iscell(val)
    % force to be a row vector cell
    val = val(:)';
else
    val = {val};
end

