%
% val = getvalue(S, fld, dval)
%
% returns field fld of struct S, or dval if not present
% fld can also be dot-delimited list of fields for nested structs
%
%
function val = getvalue(S, flds, dval)

val = dval;

flds = regexp(flds, '\.', 'split');
for i = 1:length(flds)-1
    fld = flds{i};
    if isfield(S, fld)
        S = getfield(S, fld);
    else
        return
    end
end

fld = flds{end};
if isfield(S, fld)
    val = getfield(S, fld);
end
