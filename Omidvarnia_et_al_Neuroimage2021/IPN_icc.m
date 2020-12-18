%% Computes the interclass correlations for indexing the reliability analysis
%% according to shrout & fleiss' schema.
function ICC = IPN_icc(x,cse,typ)
% Ref: https://au.mathworks.com/matlabcentral/fileexchange/22122-ipn-tools-for-test-retest-reliability-analysis
% INPUT:
%   x   - ratings data matrix, data whose columns represent different
%         ratings/raters & whose rows represent different cases or
%         targets being measured. Each target is assumed too be a random
%         sample from a population of targets.
%   cse - 1 2 or 3: 1 if each target is measured by a different set of
%         raters from a population of raters, 2 if each target is measured
%         by the same raters, but that these raters are sampled from a
%         population of raters, 3 if each target is measured by the same
%         raters and these raters are the only raters of interest.
%   typ - 'single' or 'k': denotes whether the ICC is based on a single
%         measurement or on an average of k measurements, where
%         k = the number of ratings/raters.
%
% REFERENCE:
%   Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
%   reliability. Psychol Bull. 1979;86:420-428
%
% NOTE:
%   This code was mainly modified with the Kevin's codes in web.
%   (London kevin.brownhill@kcl.ac.uk)
%
% XINIAN ZUO
% Email: zuoxinian@gmail.com

% if isanova
%     [p,table,stats] = anova1(x',{},'off');
%     ICC=(table{2,4}-table{3,4})/(table{2,4}+table{3,3}/(table{2,3}+1)*table{3,4});
% else

%k is the number of raters, and n is the number of tagets
[N_subj, N_run] = size(x);

%mean per target
mpt = mean(x,2);

%mean per rater/rating
mpr = mean(x);

%get total mean
tm = mean(x(:));

%within target sum sqrs
tmp = (x - repmat(mpt,1,N_run)).^2;
WSS = sum(tmp(:));

%within target mean sqrs
WMS = WSS / (N_subj*(N_run - 1));

%between rater sum sqrs
RSS = sum((mpr - tm).^2) * N_subj;

%between rater mean sqrs
RMS = RSS / (N_run - 1);

%between target sum sqrs
BSS = sum((mpt - tm).^2) * N_run;

%between targets mean squares
BMS = BSS / (N_subj - 1);

%residual sum of squares
ESS = WSS - RSS;

%residual mean sqrs
EMS = ESS / ((N_run - 1) * (N_subj - 1));

switch cse
    case 1
        switch typ
            case 'single'
                ICC = (BMS - WMS) / (BMS + (N_run - 1) * WMS);
            case 'k'
                ICC = (BMS - WMS) / BMS;
            otherwise
                error('Wrong value for input typ')
        end
    case 2
        switch typ
            case 'single'
                ICC = (BMS - EMS) / (BMS + (N_run - 1) * EMS + N_run * (RMS - EMS) / N_subj);
            case 'k'
                ICC = (BMS - EMS) / (BMS + (RMS - EMS) / N_subj);
            otherwise
                error('Wrong value for input typ')
        end
    case 3
        switch typ
            case 'single'
                ICC = (BMS - EMS) / (BMS + (N_run - 1) * EMS);
            case 'k'
                ICC = (BMS - EMS) / BMS;
            otherwise
                error('Wrong value for input typ')
        end
    otherwise
        error('Wrong value for input cse')
end