function [FDts,Stat]=FDCalc(MP,varargin)
% [FDts,SS]=FDCalc(MP,drflag,radius)
%   Calculates the Frame-Wise Displacement
%
%%%INPUTS:
%
%   MP :   Movement Parameters (regressors). Should be a matrix of
%   size TxP. Where P is number of params, and T is number of volumes.
%   drflag  :   Rotation and Disp indicator. if a param is for rotation set
%   the index to 1 otherwise to 0. e.g. drflag=[0 0 0 1 1 1] for three
%   disp param and three rotation param. 
%
%%%OUTPUTS:
%
%   FDts                : Frame-Wise Displacement
%   Stat.SS             : Sum of square of the movements.
%   Stat.FD_0{2,5}_Idx  : Index of scans exceeding the 0.2/0.5mm threshold
%   Stat.FD_0{2,5}_p    : % of the scans explained above
%   Stat.AbsRot         : Absolute sum of rotation dip 
%   Stat.AbsTrans       : Absolute sum of translational disp
%   Stat.AbsRot & Stat.AbsRot : Absolute sum of one-lag difference 
%
%%%NOTES:
%           1) Rotation params should be in degree 
%           2) Only first six params takes into account
%
%_________________________________________________________________________
% Soroosh Afyouni, NISOx.org, 2017
% srafyouni@gmail.com
fnnf=mfilename; if ~nargin; help(fnnf); return; end; clear fnnf;
%_________________________________________________________________________


if isempty(varargin)
    drflag = [0 0 0 1 1 1]; radius = 50;          verbose = 1;
elseif nargin==2
    drflag = varargin{1};   radius = 50;          verbose = 1;
elseif nargin==3
    drflag = varargin{1};   radius = varargin{2}; verbose = 1;
elseif nargin==4
    drflag = varargin{1};   radius = varargin{2}; verbose = varargin{3};
else
    error('FDCalc :: Too many inputs.')
end
verbose = 0;

if size(MP,2) > size(MP,1)
    error('FDCalc :: Check dim of params. It should be TxP. Use MP=transpose(MP) if necessary.')
end

if size(MP,2) > 6 && verbose
    warning('-Only first six colums is taken to acount as mov par.')
end

r_Idx   = find(drflag);
t_idx   = find(~drflag);

if verbose
    disp(['-Rotation Index:' num2str(r_Idx) ', Dsplcmnt Index: ' num2str(t_idx)])
end

MP          = MP(:,1:6);
MP(:,r_Idx) = (2*radius*pi/360)*MP(:,r_Idx); %Oi! Degree Degree!

dMP  = diff(MP);
FDts = sum(abs(dMP),2);
SS   = sqrt(sum(dMP.^2,2));

Stat.SS         = SS;
Stat.FD_02_Idx  = find(FDts>0.2);
Stat.FD_05_Idx  = find(FDts>0.5);
Stat.FD_02_p    = length(Stat.FD_02_Idx)./length(FDts)*100;
Stat.FD_05_p    = length(Stat.FD_05_Idx)./length(FDts)*100;
Stat.AbsRot     = sum(abs(MP(:,r_Idx)),2);
Stat.AbsTrans   = sum(abs(MP(:,t_idx)),2);
Stat.AbsDiffRot     = sum(abs(diff(MP(:,r_Idx))),2);
Stat.AbsDiffTrans   = sum(abs(diff(MP(:,t_idx))),2);
