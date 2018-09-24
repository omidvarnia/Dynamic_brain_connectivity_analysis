function iPS = IPSA(rsfMRI_2D)
% Instantaneous phase synchrony analysis (IPSA) for estimating time-resolved fMRI connectivity: 
%
% Input:     rsfMRI_2D,      2D narrow band-pass filtered fMRI data (time-points x nodes, or T x N).
%
% Output:    iPS,           3D IPSA (NxNxT) - NB! high storage capacity needed with many nodes and/or time-points.
%
% Reference: Pedersen M, Omidvarnia A, Zalesky A, & Jackson G.J. 2018. NeuroImage. 181:85-94.

fprintf('\n\t Instantaneous phase synchrony analysis (IPSA):\n')
reverseStr = '';
theElapsedTime = tic;

iPS = zeros(size(rsfMRI_2D,2),size(rsfMRI_2D,2),size(rsfMRI_2D,1)); % pre-allocate 3D IPSA matrix
inst_phase = angle(hilbert(rsfMRI_2D)); % hilbert transform of fMRI data

for t = 1:size(iPS,3)
    ip = 1-abs(sin(bsxfun(@minus,inst_phase(t,:)',inst_phase(t,:)))); % calculate IPSA matrix for each time-point, t 
    
    ip(isnan(ip)) = 0; % ensure there are no NaN values in matrix
    ip(isinf(ip)) = 0; % ensure there are no inf values in matrix
    ip = ip-eye(size(ip,1)); % convert matrix diagonal to zero
    
    iPS(:,:,t) = ip; % store 3D IPSA matrix (NxNxT)
    
    msg = sprintf('\n\t - Time-point %d of %d ...\n',t,size(iPS,3));
    fprintf([reverseStr,msg]);
    reverseStr = repmat(sprintf('\b'),1,length(msg));
    
end

theElapsedTime = toc(theElapsedTime);
fprintf('\n\t - Elapsed time: %g seconds.\n', theElapsedTime);