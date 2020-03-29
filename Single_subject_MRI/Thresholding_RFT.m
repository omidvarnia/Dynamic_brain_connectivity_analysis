%%%
% A script thresholding z-maps based on Gaussian Random Field Theory (Friston et al. 1994. Human Brain Mapping)
% This code is amended the from DPABI toolbox

clc
fprintf('\n\t Thresholding using Gaussian Random Field Theory ... \n');

[BrainVolume,VoxelSize,Header] = y_ReadRPI(in_file);

Header_DF = w_ReadDF(Header);
Header.DF.TestFlag = 'z';
Header.DF.Df = 0;

fprintf('\nSmoothness and other information of MRI map: \n')
[dLh,resels,FWHM,nVoxels] = y_Smoothest(in_file,mask,0,VoxelSize);
zThrd = norminv(1 - VoxelPThreshold/2);
fprintf('\n Voxel-level threshold: p < %g (Z = %g) \n',VoxelPThreshold,zThrd);

% Calculate Expectations of m clusters Em and exponent Beta for inference.
D = 3;
Em = nVoxels * (2*pi)^(-(D+1)/2) * dLh * (zThrd*zThrd-1)^((D-1)/2) * exp(-zThrd*zThrd/2);
EN = nVoxels * (1-normcdf(zThrd)); 
Beta = ((gamma(D/2+1)*Em)/(EN)) ^ (2/D); 

% Get the minimum cluster size
pTemp = 1;
ClusterSize = 0;

while pTemp >= ClusterPThreshold
    ClusterSize = ClusterSize+1;
    pTemp = 1 - exp(-Em * exp(-Beta * ClusterSize^(2/D))); 
    
end

fprintf('\n Cluster-level threshold: p < %g = %g connected voxels \n',ClusterPThreshold,ClusterSize);

ConnectivityCriterion = 26; % a voxel is fully connected - in 3D space

if ClusterSize > 0
    BrainVolumeNegative = BrainVolume .* (BrainVolume <= -1*zThrd);
    [theObjMask,theObjNum] = bwlabeln(BrainVolumeNegative,ConnectivityCriterion);
    
    for x=1:theObjNum
        theCurrentCluster = theObjMask == x;
        if length(find(theCurrentCluster)) < ClusterSize
            BrainVolumeNegative(logical(theCurrentCluster)) = 0;
            
        end
    end
    
    % Correct positive values to Cluster P
    BrainVolume = BrainVolume .* (BrainVolume >= zThrd);
    [theObjMask,theObjNum] = bwlabeln(BrainVolume,ConnectivityCriterion);
    
    for x=1:theObjNum
        theCurrentCluster = theObjMask == x;
        if length(find(theCurrentCluster)) < ClusterSize
            BrainVolume(logical(theCurrentCluster)) = 0;
            
        end
    end
    
    BrainVolume = BrainVolume + BrainVolumeNegative;

end

% report number of cluster - only make spatial map if n_clust > 0
[~,n_cluster] = bwlabeln(BrainVolume,ConnectivityCriterion);
if n_cluster > 0
    fprintf('\n There are %g suprathreshold clusters \n',n_cluster);
    y_Write(BrainVolume,Header,['Z_score_RFT_voxel_thr_' num2str(VoxelPThreshold) '_and_cluster_thr_' num2str(ClusterPThreshold) '_subject_' cell2mat(string) '.nii']);
    
else
    fprintf('\n There are no suprathreshold clusters!!! \n');
    
end