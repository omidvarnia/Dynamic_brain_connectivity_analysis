clc
fprintf('\n\t Thresholding using False Discovery Rate ... \n');

MaskData = y_Read(mask);
MaskDataOneDim = reshape(MaskData,1,[]);

[zmap,Header] = y_Read(in_file);
[nDim1, nDim2, nDim3, nDimTimePoints] = size(zmap);
zmap = reshape(zmap,[],1)';
zmap = zmap(:,find(MaskDataOneDim));
zmap(isnan(zmap)) = eps;
zmap(isinf(zmap)) = eps;

pvals = 1-normcdf(abs(zmap));
m = length(pvals);
p_sorted = sort(pvals);

thresh = (1:m) * QThreshold / m;
wtd_p = m * p_sorted ./ (1:m);

rej = p_sorted<=thresh;
max_id = find(rej,1,'last'); %find greatest significant pvalue

if max_id ~= 0
    crit_p = p_sorted(max_id);
    fd = pvals <= crit_p;
    
    out = zeros(size(MaskDataOneDim));
    out(1,find(MaskDataOneDim)) = zmap.*fd;
    out = reshape(out,nDim1, nDim2, nDim3);
    
    y_Write(out,Header,['Z_score_FDR_Qthr_' num2str(QThreshold) '_subject_' cell2mat(string) '.nii']);
    fprintf('\n There are %g FDR-corrected voxels above critical threshold %g \n',max_id,crit_p);
    
else
    fprintf('\n There are no FDR-corrected voxels!!! \n');
    
end