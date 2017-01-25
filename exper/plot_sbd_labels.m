% plot labels for iccv09 data

net_id = 'sspn_scenecat_deeplab_resnet101';
% net_id = 'sspn_scenecat_shared_deeplab_resnet101';
featdir = 'features';
fold = '1';
testtype = 'test';
caffelayer = 'sspn';
numtermlabels = 8;

imgdir = '~/proj/sspn/data/iccv09Data/images/';
labeldir = '~/proj/sspn/data/iccv09Data/labels/';
resultdir = ['~/proj/external/caffe/exper/sbd/' featdir '/' net_id '/' testtype '.' fold '/' caffelayer];

D = dir(fullfile(resultdir, '*.mat'));

% sort by creation time
% [~,index] = sortrows({D.date}.'); D = D(index(end:-1:1)); clear index; % descending
[~,index] = sortrows({D.date}.'); D = D(index); clear index; % ascending

for ii = 1:numel(D),
    im = imread(fullfile(imgdir, [D(ii).name(1:end-11) '.jpg']));
    truelbls = dlmread(fullfile(labeldir, [D(ii).name(1:end-11) '.regions.txt']), ' ');
    load(fullfile(resultdir, D(ii).name));
    unaries = data;
    
    [~, predlbls] = max(unaries, [], 3);
    predlbls = rot90(fliplr(predlbls-1 ));
    
    errors = bitand(predlbls ~= truelbls, truelbls >= 0);
    
    figure(1);
    subplot(2,2,1); imshow(im); title(sprintf('image %s', [D(ii).name(1:end-11) '.jpg']));
    subplot(2,2,2); imshow(truelbls, [0, 7]); colormap(jet); title('true labels');
    subplot(2,2,4); imshow(predlbls, [0, 7]); colormap(jet); title('predicted labels');
    subplot(2,2,3); imshow(errors, [0, 1]); colormap(jet); title('errors');
    
    % drop pixels with negative (unknown) labels
    predlbls(truelbls < 0) = [];
    truelbls(truelbls < 0 ) = [];
    
    acc = sum(predlbls == truelbls) / sum(truelbls >= 0);
    
    clear cm;
    cm = confusionmat(reshape(truelbls,[],1), reshape(predlbls,[],1));
%     assert( acc == sum(diag(cm)) / sum(sum(cm)) );
    
    avgacc = diag(cm) ./ sum(cm,2);
    avgacc(isinf(avgacc) | isnan(avgacc)) = 0.0;
    avgacc = sum(avgacc) / nnz(avgacc);
    
    jacc = diag(cm) ./ (sum(cm, 1)' + sum(cm, 2) - diag(cm));
    jacc(isinf(jacc) | isnan(jacc)) = 1.0;
    avgjacc = mean(jacc);
    
    avgjaccall = (sum(jacc) + 1.0*(numtermlabels - numel(jacc))) / numtermlabels;
    
    fprintf('%d / %d: image %s: mAP = %g, avg mAP = %g, avg jacc = %g;    [avg jacc (reduced) = %g]\n', ...
        ii, numel(D), [D(ii).name(1:end-11) '.jpg'], acc, avgacc, avgjaccall, avgjacc );
    
    %pause(2);
%     keydown = waitforbuttonpress;
%     fprintf('keydown = '); disp(keydown); fprintf('\n');
    if ii > 1; break; end
end