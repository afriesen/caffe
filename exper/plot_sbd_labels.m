% plot labels for iccv09 data

fold = '1';
testtype = 'test';

imgdir = '~/proj/sspn/data/iccv09Data/images/';
labeldir = '~/proj/sspn/data/iccv09Data/labels/';
resultdir = ['~/proj/external/caffe/exper/sbd/features2/deeplab_resnet101/' testtype '.' fold '/fc1'];

D = dir(fullfile(resultdir, '*.mat'));

for ii = 1:numel(D),
    im = imread(fullfile(imgdir, [D(ii).name(1:end-11) '.jpg']));
    truelbls = dlmread(fullfile(labeldir, [D(ii).name(1:end-11) '.regions.txt']), ' ');
    load(fullfile(resultdir, D(ii).name));
    unaries = data;
    
    [~, predlbls] = max(unaries, [], 3);
    predlbls = rot90(fliplr(predlbls-1 ));
    
    errors = bitand(predlbls ~= truelbls, truelbls >= 0);
    
    figure(1); imshow(im); title(sprintf('image %s', [D(ii).name(1:end-11) '.jpg']));
    figure(2); imshow(truelbls, [0, 7]); colormap(jet); title('true labels');
    figure(3); imshow(predlbls, [0, 7]); colormap(jet); title('predicted labels');
    figure(4); imshow(errors, [0, 1]); colormap(jet); title('errors');
    
    acc = sum(predlbls == truelbls) / sum(truelbls >= 0);
    
    fprintf('%d / %d: image %s: avg acc = %g\n', ii, numel(D), [D(ii).name(1:end-11) '.jpg'], acc );
    
    pause(0.5);
end