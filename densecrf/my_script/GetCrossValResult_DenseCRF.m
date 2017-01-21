root_dir = '/home/afriesen/proj/external/caffe';
dataset = 'sbd';
model = 'deeplab_resnet101';
featdir = 'features';
fold = 1;
%'post_densecrf_W4_XStd49_RStd5_PosW3_PosXStd3'
map_folder_folder = sprintf('%s/exper/%s/res/%s/%s/val.%d/fc1', ...
                root_dir, dataset, featdir, model, fold); 
map_dir_dir = dir(fullfile(map_folder_folder, '*numSample*'));

label_dir = '/home/afriesen/proj/sspn/data/iccv09Data/labels';

display_labels = false;

numtermlabels = 8;

best_acc = 0;
best_acc_dir = '';
best_jacc = 0;
best_jacc_dir = '';

for jj = 1:numel(map_dir_dir)
    y_true = [];
    y_dcrf = [];
    map_folder = fullfile(map_folder_folder, map_dir_dir(jj).name);
    map_dir = dir(fullfile(map_folder, '*.bin'));
    fprintf('processing folder %s\n', map_folder);
    for i = 1: numel(map_dir)
%         fprintf(1, 'processing file %d (%d)...', i, numel(map_dir));

        map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');
        labels = int16(dlmread(fullfile(label_dir, [map_dir(i).name(1:end-4) '.regions.txt']), ' '));

        y_true = [y_true; reshape(labels, [], 1)];
        y_dcrf = [y_dcrf; reshape(map, [], 1)];

        if display_labels,
            figure(1); imshow(labels, [-1, 7]); colormap(jet); title('true labels');
            figure(2); imshow(map, [-1, 7]); colormap(jet); title('dense CRF labels');
            pause(1);
        end

%         acc = 1 - sum(sum(bitand((labels ~= map), labels >= 0))) / sum(sum(labels >= 0));
%         fprintf( ' -- accuracy = %g \n', acc);
    end

    % ignore pixels with negative labels
    y_dcrf(y_true < 0) = [];
    y_true(y_true < 0) = [];

    % construct the confusion matrix
    confmat = confusionmat(y_true, y_dcrf);
    pix_acc = sum(diag(confmat)) / sum(sum(confmat));

    avg_acc = diag(confmat) ./ sum(confmat, 2);
    avg_acc(isinf(avg_acc) | isnan(avg_acc)) = 0.0;
    avg_acc = sum(avg_acc) / nnz(avg_acc);
    
    jacc = diag(confmat) ./ (sum(confmat, 1)' + sum(confmat, 2) - diag(confmat));
    jacc(isinf(jacc) | isnan(jacc)) = 1.0;
    avg_jacc = mean(jacc);
    
    avg_jacc_all = (sum(jacc) + 1.0*numtermlabels - numel(jacc)) / numtermlabels;
    
    fprintf('pix acc = %g, avg pix acc = %g, avg jacc = %g; [avg jacc (fewer) = %g]\n', ...
        pix_acc, avg_acc, avg_jacc_all, avg_jacc);
    if pix_acc > best_acc
        best_acc = pix_acc;
        best_acc_dir = map_dir_dir(jj).name;
    end
    if avg_jacc > best_jacc,
        best_jacc = avg_jacc;
        best_jacc_dir = map_dir_dir(jj).name;
    end
end

fprintf('best acc was %g from folder %s\n', best_acc, best_acc_dir);
fprintf('best jacc was %g from folder %s\n', best_jacc, best_jacc_dir);