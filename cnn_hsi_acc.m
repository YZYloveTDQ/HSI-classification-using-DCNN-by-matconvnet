%% Experiment with the cnn_mnist_fc_bnorm
clc;
clear
addpath data
[net_bn, info_bn,OA,AA,Kappa,CA,accu_martix] = cnn_HSI('expDir', 'data/mnist-bnorm', 'batchNormalization', true);

% load ('E:\ÏÂÔØ×ÊÁÏ\mnist_matconvnet\data\mnist-bnorm\imdb.mat');
load ('.\data\mnist-bnorm\imdb.mat');
test_index = find(images.set==3);
test_data = images.data(:,:,:,test_index);
test_label = images.labels(test_index);
% load ('.\data\mnist-bnorm\net-epoch-535.mat');

[OA,AA,Kappa,CA] = process_test(net_bn,test_data,test_label);
accu_martix = [CA;OA;AA;Kappa];
disp(['test accuracy: ',num2str(OA)])