function test_accuracy()


load ('E:\下载资料\mnist_matconvnet\data\mnist-bnorm\imdb.mat');
test_index = find(images.set==3);
% 挑选出样本以及真实类别
test_data = images.data(:,:,:,test_index);
test_label = images.labels(test_index);
load ('E:\下载资料\mnist_matconvnet\data\mnist-bnorm\net-epoch-100.mat');
net.layers{1, end}.type = 'softmax';

test_data = single(test_data);
% test_data = gpuArray(test_data);
for i = 1:length(test_label)
    im_ = test_data(:,:,:,i);
%     im_ = im_ - images.data_mean;
    res = vl_simplenn(net, im_) ;
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = max(scores);
    pre(i) = best;
end
accurcy = length(find(pre==test_label))/length(test_label);
disp(['accurcy = ',num2str(accurcy*100),'%']);
