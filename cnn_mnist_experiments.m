%% Experiment with the cnn_mnist_fc_bnorm
clc;
clear
% addpath data
[net_bn, info_bn] = cnn_mnist('expDir', 'data/mnist-bnorm', 'batchNormalization', true);


% [net_bn, info_bn,OA,AA,Kappa,CA] = cnn_mnist('expDir', 'data/mnist-bnorm', 'batchNormalization', true);

load ('E:\下载资料\mnist_matconvnet\data\mnist-bnorm\imdb.mat');
test_index = find(images.set==3);
% 挑选出样本以及真实类别
test_data = images.data(:,:,:,test_index);
test_label = images.labels(test_index);
load ('E:\下载资料\mnist_matconvnet\data\mnist-bnorm\net-epoch-10.mat');
net.layers{1, end}.type = 'softmax';

% test_data = single(test_data);
% test_data = gpuArray(test_data);
[OA,AA,Kappa,CA] = process_test(net_bn,test_data,test_label);
net = load('imagenet-vgg-verydeep-16');
% net.layers{1, end}.type = 'softmax';
% net.layers{1, end+1} = net.layers{1, end};
% net.layers{1, end-1} = net.layers{1, end-2};
% net.layers{1, end-1}.size = [1,1,1000,16];

% net = vl_simplenn_tidy(net) ;
% I = imresize3(test_data(:,:,:,1),[224,224,3]);
% yres = vl_simplenn(net, I) ; 

for i = 1:length(test_label)
    im_ = test_data(:,:,:,i);
%         im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    im_ = im_ - images.data_mean;
    res = vl_simplenn(net, im_) ;
    scores = squeeze(gather(res(end).x)) ;
    [bestScore, best] = min(scores);
    pre(i) = best
end
for i = 1:length(test_label)
    im_ = test_data(:,:,:,i);
%     im_ = bsxfun(@minus, im_, net1.meta.normalization.averageImage) ;
    images = single(im_);
    images = gpuArray(images);
    inputs = {'input',images};   
    net.eval(inputs) ;
%     scores = squeeze(gather(res(end).x)) ;
%     [bestScore, best] = max(scores);
%     pre(i) = best;
    scores = squeeze(gather(net.vars(length(net.vars)-3).value));
    
    [~,tmp] = max(scores,[],1);
    prediction_ms(i) = [tmp];

end


accurcy = length(find(pre==test_label))/length(test_label);
disp(['accurcy = ',num2str(accurcy*100),'%']);
err_cnt=0; 
test_set=single(test_data);
train_imdb_path=fullfile('E:\下载资料\mnist_matconvnet\data\mnist-bnorm\imdb.mat');
train_imdb=load(train_imdb_path);
train_data_mean=train_imdb.images.data_mean;
test_set=bsxfun(@minus,test_set,train_data_mean);
% best_scores=zeros(1,female_num+male_num);
% err_id=zeros(1,female_num+male_num);
for i=1:length(test_label)
    res=vl_simplenn(net,test_set(:,:,:,i));
    scores=squeeze(gather(res(end).x));
    [best_score, best_id] = max(scores);
    best_scores(i)=best_score;
    if abs(str2double(net.meta.classes.name{best_id})-test_label(i))>1e-1;
        err_cnt=err_cnt+1;
        err_id(i)=1;
    end
    
end
err_rate=err_cnt/(female_num+male_num);
[net_fc, info_fc] = cnn_mnist('expDir', 'data/mnist-baseline', 'batchNormalization', false);

figure(1) ; clf ;
subplot(1,2,1) ;
semilogy([info_fc.val.objective]', 'o-') ; hold all ;
semilogy([info_bn.val.objective]', '+--') ;
xlabel('Training samples [x 10^3]'); ylabel('energy') ;
grid on ;
h=legend('BSLN', 'BNORM') ;
set(h,'color','none');
title('objective') ;
subplot(1,2,2) ;
plot([info_fc.val.top1err]', 'o-') ; hold all ;
plot([info_fc.val.top5err]', '*-') ;
plot([info_bn.val.top1err]', '+--') ;
plot([info_bn.val.top5err]', 'x--') ;
h=legend('BSLN-val','BSLN-val-5','BNORM-val','BNORM-val-5') ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title('error') ;
drawnow ;
