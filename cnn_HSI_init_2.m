function net = cnn_HSI_init(varargin)
%% --------------------------------------------------------------
%   主函数： cnn_mnist_init
%   功能：   1.初始化CNN结构为LeNet
% ------------------------------------------------------------------------
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

opts.batchNormalization = true ;            %选择batchNormalization为true
opts.networkType = 'dagnn' ;                %选择CNN结构为simplenn
opts = vl_argparse(opts, varargin) ;        %调用vl_argparse（通过外部参数修改初始值）

rng('default');                             %设置随机数发生器，重现每次运行结果
rng(0) ;

% 开始构建网络结构，这里是LeNet5
%%====================C-R-C-R-P===========================
f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C1，randn函数产生4维标准正态分布矩阵，设置偏置有20个
    'weights', {{f*randn(3,3,3,8, 'single'), zeros(1, 8, 'single')}}, ...  %filter大小是5*5*1
    'stride', 1, ...             %stride = 1
    'pad', 1,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C2
    'weights', {{f*randn(3,3,8,8, 'single'),zeros(1,8,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'pool', ...          %池化层P2
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 1) ;
%%====================C-R-C-R-P===========================
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C3
    'weights', {{f*randn(3,3,8,16, 'single'),  zeros(1,16,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C3
    'weights', {{f*randn(3,3,16,16, 'single'),  zeros(1,16,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'pool', ...          %池化层P2
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 1) ;
%%====================C-R-C-R-P===========================
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C1，randn函数产生4维标准正态分布矩阵，设置偏置有20个
    'weights', {{f*randn(3,3,16,32, 'single'), zeros(1, 32, 'single')}}, ...  %filter大小是5*5*1
    'stride', 1, ...             %stride = 1
    'pad', 1,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C2
    'weights', {{f*randn(3,3,32,32, 'single'),zeros(1,32,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'pool', ...          %池化层P2
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad',1) ;
%%====================C-R-C-R===========================
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C1，randn函数产生4维标准正态分布矩阵，设置偏置有20个
    'weights', {{f*randn(3,3,32,64, 'single'), zeros(1, 64, 'single')}}, ...  %filter大小是5*5*1
    'stride', 1, ...             %stride = 1
    'pad', 1,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'conv', ...          %卷积层C2
    'weights', {{f*randn(3,3,64,64, 'single'),zeros(1,64,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
% net.layers{end+1} = struct('type', 'pool', ...          %池化层P2
%     'method', 'max', ...
%     'pool', [2 2], ...
%     'stride', 2, ...
%     'pad',1) ;
%%=====================================================
% net.layers{end+1} = struct('type', 'conv', ...          %FC层
%     'weights', {{f*randn(3,3,64,128, 'single'), zeros(1,128,'single')}}, ...
%     'stride', 1, ...
%     'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...          %FC层
    'weights', {{f*randn(3,3,64,16, 'single'), zeros(1,16,'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;            %ReLu层
net.layers{end+1} = struct('type', 'softmaxloss') ;     %softmax层

% optionally switch to batch normalization
if opts.batchNormalization                  %如果opts.batchNormalization为真：
    net = insertBnorm(net, 1) ;               %在原网络第一层后添加Bnorm
    net = insertBnorm(net, 4) ;               %在原网络第四层后添加Bnorm
    net = insertBnorm(net, 8) ;               %在原网络第一层后添加Bnorm
    net = insertBnorm(net, 11) ;               %在原网络第四层后添加Bnorm
    net = insertBnorm(net, 15) ;               %在原网络第七层后添加Bnorm
    net = insertBnorm(net, 18) ;               %在原网络第四层后添加Bnorm
    net = insertBnorm(net, 22) ;               %在原网络第一层后添加Bnorm
    net = insertBnorm(net, 25) ;               %在原网络第一层后添加Bnorm
end

% Meta parameters 结构元参数
net.meta.inputSize = [25 25 3] ;            %大小为28*28*1的input data
net.meta.trainOpts.learningRate = 0.0001 ;   %学习率为0.001
net.meta.trainOpts.numEpochs = 10 ;        %Epoch为100
net.meta.trainOpts.batchSize = 100 ;        %批的大小为100

% Fill in defaul values
net = vl_simplenn_tidy(net) ;               %添加默认的属性值

% Switch to DagNN if requested
switch lower(opts.networkType)              %选择网络结构
    case 'simplenn'                           %simplenn结构
        % done
    case 'dagnn'                              %dagnn结构
        net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
        net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
            {'prediction', 'label'}, 'error') ;
        net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
            'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
    otherwise                               %
        assert(false) ;
end

% --------------------------------------------------------------------
function net = insertBnorm(net, l)                      %
%% --------------------------------------------------------------
%   函数名：insertBnorm
%   功能：  1.在第l层和第l+1层之间插入Bnorm层
% ------------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));                              %断言以确保第l层有权重项
ndim = size(net.layers{l}.weights{1}, 4);                               %第l层的神经元的个数
layer = struct('type', 'bnorm', ...                                     %初始化Bnorm层
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1 0.05], ...                          %Bnorm层的权值=上一层的神经元个数
    'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;     %添加Bnorm层
