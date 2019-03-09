function net = cnn_HSI_init(varargin)
%% --------------------------------------------------------------
%   �������� cnn_mnist_init
%   ���ܣ�   1.��ʼ��CNN�ṹΪLeNet
% ------------------------------------------------------------------------
% CNN_MNIST_LENET Initialize a CNN similar for MNIST

opts.batchNormalization = true ;            %ѡ��batchNormalizationΪtrue
opts.networkType = 'dagnn' ;                %ѡ��CNN�ṹΪsimplenn
opts = vl_argparse(opts, varargin) ;        %����vl_argparse��ͨ���ⲿ�����޸ĳ�ʼֵ��

rng('default');                             %���������������������ÿ�����н��
rng(0) ;

% ��ʼ��������ṹ��������LeNet5
%%====================C-R-C-R-P===========================
f=1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...          %�����C1��randn��������4ά��׼��̬�ֲ���������ƫ����20��
    'weights', {{f*randn(3,3,3,8, 'single'), zeros(1, 8, 'single')}}, ...  %filter��С��5*5*1
    'stride', 1, ...             %stride = 1
    'pad', 1,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'conv', ...          %�����C2
    'weights', {{f*randn(3,3,8,8, 'single'),zeros(1,8,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'pool', ...          %�ػ���P2
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 1) ;
%%====================C-R-C-R-P===========================
net.layers{end+1} = struct('type', 'conv', ...          %�����C3
    'weights', {{f*randn(3,3,8,16, 'single'),  zeros(1,16,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'conv', ...          %�����C3
    'weights', {{f*randn(3,3,16,16, 'single'),  zeros(1,16,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'pool', ...          %�ػ���P2
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad', 1) ;
%%====================C-R-C-R-P===========================
net.layers{end+1} = struct('type', 'conv', ...          %�����C1��randn��������4ά��׼��̬�ֲ���������ƫ����20��
    'weights', {{f*randn(3,3,16,32, 'single'), zeros(1, 32, 'single')}}, ...  %filter��С��5*5*1
    'stride', 1, ...             %stride = 1
    'pad', 1,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'conv', ...          %�����C2
    'weights', {{f*randn(3,3,32,32, 'single'),zeros(1,32,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'pool', ...          %�ػ���P2
    'method', 'max', ...
    'pool', [2 2], ...
    'stride', 2, ...
    'pad',1) ;
%%====================C-R-C-R===========================
net.layers{end+1} = struct('type', 'conv', ...          %�����C1��randn��������4ά��׼��̬�ֲ���������ƫ����20��
    'weights', {{f*randn(3,3,32,64, 'single'), zeros(1, 64, 'single')}}, ...  %filter��С��5*5*1
    'stride', 1, ...             %stride = 1
    'pad', 1,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'conv', ...          %�����C2
    'weights', {{f*randn(3,3,64,64, 'single'),zeros(1,64,'single')}}, ...
    'stride', 1, ...
    'pad', 1) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
% net.layers{end+1} = struct('type', 'pool', ...          %�ػ���P2
%     'method', 'max', ...
%     'pool', [2 2], ...
%     'stride', 2, ...
%     'pad',1) ;
%%=====================================================
% net.layers{end+1} = struct('type', 'conv', ...          %FC��
%     'weights', {{f*randn(3,3,64,128, 'single'), zeros(1,128,'single')}}, ...
%     'stride', 1, ...
%     'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...          %FC��
    'weights', {{f*randn(3,3,64,16, 'single'), zeros(1,16,'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'softmaxloss') ;     %softmax��

% optionally switch to batch normalization
if opts.batchNormalization                  %���opts.batchNormalizationΪ�棺
    net = insertBnorm(net, 1) ;               %��ԭ�����һ������Bnorm
    net = insertBnorm(net, 4) ;               %��ԭ������Ĳ�����Bnorm
    net = insertBnorm(net, 8) ;               %��ԭ�����һ������Bnorm
    net = insertBnorm(net, 11) ;               %��ԭ������Ĳ�����Bnorm
    net = insertBnorm(net, 15) ;               %��ԭ������߲�����Bnorm
    net = insertBnorm(net, 18) ;               %��ԭ������Ĳ�����Bnorm
    net = insertBnorm(net, 22) ;               %��ԭ�����һ������Bnorm
    net = insertBnorm(net, 25) ;               %��ԭ�����һ������Bnorm
end

% Meta parameters �ṹԪ����
net.meta.inputSize = [25 25 3] ;            %��СΪ28*28*1��input data
net.meta.trainOpts.learningRate = 0.0001 ;   %ѧϰ��Ϊ0.001
net.meta.trainOpts.numEpochs = 10 ;        %EpochΪ100
net.meta.trainOpts.batchSize = 100 ;        %���Ĵ�СΪ100

% Fill in defaul values
net = vl_simplenn_tidy(net) ;               %���Ĭ�ϵ�����ֵ

% Switch to DagNN if requested
switch lower(opts.networkType)              %ѡ������ṹ
    case 'simplenn'                           %simplenn�ṹ
        % done
    case 'dagnn'                              %dagnn�ṹ
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
%   ��������insertBnorm
%   ���ܣ�  1.�ڵ�l��͵�l+1��֮�����Bnorm��
% ------------------------------------------------------------------------
assert(isfield(net.layers{l}, 'weights'));                              %������ȷ����l����Ȩ����
ndim = size(net.layers{l}.weights{1}, 4);                               %��l�����Ԫ�ĸ���
layer = struct('type', 'bnorm', ...                                     %��ʼ��Bnorm��
    'weights', {{ones(ndim, 1, 'single'), zeros(ndim, 1, 'single')}}, ...
    'learningRate', [1 1 0.05], ...                          %Bnorm���Ȩֵ=��һ�����Ԫ����
    'weightDecay', [0 0]) ;
net.layers{l}.biases = [] ;
net.layers = horzcat(net.layers(1:l), layer, net.layers(l+1:end)) ;     %���Bnorm��
