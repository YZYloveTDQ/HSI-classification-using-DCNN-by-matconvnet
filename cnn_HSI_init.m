function net = cnn_HSI_init(varargin)
%% --------------------------------------------------------------
% ------------------------------------------------------------------------

 
opts.batchNormalization = true ;            %ѡ��batchNormalizationΪtrue
opts.networkType = 'simplenn' ;             %ѡ��CNN�ṹΪsimplenn or dagnn
opts = vl_argparse(opts, varargin) ;        %����vl_argparse��ͨ���ⲿ�����޸ĳ�ʼֵ��
 
rng('default');                             %���������������������ÿ�����н��
rng(0) ;                                    
 
% ��ʼ��������ṹ��������LeNet5
f=1/100 ;                                   
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...          %�����C1��randn��������4ά��׼��̬�ֲ���������ƫ����20��
                           'weights', {{f*randn(3,3,3,20, 'single'), zeros(1, 20, 'single')}}, ...  %filter��С��5*5*1
                           'stride', 1, ...             %stride = 1
                           'pad', 0,'inputs','xno1') ;                  %pad = 0
net.layers{end+1} = struct('type', 'pool', ...          %�ػ���P1
                           'method', 'max', ...
                           'pool', [2 2], ...           %�ػ��˴�СΪ2*2
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...          %�����C2
                           'weights', {{f*randn(3,3,20,50, 'single'),zeros(1,50,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...          %�ػ���P2
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...          %�����C3
                           'weights', {{f*randn(3,3,50,500, 'single'),  zeros(1,500,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;            %ReLu��
net.layers{end+1} = struct('type', 'conv', ...          %FC��
                           'weights', {{f*randn(2,2,500,128, 'single'), zeros(1,128,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'conv', ...          %FC��
                           'weights', {{f*randn(1,1,128,16, 'single'), zeros(1,16,'single')}}, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss');     %softmax��
 
% optionally switch to batch normalization
if opts.batchNormalization                  %���opts.batchNormalizationΪ�棺
  net = insertBnorm(net, 1) ;               %��ԭ�����һ������Bnorm
  net = insertBnorm(net, 4) ;               %��ԭ������Ĳ�����Bnorm
  net = insertBnorm(net, 7) ;               %��ԭ������߲�����Bnorm
end
 
% Meta parameters �ṹԪ����
net.meta.inputSize = [25 25 3] ;            %��СΪ28*28*1��input data
net.meta.trainOpts.learningRate = 0.0001 ;   %ѧϰ��Ϊ0.001
net.meta.trainOpts.numEpochs = 1000 ;        %EpochΪ100
net.meta.trainOpts.batchSize = 128 ;        %���Ĵ�СΪ100
 
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
