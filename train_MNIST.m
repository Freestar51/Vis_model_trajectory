% Train NN with DBN pre-trained and not
% Using DeepLearnToolbox library, train MNIST twice,
% storing trajectory data.
% 
% Written by Giyoung Jeon
% Probabilistic Artificial Intelligence Lab at UNIST
% v1.3 May, 26th, 2015

addpath(genpath('./'));
load('./data/mnist_uint8.mat');

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng('default'),rng(0);
dbn.sizes = [100];
opts.numepochs =   10;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;    %learning rate
opts.cdn       =   10;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rng(0);
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
traj_comp = {};
parfor pre_train=1:2
    if(pre_train==1)
        nn = dbnunfoldtonn(dbn, 10);
    else
        nn = nnsetup([dbn.sizes 10]);
    end
    nn.activation_function = 'sigm';

    %train nn
    [nn, traj_comp{pre_train}] = nntrain(nn, train_x, train_y, opts);
%     [er, bad] = nntest(nn, test_x, test_y);
end

for pt=1:2
    if(pt==1)
        prefix = 'pre_trained_';
    else
        prefix = 'no_pre_trained_';
    end
    for batch=1:600
        tmp = traj_comp{pt}{batch,1};
        save(strcat(prefix,int2str(batch)), 'tmp');
    end
end

clear all;