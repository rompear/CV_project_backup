%% main function 


%% fine-tune cnn

[net, info, expdir] = finetune_cnn();

%% extract features and train svm

% TODO: Replace the name with the name of your fine-tuned model
nets.fine_tuned = load(fullfile(expdir, 'net-epoch-50.mat')); nets.fine_tuned = nets.fine_tuned.net;
nets.pre_trained = load(fullfile('data', 'pre_trained_model.mat')); nets.pre_trained = nets.pre_trained.net; 
data = load(fullfile(expdir, 'imdb-caltech.mat'));


%%
train_svm(nets, data);

%% extract features and train SVM classifiers, by validating their hyperparameters
nets.pre_trained.layers{end}.type = 'softmax';
nets.fine_tuned.layers{end}.type = 'softmax';
[svm.pre_trained.trainset, svm.pre_trained.testset] = get_svm_data(data, nets.pre_trained);
[svm.fine_tuned.trainset,  svm.fine_tuned.testset] = get_svm_data(data, nets.fine_tuned);

%%
train_X = svm.fine_tuned.trainset.features;
train_labels = svm.fine_tuned.trainset.labels;

pre_train_X = svm.pre_trained.trainset.features;
pre_train_labels = svm.pre_trained.trainset.labels;

% Run t−SNE
mappedX = tsne(full(train_X));
pre_mappedX = tsne(full(pre_train_X));

% Plot results
figure(1)
subplot(121)
gscatter(mappedX(:,1), mappedX(:,2), train_labels);
subplot(122)
gscatter(pre_mappedX(:,1), pre_mappedX(:,2), train_labels);


function [trainset, testset] = get_svm_data(data, net)

trainset.labels = [];
trainset.features = [];

testset.labels = [];
testset.features = [];
for i = 1:size(data.images.data, 4)
    
    res = vl_simplenn(net, data.images.data(:, :,:, i));
    feat = res(end-3).x; feat = squeeze(feat);
    
    if(data.images.set(i) == 1)
        
        trainset.features = [trainset.features feat];
        trainset.labels   = [trainset.labels;  data.images.labels(i)];
        
    else
        
        testset.features = [testset.features feat];
        testset.labels   = [testset.labels;  data.images.labels(i)];
        
        
    end
    
end

trainset.labels = double(trainset.labels);
trainset.features = sparse(double(trainset.features'));

testset.labels = double(testset.labels);
testset.features = sparse(double(testset.features'));
end