%% Clear all output
clear all

%% Image paths
data_main_path = "./Caltech4/";
image_path = "ImageData/";
test_set = ["Annotation/airplanes_test.txt", "Annotation/cars_test.txt", "Annotation/faces_test.txt", "Annotation/motorbikes_test.txt"]; 
train_set = ["Annotation/airplanes_train.txt", "Annotation/cars_train.txt", "Annotation/faces_train.txt", "Annotation/motorbikes_train.txt"] ; 
classes_set = [Classes.Airplanes, Classes.Cars, Classes.Faces, Classes.Motorbikes];

train_image_set = string([[], [], [], []]);
test_image_set = string([[], [], [], []]);

% Paramater settings
n = 400;
step_size = 20;
ksize = 2000;
sift_type = "dense";
color_space = "gray";
sift_method_string = "vlfeat_sift";

step_size_sift_string = "-";
block_size_sift_string = "4x4"; 
if sift_type == "dense"
    step_size_sift_string = num2str(step_size);
    block_size_sift_string = "4x4"; 
end

fraction = round(n / 4);
pos_sample = fraction;
neg_sample = fraction * 3;
vocab_fraction = n * 2 / 4;
n_total = n * 4;
fprintf("START \n")

%% READ DATA 
% read train set
fprintf("READ TRAIN \n")
for i = 1:size(classes_set,2) 
    % get path
    full_train_path = data_main_path + train_set(1, i);
   
    % read text file
    fid = fopen(full_train_path);
    tline = fgetl(fid);
    j = 1;
    while ischar(tline)
        line_cells = strsplit(tline, ' ');
        path = line_cells{1};
        relevant = line_cells{2};
        
        if relevant == '1' 
            % Store path
            train_image_set(i, j) =  string(data_main_path + image_path + path + '.jpg');
            if j < n
                j = j + 1;
            end
        end
        tline = fgetl(fid);
    end
    fclose(fid);
end

fprintf("READ TEST \n")
% read test set1000
for i = 1:size(classes_set,2) 
    % get path
    full_test_path = data_main_path + test_set(1, i);
    
    % read text file
    fid = fopen(full_test_path);
    tline = fgetl(fid);
    j = 1;
    while ischar(tline)
        line_cells = strsplit(tline, ' ');
        path = line_cells{1};
        relevant = line_cells{2};
        
        if relevant == '1' 
            % Store path
            test_image_set(i, j) =  string(data_main_path + image_path + path + '.jpg');
            j = j + 1;
        end
        
        tline = fgetl(fid);
    end
    fclose(fid);
end

%% PreProcessing
% set path to sift features
run('~/Desktop/vlfeat/toolbox/vl_setup')

%Loop over all the images
Cfeatures = [];
all_features = [];
idx = 1;

image_feature = containers.Map;
image_class = containers.Map;
fprintf("TRAIN DATA \n")
%for train_set
for i = 1:size(classes_set,2)
    counter = 0;
    
    for j = 1:size(train_image_set(i, :), 2)   
        %take image from set
        image = imread(char(train_image_set(i,j)));
        
        [~, ~, channels] = size(image);
        if channels ~= 3
            continue
        end
        
        if mod(counter, 10) == 0
            fprintf("%d , ", counter);
        end
        
        [frames, des] = get_features(image, sift_type, color_space, step_size); 
        
        %increase counter
        counter = counter + 1;
        
        % only loop over all sift features if counter < fraction
        if counter < round(fraction)
            Cfeatures = vertcat(des', Cfeatures);
        else
            image_feature(char(train_image_set(i,j))) = des;
            image_class(char(train_image_set(i,j))) = classes_set(i);
        end
    end
    fprintf("\n")
end

fprintf("TEST DATA \n")
% get Test_set features 
test_image_feature = containers.Map;
for i = 1:size(classes_set,2)
    counter = 0;
    for j = 1:size(test_image_set, 2)  
        % calculate features
        image = imread(char(test_image_set(i,j)));
                
        [~, ~, channels] = size(image);
        if channels ~= 3
            continue
        end
        [frames, des] = get_features(image, sift_type, color_space, step_size); 
        
        test_image_feature(char(test_image_set(i,j))) = des;
        
        image_class(char(test_image_set(i,j))) = classes_set(i);
    end
end


%% Make vocabulary
% KMEANS
fprintf("START KMEANS \n")
tic; 
[~, C] = kmeans(single(Cfeatures) , ksize, 'MaxIter', 500); 
toc

fprintf("ASSIGNING WORDS TRAIN \n")
image_hist = [];
image_hist_class = [Classes.Faces];
image_hist_class(1) = [];

% assign features (train) to clusters
for key = keys(image_feature) 
    des = image_feature(char(key));
    fhistogram = zeros(1, ksize);
    dists = [];

    %Loop over all features of an image    
    X = des';
  
    for i = 1:size(C,1)  
         % Subtract model i from all data points.
        diffs = bsxfun(@minus, double(X), double(C(i, :)));

        % Take the square root of the sum of squared differences.
        dists(:, i) = sqrt(sum(diffs.^2, 2));
    end
    
    for i = 1:size(des, 2)
        [c,idx] = min(dists(i,:));
        fhistogram(idx) = fhistogram(idx) + 1;
    end 

    image_hist(end+1, :) = fhistogram ./ sum(sum(fhistogram));
    image_hist_class(end+1) = image_class(char(key));
end

% assign features (test) to clusters
%Get histogram for test_set 
test_image_hist = [];
test_image_hist_class = [Classes.Faces];
test_image_hist_class(1) = [];
image_keys_test = containers.Map;

key_id = 0;
fprintf("ASSIGNING WORDS TEST \n")
for key = keys(test_image_feature) 
    des = test_image_feature(char(key));
    fhistogram = zeros(1, ksize);
    dists = [];

    X = des';
    for i = 1:size(C,1)  
         % Subtract model i from all data points.
        diffs = bsxfun(@minus, double(X), double(C(i, :)));

        % Take the square root of the sum of squared differences.
        dists(:, i) = sqrt(sum(diffs.^2, 2));
    end
    
    for i = 1:size(des, 2)
        [c,idx] = min(dists(i,:));
        fhistogram(idx) = fhistogram(idx) + 1;
    end 
    
    test_image_hist(end+1, :) = fhistogram ./ sum(sum(fhistogram));
    test_image_hist_class(end+1) = image_class(char(key));
    
    key_id = key_id + 1;
    image_keys_test(num2str(key_id)) = char(key);
end    


%% SVM
fprintf("SVM \n")
model = cell(4,1);
for k = 1:4
    best = train(double(image_hist_class == classes_set(k))', sparse(image_hist), '-C ');
    model{k} = train(double(image_hist_class == classes_set(k))', sparse(image_hist), sprintf('-c %f', best))
end 

%% predict svm
fprintf("Predict SVM \n")
prediction = [];
multipliers = [0,0,0,0];
for k = 1:4
    % We predict our train data so that we know if the class is on the
    % positive or negative side/bound.
    % see: https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f430.
    [~, ~, decision_values]= predict(double(image_hist_class == classes_set(k))', sparse(image_hist), model{k});
    multipliers(k) = 1;
    if sum(sum(decision_values > 0)) > sum(sum(decision_values < 0))
        multipliers(k) = -1;
    end
    
    [predicted_label, accuracy, decision_values]= predict(double(test_image_hist_class == classes_set(k))', sparse(test_image_hist), model{k});
    prediction(:, k) = decision_values(:) * multipliers(k);
end 

%% print accuracy and MAP
prediction_order = [Classes.Airplanes];
prediction_order(1) = [];
correct_predictions = 0;

hist2 = [0,0,0,0];
for i = 1: size(prediction,1)
    [out,idx] = sort(prediction(i, :), 'desc');
    prediction_order(i, 1:4) = classes_set(idx);
    
    hist2(idx(1)) = hist2(idx(1)) + 1;
    if prediction_order(i, 1) == test_image_hist_class(i)
       correct_predictions = correct_predictions + 1;
    end
end
accuracy = correct_predictions / size(test_image_hist_class, 2)


aps = [];
sorted_idxs = [[],[],[],[]];
for i = 1:4
    P = [];
    R = 0;
    [out,idx] = sort(prediction(:, i), 'desc');
    sorted_idxs(i, :) = idx;
    for rank_i = idx'
        i_prediction = prediction(rank_i, :);
        label = find(classes_set == test_image_hist_class(rank_i));
        if label == i
            R = R + 1;
            P(end + 1) = R / (size(P,2) + 1);
        else 
            P(end + 1) = 0 / (size(P,2) + 1);
        end
    end
    
    aps(end+1) = sum(sum(P)) / 50;
end

map = mean(aps)
bar(hist2);


%% Generate HTML
html = '<html lang="en"><head><meta charset="utf-8"><title>Image list prediction</title><style type="text/css">img {width:200px;}</style></head><body>';
html = strcat(html, '<h2>Romeo Goosens, Jorn Ranzijn</h2>');
html = strcat(html, '<h1>Settings</h1><table>');
html = strcat(html, '<tr><th>SIFT step size</th><td><td>');
html = strcat(html, step_size_sift_string);
html = strcat(html, ' pixels </td></tr>');

html = strcat(html, '<tr><th>SIFT block sizes</th><td><td>');
html = strcat(html, block_size_sift_string);
html = strcat(html, ' pixels </td></tr>');

html = strcat(html, '<tr><th>SIFT method</th><td><td>');
html = strcat(html, sift_method_string);
html = strcat(html, '</td></tr>');

html = strcat(html, '<tr><th>Vocabulary size</th><td><td>');
html = strcat(html, num2str(ksize));
html = strcat(html, ' words </td></tr>');

html = strcat(html, '<tr><th>Vocabulary fraction</th><td><td>');
html = strcat(html, strcat(num2str(vocab_fraction), '/', num2str(n_total)));
html = strcat(html, '</td></tr>');

html = strcat(html, '<tr><th>SVM training data</th><td><td>');
html = strcat(html, string(strcat({num2str(pos_sample)}, {' postive samples and '}, {num2str(neg_sample)}, {' negative samples'})));
html = strcat(html, '</td></tr>');

html = strcat(html, '<tr><th>SVM kernel type</th><td><td>');
html = strcat(html, "Linear");
html = strcat(html, '</td></tr></table><h1>Prediction lists (MAP: ');


html = strcat(html, num2str(map));
html = strcat(html, ')<table><thead><tr><th>Airplanes (AP: ');
html = strcat(html, num2str(aps(1)));
html = strcat(html, ')</th><th>Cars (AP: ');
html = strcat(html, num2str(aps(2)));
html = strcat(html, ')</th><th>Faces (AP: ');
html = strcat(html, num2str(aps(3)));
html = strcat(html, ')</th><th>Motorbikes (AP: ');
html = strcat(html, num2str(aps(4)));

html =  strcat(html, ')</th></tr></thead><tbody>');
all_images = test_image_set(:);
for i = 1:size(sorted_idxs, 2)
    html =  strcat(html,'<tr><td><img src="../.');
    html =  strcat(html, image_keys_test(num2str(sorted_idxs(1, i))));
    html =  strcat(html,'" /></td><td><img src="../.');
    html =  strcat(html, image_keys_test(num2str(sorted_idxs(2, i))));
    html =  strcat(html,'" /></td><td><img src="../.');
    html =  strcat(html,image_keys_test(num2str(sorted_idxs(3, i))));
    html =  strcat(html,'" /></td><td><img src="../.');
    html =  strcat(html,image_keys_test(num2str(sorted_idxs(4, i))));
    html =  strcat(html, '" /></td></tr>');
end

fid = fopen( 'results.html', 'wt');
fprintf(fid, html);
fclose(fid);
%%
% %overfit experiment
% for k = 1:4
%     [predicted_label, accuracy, decision_values]= predict(double(image_hist_class == classes_set(k))', sparse(image_hist), model{k});
%     predictionO(:, k) = decision_values(:);
% end 
% 
% figure()
% bar(predictionO(:,1))
% ylabel("decision value")
% xlabel("image id")




