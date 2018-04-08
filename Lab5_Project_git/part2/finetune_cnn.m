function [net, info, expdir] = finetune_cnn(varargin)

%% Define options
run(fullfile(fileparts(mfilename('fullpath')),...
    'matconvnet-1.0-beta25','matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile('data', ...
  sprintf('cnn_assignment-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = './data/' ;
opts.imdbPath = fullfile(opts.expDir, 'imdb-caltech.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

opts.train.gpus = [1];
%% update model

net = update_model();

%% TODO: Implement getCaltechIMDB function below

if exist(opts.imdbPath, 'file')
  'NOOOO'
  imdb = load(opts.imdbPath) ;
else
  imdb = getCaltechIMDB() ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


%%
net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train ;
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 2)) ;

expdir = opts.expDir;
end
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

end

function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

end

% -------------------------------------------------------------------------
function imdb = getCaltechIMDB()
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
classes = {'airplanes', 'cars', 'faces', 'motorbikes'};
splits = {'train', 'test'};

% TODO: Implement your loop here, to create the data structure described in the assignment
data_main_path = "../Caltech4/";
image_path = "ImageData/";
image_number = 1;
grayscale_images = 0;
data = [];
sets = [];
labels = [];
for i = 1:size(splits, 2)
    if strcmp(splits(i), 'train')
        n_images = 400;
        im_set = 1;
    else
        n_images = 50;
        im_set = 2;
    end
    
    for j = 1:size(classes, 2)
        im_class = string(classes(j));
        folder_path = string(data_main_path + image_path + im_class + '_' + splits(i));
        if strcmp(im_class, 'airplanes')
            im_label = 1;
        end
        if strcmp(im_class, 'cars')
            im_label = 2;
        end
        if strcmp(im_class, 'faces')
            im_label = 3;
        end
        if strcmp(im_class, 'motorbikes')
            im_label = 4;
        end
        
        for z = 1:n_images
            image_name = string(folder_path + '/img' + num2str(z,'%.3d') + '.jpg');
            image_name = char(image_name);
            im = imread(image_name);
            if size(im, 3) ~= 3
                grayscale_images = grayscale_images + 1;
                continue
            end
            im = single(imresize(im, [32, 32]));
            data(:, :, :, image_number) = im;
            sets(1, image_number) = single(im_set);
            labels(1, image_number) = single(im_label);
            image_number = image_number + 1;
        end
    end
end

grayscale_images = 0;

%
% subtract mean
dataMean = mean(data(:, :, :, sets == 1), 4);
data = bsxfun(@minus, data, dataMean);

imdb.images.data = single(data) ;
size(imdb.images.data)
imdb.images.labels = single(labels) ;
imdb.images.set = single(sets);
imdb.meta.sets = {'train', 'val'} ;
imdb.meta.classes = classes;

perm = randperm(numel(imdb.images.labels));
imdb.images.data = imdb.images.data(:,:,:, perm);
imdb.images.labels = imdb.images.labels(perm);
imdb.images.set = imdb.images.set(perm);

end