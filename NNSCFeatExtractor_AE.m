function NNSCFeatExtractor_AE
addpath('/usr/local/MATLAB/R2016b/toolbox/vlfeat-0.9.20/toolbox');
vl_setup();
listpath = '/data/Leon/Data/AudioEventDataset/list/';
featpath = '/data/Leon/Data/AudioEventDataset/feat/';
datalisttype = {'train', 'test'};
dict_size = 4000;
clip_size = 10;
feat_dimension = 126;
data = zeros(feat_dimension, 1);
dict_file = 'nnsc/dict/AE_16khz_mfcc40_win400-160_dict_4000.mat';

for f = 1 : numel(datalisttype)
    datalistdir = sprintf('%s%s_audio.txt', listpath, datalisttype{f});
    labellistdir = sprintf('%s%s_label.txt', listpath, datalisttype{f});
    datalist = importdata(datalistdir);
    labellist = importdata(labellistdir);
    datanum = size(datalist,1);
    label = zeros(datanum,1);
    for i = 1 : datanum
        %disp(i);
        wavdir = datalist{i};
        feats{i} = MFCC_E0dD(wavdir);%[numFrm, Dimension]
        % only training data are used to construct dict
        if datalisttype{f}(1:4) == 'trai'
            data = [data feats{i}'];
            %disp(size(data));% [Feat_Dim, NumFrm]
        end
        %disp(size(feats{i}));
        scene = labellist{i};
        switch scene 
            case 'acoustic_guitar'
                label(i) = 1;
            case 'airplane'
                label(i) = 2;
            case 'applause'
                label(i) = 3;
            case 'bird'
                label(i) = 4;
            case 'car'
                label(i) = 5;
            case 'cat'
                label(i) = 6;
            case 'child'
                label(i) = 7;
            case 'church_bell'
                label(i) = 8;
            case 'crowd'
                label(i) = 9;
            case 'dog_barking'
                label(i) = 10;
            case 'engine'
                label(i) = 11;
            case 'fireworks'
                label(i) = 12;
            case 'footstep'
                label(i) = 13;
            case 'glass_breaking'
                label(i) = 14;
            case 'hammer'
                label(i) = 15;
            case 'helicopter'
                label(i) = 16;
            case 'knock'
                label(i) = 17;
            case 'laughter'
                label(i) = 18;
            case 'mouse_click'
                label(i) = 19;
            case 'ocean_surf'
                label(i) = 20;
            case 'rustle'
                label(i) = 21;
            case 'scream'
                label(i) = 22;
            case 'speech_fs'
                label(i) = 23;
            case 'squeak'
                label(i) = 24;
            case 'tone'
                label(i) = 25;
            case 'violin'
                label(i) = 26;
            case 'water_tap'
                label(i) = 27;
            case 'whistle'
                label(i) = 28;
        end     
    end
    fprintf('Generate %sing descriptors complete...\n', datalisttype{f});
    savelabeldir = sprintf('%s%s_label.mat', featpath, datalisttype{f});
    save(savelabeldir, 'label');
    
    if datalisttype{f}(1:4) == 'trai'
        data(:,1) = [];
        disp(size(data));
        if exist(dict_file,'file') ~= 2
            fprintf('No such dictionary! Start creating:\n');
            fprintf('Dictionary will save @ %s\n', dict_file);
            dictionary = GenNNSCDict(data, dict_size);
            save(dict_file, 'dictionary');
        else
            load(dict_file);
        end
    else
        load(dict_file);
        disp(size(dictionary));
        fprintf('Generate Non-Negative Sparse Coding dictionary complete...\n');
    end

    clip_num = zeros(datanum);
    total_clips = 0;
    for i = 1 : datanum
        clip_num(i) = floor(size(feats{i}, 1) / clip_size);
        total_clips = total_clips + clip_num(i); 
    end
    nnsc_label = zeros(total_clips, 1);
    k = 1;
    for i = 1 : datanum
		coefs = GetNNSCVectors(feats{i}', dictionary, dict_size);
		for l = 1 : size(coefs, 2)
			coefs(:,l) = coefs(:,l) / sum(coefs(:,l));
		end
        hstgrms = zeros(clip_num(i), dict_size);
        for j = 1 : clip_num(i)
			hstgrm = sum(coefs(:,((j-1)*clip_size+1):(j*clip_size)), 2);
            %disp(size(hstgrm'));
            hstgrms(j, :) = hstgrm';
            nnsc_label(k) = label(i);
            k = k + 1;
        end
        nnsc_hstgrms{i} = hstgrms;
    end
    disp(k);
    disp(total_clips);
    fprintf('Get %sing histograms for Non-Negative Sparse Coding complete...\n', datalisttype{f});
    nnsclabeldir = sprintf('nnsc/%s_nnsc_label.mat', datalisttype{f});
    save(nnsclabeldir, 'nnsc_label');
    nnschstdir = sprintf('nnsc/hstgrms/%s_16khz_mfcc40_nnsc.mat', datalisttype{f});
    save(nnschstdir, 'nnsc_hstgrms');
    clear nnsc_hstgrms; clear data; clear label;
end

end

function feat = MFCC_E0dD(filename)
if exist(filename,'file') ~= 2
    fprintf('file: "%s" is not exist!!!\n', filename);
end
[x,Fs] = audioread(filename);
%feat = melcepst(x, 22050, 'ME0dD', 40, 128, 2048, 512, 0, 0.5);
feat = melcepst(x, Fs, 'ME0dD', 40, 128, 400, 160, 0, 0.5);
end

function dict = GenNNSCDict(data, dict_size)
param.K = dict_size;
param.lambda = 1;% L1 Norm Regularizer for coef
param.lambda2 = 1;% L2 Norm Regularizer for coef
param.iter = 1000;
param.mode = 2;
param.posAlpha = 1;
param.posD = 1;
param.whiten = 0;
param.numThreads = -1;
% mexTrainDL: data(feat_dim * n_samples)
dict = mexTrainDL(data, param);
end

function coefs = GetNNSCVectors(feat, dict, dict_size)
param.K = dict_size;
param.lambda = 1;% L1 Norm Regularizer for coef
param.lambda2 = 1;% L2 Norm Regularizer for coef
param.iter = 1000;
param.mode = 2;
param.posAlpha = 1;
param.posD = 1;
param.whiten = 0;
param.numThreads = -1;
param.pos = 1;
coefs = mexLasso(clip, dict, param); %coef [dict_size, frm_num]
% for i = 1 : size(coef, 2)
% 	coef(:,i) = coef(:,i) / sum(coef(:,i));
% end
%hstgrm = sum(coef, 2);
end
