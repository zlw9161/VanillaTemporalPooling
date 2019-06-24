function SegFeatExtractor_AE

featdir = '/data/Leon/Data/AudioEventDataset/feat/train_seg_mfcc.mat';
datalistdir = '/data/Leon/Data/AudioEventDataset/list/train_audio.txt';
labellistdir = '/data/Leon/Data/AudioEventDataset/list/train_label.txt';
datalist = importdata(datalistdir);
labellist = importdata(labellistdir);
datanum = size(datalist,1);
label = zeros(datanum,1);
disp(datanum);
seglen = 16000;
shift = 8000;
isovlp = 0;
[segnum, totalsegs] = cal_segnum(datalist,seglen,shift,isovlp);
k = 0;
for i = 1 : datanum
    [x,Fs] = audioread(datalist{i});
    for j = 1 : segnum(i)
        if isovlp == 0
            head = 1 + (j-1) * seglen; 
            tail = j * seglen;
%             feats{k+j} = melcepst(x(head:tail), 22050, 'ME0dD', 40, 128, 2048, 512, 0, 0.5);
            feats{k+j} = melcepst(x(head:tail), 22050, 'M', 40, 128, 2048, 512, 0, 0.5);
        else
            head = 1 + (j-1) * shift;
            tail = (j-1) * shift + seglen;
%             feats{k+j} = melcepst(x(head:tail), 22050, 'ME0dD', 40, 128, 2048, 512, 0, 0.5);
            feats{k+j} = melcepst(x(head:tail), 22050, 'M', 40, 128, 2048, 512, 0, 0.5);
        end
        scene = labellist{i};
        switch scene 
            case 'acoustic_guitar'
                label(k+j) = 1;
            case 'airplane'
                label(k+j) = 2;
            case 'applause'
                label(k+j) = 3;
            case 'bird'
                label(k+j) = 4;
            case 'car'
                label(k+j) = 5;
            case 'cat'
                label(k+j) = 6;
            case 'child'
                label(k+j) = 7;
            case 'church_bell'
                label(k+j) = 8;
            case 'crowd'
                label(k+j) = 9;
            case 'dog_barking'
                label(k+j) = 10;
            case 'engine'
                label(k+j) = 11;
            case 'fireworks'
                label(k+j) = 12;
            case 'footstep'
                label(k+j) = 13;
            case 'glass_breaking'
                label(k+j) = 14;
            case 'hammer'
                label(k+j) = 15;
            case 'helicopter'
                label(k+j) = 16;
            case 'knock'
                label(k+j) = 17;
            case 'laughter'
                label(k+j) = 18;
            case 'mouse_click'
                label(k+j) = 19;
            case 'ocean_surf'
                label(k+j) = 20;
            case 'rustle'
                label(k+j) = 21;
            case 'scream'
                label(k+j) = 22;
            case 'speech_fs'
                label(k+j) = 23;
            case 'squeak'
                label(k+j) = 24;
            case 'tone'
                label(k+j) = 25;
            case 'violin'
                label(k+j) = 26;
            case 'water_tap'
                label(k+j) = 27;
            case 'whistle'
                label(k+j) = 28;
        end
    end
    k = k + segnum(i);
    disp(i);
end
disp(totalsegs);
disp(k);
%save('/data/Leon/Data/AudioEventDataset/feat/test_seg_label.mat','label');
save(featdir,'feats');
fprintf('Complete...\n');
end

function [segnum, totalsegs] = cal_segnum(datalist,seglen,shift,isovlp)
datanum = size(datalist, 1);
segnum = zeros(datanum, 1);
for i = 1 : datanum
    [x, ~] = audioread(datalist{i});
    audiolen = size(x,1);
    if isovlp == 0
        segnum(i) = floor(audiolen / seglen);
    else
        segnum(i) = floor(audiolen / shift) - 1;
    end
end        
totalsegs = sum(segnum);
end