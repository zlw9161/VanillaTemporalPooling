function FeatureExtractor_AE

featdir = '/data/Leon/Data/AudioEventDataset/feat/test_mfcc_22khz_e0dd.mat';
datalistdir = '/data/Leon/Data/AudioEventDataset/list/test_audio.txt';
labellistdir = '/data/Leon/Data/AudioEventDataset/list/test_label.txt';
datalist = importdata(datalistdir);
labellist = importdata(labellistdir);
datanum = size(datalist,1);
label = zeros(datanum,1);
disp(datanum);
for i = 1 : datanum
        disp(i);
        wavdir = datalist{i};
        feats{i} = MFCC_E0dD(wavdir);
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
%save('/data/Leon/Data/AudioEventDataset/feat/train_label.mat','label');
fprintf('Complete...\n');
save(featdir,'feats','label');
end

function feat = STFT(filename)
[x,Fs] = audioread(filename);
wlen = 1024;% window len
inc = wlen * 0.5;% frame increment len
win = hamming(wlen);% Hamming Windowing

% framing the wav data
y = enframe(x, win, inc)';
% apply stft on each frame
Y = fft(y);
% absolute value of each bin
feat = abs(Y)';
end

function feat = LogMel(filename)

if exist(filename,'file') ~= 2
    fprintf('file: "%s" is not exist!!!\n', filename);
end
[x,Fs] = audioread(filename);
wlen = 256;% window length: Fs * 40ms(frame size)
inc = wlen * 0.5;% overlapping size
win = hamming(wlen);% Hamming Windowing
% pre-emphasis
xx = double(x);
xx = filter([1 -0.9375], 1, xx);
% framing the data
xx = enframe(xx, win, inc);

% Compute Mel-Filterbank Frequency Response
Bank = melbankm(24, wlen, Fs, 0, 0.5, 'm');

% MelBank coefficients normalization
Bank = full(Bank);
Bank = Bank/max(Bank(:));

% Compute Log Mel-Filterbank Energy
n = fix(wlen/2)+1;
for i = 1:size(xx,1)
    y = xx(i,:);
    s = y'.*hamming(wlen);
    t = abs(fft(s));
    t = t.^2;
    c = log(Bank*t(1:n)+eps*(10^10));
    feat(i,:) = c';
end

end

function feat = MFCC24(filename)

if exist(filename,'file') ~= 2
    fprintf('file: "%s" is not exist!!!\n', filename);
end
[x,Fs] = audioread(filename);
wlen = 256;% window length: Fs * 40ms(frame size)
inc = wlen * 0.5;% overlapping size
win = hamming(wlen);% Hamming Windowing

% Compute Mel-Filterbank Frequency Response
Bank = melbankm(24, wlen, Fs, 0, 0.5, 'm');
% MelBank coefficients normalization
Bank = full(Bank);
Bank = Bank/max(Bank(:));

% DCT coefficients
for k = 1:12
    n = 0:24-1;
    dctcoef(k,:) = cos((2*n+1)*k*pi/(2*24));
end

% Normalization cepstrum
w = 1+6*sin(pi*[1:12]./12);
w = w/max(w);

% pre-emphasis
xx = double(x);
xx = filter([1 -0.9375], 1, xx);
% framing the data
xx = enframe(xx, win, inc);

% Compute Log Mel-Filterbank Energy
n = fix(wlen/2)+1;
for i = 1:size(xx,1)
    y = xx(i,:);
    s = y'.*hamming(wlen);
    t = abs(fft(s));
    t = t.^2;
    c1 = dctcoef*log(Bank*t(1:n));
    c2 = c1.*w';
    m(i,:) = c2';
end

dtm = zeros(size(m));
for i = 3:size(m,1)-2
    dtm(i,:) = -2*m(i-2,:)-m(i-1,:)+m(i+1,:)+2*m(i+2,:);
end
dtm = dtm/3;
feat = [m dtm];
feat = feat(3:size(m,1)-2,:);

end

function feat = MFCC_E0dD(filename)
if exist(filename,'file') ~= 2
    fprintf('file: "%s" is not exist!!!\n', filename);
end
[x,Fs] = audioread(filename);
%feat = melcepst(x, Fs, 'ME0dD', 40, 128, 2048, 512, 0, 0.5);
feat = melcepst(x, 22050, 'ME0dD', 40, 128, 2048, 512, 0, 0.5);
end