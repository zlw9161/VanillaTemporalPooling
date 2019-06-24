rootdir = '/data/Leon/Data/AudioEventDataset/test';  
subdir = dir(rootdir); 
audiolist = fopen('/data/Leon/Data/AudioEventDataset/list/test_audio.txt', 'wt');
labellist = fopen('/data/Leon/Data/AudioEventDataset/list/test_label.txt', 'wt');

for i=1:length(subdir)  
   if(isequal(subdir(i),'.') || isequal(subdir(i),'..') || ~subdir(i).isdir)   
       continue;  
   end  
   subdirpath = fullfile(rootdir,subdir(i).name,'*.wav');
   %fprintf('No. "%d" subdirpath: "%s" \n', i, subdirpath); 
   audios = dir(subdirpath); % get audios list  
   for j=1:length(audios)  
       AudioName = fullfile(rootdir,subdir(i).name,audios(j).name);
       fprintf(audiolist, '%s\n', AudioName);
       %fprintf(labellist, '%s\n', subdir(i).name);% for the training data
       Tmp = regexp(AudioName, 't/.*?_', 'match');
       label = Tmp{1};
       fprintf(labellist, '%s\n', label);
       %fprintf('No. "%d" audiopath: "%s" \n', j, AudioName);
   end  
end  