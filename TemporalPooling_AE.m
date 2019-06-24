function TemporalPooling_AE()    
    %data loading and pre-processing
    Options.KERN = 5;    % NonLinKernType(SignSqr:1 Hmk:2 SqrAbs:3 Sqr:4 PosNeg:5)
    Options.Norm = 2;     % L2 normalization
    Options.Stdsc = 0;    % Standard Scales
    featfiles = {'train21_bow_hstgrms','test21_bow_hstgrms'};   
    for f = 1 : numel(featfiles)
        %featfile = sprintf('/data/Leon/Data/AudioEventDataset/feat/%s.mat',featfiles{f});
        featfile = sprintf('/data/Leon/audiodarwin/bow/%s.mat',featfiles{f});
        load(featfile);
        %labels{f} = label;
        file = sprintf('darwin_feats/bow/%s_darwin.mat',featfiles{f});
        if exist(file,'file') ~= 2
            %[AudioDarwinFeats] = getAudioDarwin(feats, 0.00001, Options.KERN);
            [AudioDarwinFeats] = getAudioDarwin(bow_hstgrms, 0.00001, Options.KERN);
            save(file,'AudioDarwinFeats');
        else
            load(file);
        end
        ALL_Data_cell{f} = AudioDarwinFeats;  
    end    
    
	if Options.KERN ~= 0        
        for ch = 1 : size(ALL_Data_cell,2)
            %ALL_Data_cell{ch} = getNonLinearity(ALL_Data_cell{ch},3);
            ALL_Data_cell{ch} = getNonLinearity(ALL_Data_cell{ch},Options.KERN);
        end
    end  	
    
	if Options.Norm == 2       
        for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL2(ALL_Data_cell{ch});
        end
    end  
    
    if Options.Norm == 1       
         for ch = 1 : size(ALL_Data_cell,2)                 
            ALL_Data_cell{ch} = normalizeL1(ALL_Data_cell{ch});
        end
    end
    
    % Standard Scaler by feature
    if Options.Stdsc == 1
        for ch = 1 : size(ALL_Data_cell,2)
            ALL_Data_cell{ch} = zscore(ALL_Data_cell{ch});
        end
    end
    
    % Standard Scaler by sample
    if Options.Stdsc == 2
        for ch = 1 : size(ALL_Data_cell,2)
            temp = ALL_Data_cell{ch}';
            temp = zscore(temp);
            ALL_Data_cell{ch} = temp';
        end
        clear temp;
    end
    
    weights = 1;
    load('/data/Leon/Data/AudioEventDataset/feat/train21_label.mat');
    TrainClass = label;
    load('/data/Leon/Data/AudioEventDataset/feat/test21_label.mat');
    TestClass = label;

    TrainData = ALL_Data_cell{1};        
    TestData = ALL_Data_cell{2};
    TrainData_Kern_cell = [TrainData * TrainData'];    
    TestData_Kern_cell = [TestData * TrainData'];
    clear TrainData; clear TestData;
            
    TrainData_Kern = zeros(size(TrainData_Kern_cell));
    TestData_Kern = zeros(size(TestData_Kern_cell));    
    TrainData_Kern = TrainData_Kern + weights * TrainData_Kern_cell;
    TestData_Kern = TestData_Kern + weights * TestData_Kern_cell;

    clear TrainData_Kern_cell; clear TestData_Kern_cell;
    
%     % Standard Scaler by sample
%     if Options.Stdsc == 2
%         temp1 = TrainData_Kern';
%         temp1 = zscore(temp1);
%         TrainData_Kern = temp1';
%         temp2 = TestData_Kern';
%         temp2 = zscore(temp2);
%         TestData_Kern = temp2';
%         clear temp1; clear temp2;
%     end
    [precision(weights,:),recall(weights,:),acc(weights) ] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass);
            
    [accuracy,indx] = max(acc);            
    precision = precision(indx,:);
    precision(isnan(precision)) = 0;
    recall = recall(indx,:);
    recall(isnan(recall)) = 0;
    F = 2*(precision .* recall)./(precision+recall);
    F(isnan(F)) = 0;
    fprintf('Mean F score = %1.2f\n',mean(F));
    save(sprintf('results.mat'),'accuracy','precision','recall','F');
        
end

function W = genRepresentation(data,CVAL,NonLinType)
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end
    Data = getNonLinearity(Data,NonLinType);
    %disp(size(Data));
    W_fow = liblinearsvr(Data,CVAL,2); 			
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data =  zeros(size(data,1)-1,size(data,2));
    for j = 2 : size(data,1)                
        Data(j-1,:) = mean(data(1:j,:));
    end
    Data = getNonLinearity(Data,NonLinType);
    W_rev = liblinearsvr(Data,CVAL,0); 			              
    W = [W_fow ; W_rev];
    %disp(size(W));
end

function Data = getNonLinearity(Data, NonLinType)
	if NonLinType == 1
		Data = sign(Data).*sqrt(abs(Data));
	end
	if NonLinType == 2
		Data = vl_homkermap(Data',4,'kchi2');
        Data = Data';
	end
	if NonLinType == 3
		Data = sqrt(abs(Data));
	end
	if NonLinType == 4
		Data = sqrt(Data);
    end
    if NonLinType == 5
        Data = PosNegFmap(Data);
    end
end

function o = PosNegFmap(x)
    s = sign(x);
    y = sqrt(s.*x);
    o = [y.*(s == 1) y.*(s == -1)];
end

function [ALL_Data] = getAudioDarwin(feats, CVAL, NonLinType)    
    %CVAL = 0.01; % C value for the ranking function or SVR    
	TOTAL = numel(feats);  
    for i = 1:TOTAL
        data = feats{i};
        %disp(size(data));
        %disp(i);
        W = genRepresentation(data,CVAL,NonLinType);
        if i == 1
             ALL_Data =  zeros(TOTAL,size(W,1)) ;          
        end
        if mod(i,100) == 0
            fprintf('.')
        end
        ALL_Data(i,:) = W';
    end
    fprintf('Complete...\n')
end

function X = normalizeL2(X)
	for i = 1 : size(X,1)
		if norm(X(i,:)) ~= 0
			X(i,:) = X(i,:) ./ norm(X(i,:));
		end
    end	   
end

function w = liblinearsvr(Data,C,normD)
    if normD == 2
        Data = normalizeL2(Data);
    end
    
    if normD == 1
        Data = normalizeL1(Data);
    end
    
    N = size(Data,1);
    Labels = [1:N]';
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %d -s 11 -p 0.1 -q',C) );
    w = model.w';
end

function [precision, recall, acc] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass)
        % precomputed kernel c-svm
        nTrain = 1 : size(TrainData_Kern,1);
        TrainData_Kern = [nTrain' TrainData_Kern];         
        nTest = 1 : size(TestData_Kern,1);
        TestData_Kern = [nTest' TestData_Kern];% add test index??? 
        disp(size(TestData_Kern));
        %C = [20];
        C = [20.3 20.4 20.5 20.6 20.7 20.8 20.9];
        %C = [0.001 0.01 0.1 1 5 10 100 500 1000 5000 10000];

        CoreNum = 4;
        TrainPool = parpool(CoreNum);
        for ci = 1 : numel(C)
            %model(ci) = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -v 2 -q -w6 3 -w15 3 -w23 5 -e 0.001',C(ci)));
            model(ci) = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -v 2 -q -e 0.001',C(ci)));
        end
        delete(TrainPool);
        
        [~,max_index] = max(model);
        C = C(max_index);
        fprintf('The best C Value of Precomputed Kernel SVM: C: %1.6f\n', C);
        
        for ci = 1 : numel(C)
            % precomputed kernel c-svm
            model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -q -e 0.001',C(ci)));
            % radicial basis function c-svm
            %clear model;
            %model = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 2 -c %1.6f -g %1.6f -q -e 0.001',C(ci),gamma(gi)));
            [predicted, acc, scores{ci}] = svmpredict(TestClass, TestData_Kern ,model);
            [precision(ci,:) , recall(ci,:)] = perclass_precision_recall(TestClass,predicted);
            accuracy(ci) = acc(1,1);
            disp(ci);
            disp(accuracy(ci));
        end        
         
        [acc,cindx] = max(accuracy);   
        scores = scores{cindx};
        precision = precision(cindx,:);
        recall = recall(cindx,:);
end

function [precision , recall] = perclass_precision_recall(label,predicted)

    for cl = 1 : 21
        true_pos = sum((predicted == cl) .* (label == cl));
        false_pos = sum((predicted == cl) .* (label ~= cl));
        false_neg = sum((predicted ~= cl) .* (label == cl));
        precision(cl) = true_pos / (true_pos + false_pos);
        recall(cl) = true_pos / (true_pos + false_neg);
        
    end
    
end
