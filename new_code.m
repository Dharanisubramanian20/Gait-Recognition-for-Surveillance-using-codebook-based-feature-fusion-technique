clc;
close all;
clear all;

%% general settings ...
fontsize = 14;

%% Create Network and Train...
imageDir = 'C:\Users\ESSM\Desktop\GAIT RECOGNITION\datasets\PERSON 1';
fileExtension = '*.png';
matlabroot='C:\Users\ESSM\Desktop\GAIT RECOGNITION';
Datasetpath=fullfile(matlabroot,'ProcessedData');
imds =imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Display some sample images.
  
figure;
perm = randperm(45,20);
 for i = 1:20
     subplot(4,5,i);
    imshow(imds.Files{perm(i)});
 end
 
%% Split Data into Training and testing sets ...

[imdsTrain,imdsTest] = splitEachLabel(imds,0.7);

idx = randperm(45,16);

layers=[imageInputLayer([120 120 1])
    convolution2dLayer(5,20)  
   reluLayer
   maxPooling2dLayer(2,'stride',2)
   convolution2dLayer(5,20)  
reluLayer
 maxPooling2dLayer(2,'stride',2)
 
 fullyConnectedLayer(11)
 softmaxLayer
    classificationLayer()]


%% Training Option ...

options = trainingOptions('sgdm','MaxEpochs',40,...
	'InitialLearnRate',0.0001);

%% Train CNN ..

convnet=trainNetwork(imds,layers,options);
save convnet


classNames = convnet.Layers(end).ClassNames;
numClasses = numel(classNames);
disp(classNames(randperm(numClasses,1)))

%% Test Convlution Neural Network...
%IMAGE ACQUISITION ...
% Pick the image and Load the image
 [filename, pathname] = uigetfile( ...
       {'*.jpg;*.tif;*.tiff;*.png;*.bmp', 'All image Files (*.jpg, *.tif, *.tiff, *.png, *.bmp)'}, ...
        'Pick a file'); 
    disp('Reading image');
IM=imread([pathname,filename]);
% a=imread('test1.jpg');
figure();imshow(IM);
title('Input Image ');
impixelinfo;

size(IM)

%hog  mapping 

% Convert the image to grayscale if it's not already
if size(image, 3) > 1
    grayImage = rgb2gray(image);
else
    grayImage = image;
end

% Calculate HOG features
[hogFeatures, visualization] = extractHOGFeatures(IM, 'CellSize', [8 8]);

% Display the HOG visualization
figure;
imshow(IM);
hold on;
plot(visualization);
title('HOG Visualization');
hold off;
%motio vector 

% imageDir = '';
% fileExtension = '*.png';
imageFiles = dir(fullfile(imageDir, fileExtension));
imageSequence = cell(1, numel(imageFiles));
for i = 1:numel(imageFiles)
    % Construct the full file path
    filePath = fullfile(imageDir, imageFiles(i).name);
    
    % Read the image
    imageSequence{i} = imread(filePath);

% Parameters

numBins = 64; % Number of bins for the histogram
threshold = 0; % Threshold for motion detection

% Optical flow object
opticFlow = opticalFlowLK('NoiseThreshold',0.01);

% Histogram initialization
histogramData = zeros(1, numBins);

% Read the first frame
framePrev = imread(filePath);
%framePrevGray = rgb2gray(framePrev);

% Iterate through the frames
      
    % Read the current frame
    frameCurr = imread(filePath);
    %frameCurrGray = rgb2gray(frameCurr);
    
    % Compute optical flow
    flow = estimateFlow(opticFlow, framePrev);
    
    % Compute motion magnitude
    magnitude = sqrt(flow.Vx.^2 + flow.Vy.^2);
    
    % Compute motion boundaries
    motionBoundaries = magnitude > threshold;
    
    % Compute histogram
    histogramData = histogramData + histcounts(magnitude(motionBoundaries), numBins);
    
    % Update previous frame
    framePrev = frameCurr;
end

% Norma`lize histogram
histogramData = histogramData / sum(histogramData);

% Plot histogram
figure();
bar(linspace(0, 1, numBins), histogramData);
title('Motion Boundary Histogram');
xlabel('Magnitude');
ylabel('Frequency');

% Optional: Save histogram data
save('motion_boundary_histogram.mat', 'histogramData');









%% 1st  convolutional layer output..

% Get the network weights for the second convolutional layer
w1 = convnet.Layers(2).Weights;

% Scale and resize the weights for visualization
w1 = mat2gray(w1);
w1 = imresize(w1,5);

% Display a montage of network weights. There are 96 individual sets of
% weights in the first layer.
figure
montage(w1)
title('First convolutional layer weights')

%% Clasification ...

% [label,score] = classify(convnet,IM);
[label,score] =classify(convnet,IM);

label

figure,imshow(IM)
title(string(label) + ", " + num2str(100*score(classNames == label),3) + "%");

result = label;

%% display classify results

disp('The  detected PERSON in the test file is :- ');

if result== 'PERSON 1'
     msg = ('PERSON 1');          
    msgbox(msg,'The Result is..');
    disp('PERSON 1');
    
elseif result=='PERSON 2'
    msg = ('PERSON 2 ');          
    msgbox(msg,'The Result is..');
    disp('PERSON 2');
 elseif result=='PERSON 3'
     msg = ('PERSON 3');          
    msgbox(msg,'The Result is..');
    disp('PERSON 3');    
elseif result=='PERSON 4'
      msg = ('PERSON 4 ');          
    msgbox(msg,'The Result is..');
    disp('PERSON 4');   
elseif result=='PERSON 5'
   msg = ('PERSON 5 ');          
    msgbox(msg,'The Result is..');
    disp('PERSON 5'); 
elseif result=='PERSON 6'
    msg = ('PERSON 6');          
    msgbox(msg,'The Result is..');
    disp('PERSON 6'); 
elseif result=='PERSON 7'
    msg = ('PERSON 7');          
    msgbox(msg,'The Result is..');
    disp('PERSON 7'); 
elseif result=='PERSON 8'
    msg = ('PERSON 8 ');          
    msgbox(msg,'The Result is..');
    disp('PERSON 8'); 
elseif result=='unknown'
     msg = ('unknown ');          
    msgbox(msg,'The Result is..');
    disp('unknown'); 
end
%% Top 3 Predictions Scores..

[~,idx] = sort(score,'descend');
idx = idx(5:-1:1);
classNamesTop = convnet.Layers(end).ClassNames(idx);
scoresTop = score(idx);

figure
barh(scoresTop)
xlim([0 1])
title('Top 3 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)

%% ToTal Accuracy of Classifier system

% accuracy = sum(labels== YTest)/numel(YTest)
% Accuracy_Percent = accuracy.*100

YPred = classify(convnet,imdsTest);
YValidation = imdsTest.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
Accuracy_Percent = accuracy.*100;

disp('The ToTal Accuracy of Classifier system is :- ');

Accuracy_Percent 



