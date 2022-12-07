%% Problem 4

% load image data
image1=load('image1.txt');
image2=load('image2.txt');
image3=load('image3.txt');
meanimage=load('mean_image.txt');

% load PCA data
PCA_Components=load('PCA_Components.txt');
PCA_Scores=load('PCA_Scores.txt');

% load regression and SI data
RegressionParameters=load('RegressionParameters.txt');
RegressionIntercept=load('RegressionIntercept.txt');
SmileIndx=load('SmileIndx.txt');

%% 1: Model's predicted Smile Idxs (How you calculate this)
% How do they compare to true values?

% predict Gender Strength given an image (pixels) and a model (Regression
% parameters)
predGS=zeros(1,3);
for i=1:3
    predGS(i)=PCA_Scores(i,:)*RegressionParameters+RegressionIntercept;
end

%% Model based PCA scores. Visualize reconstruced faces and actual images
% Reconstruced faces: S*A'
I_r=PCA_Scores*PCA_Components';
% We obtain 3 rows (3 images) with 93600 pixels in each
% We need to depatch the image
I1=reshape(I_r(1,:),[360,260]);
I2=reshape(I_r(2,:),[360,260]);
I3=reshape(I_r(3,:),[360,260]);

% Concept of reshape:
% for i=1:93600/360
%     I1(:,i)=I_r(1,360*(i-1)+1:360*i);
% end

% visualize the results
figure(1)
subplot(2,3,1)
title('Subject 1')
imshow(mat2gray(image1))
subplot(2,3,2)
title('Subject 2')
imshow(mat2gray(image2))
subplot(2,3,3)
title('Subject 3')
imshow(mat2gray(image3))
subplot(2,3,4)
imshow(mat2gray(I1))
subplot(2,3,5)
imshow(mat2gray(I2))
subplot(2,3,6)
imshow(mat2gray(I3))

%% Take a the 3 faces and change them so they get a smile index of 0.2

% adapt the exisiting faces so they have a SI of 0.2 without changing other
% attributes

b=zeros(1,3);

A=PCA_Scores*RegressionParameters;
for i=1:3
    b(i)=(0.2-RegressionIntercept)/A(i)
end

% we have found how Scores for each image need to change in order to obtain
% a GSI of 0.2
% we compute the product now
I1_synt=reshape(PCA_Scores(1,:)*b(1)*PCA_Components',[360,260]);
I2_synt=reshape(PCA_Scores(2,:)*b(2)*PCA_Components',[360,260]);
I3_synt=reshape(PCA_Scores(3,:)*b(3)*PCA_Components',[360,260]);

% visualize the results
figure(1)
subplot(3,3,1)
title('Subject 1')
imshow(mat2gray(image1))
subplot(3,3,2)
title('Subject 2')
imshow(mat2gray(image2))
subplot(3,3,3)
title('Subject 3')
imshow(mat2gray(image3))
subplot(3,3,4)
imshow(mat2gray(I1))
subplot(3,3,5)
imshow(mat2gray(I2))
subplot(3,3,6)
imshow(mat2gray(I3))
subplot(3,3,7)
imshow(mat2gray(I1_synt))
subplot(3,3,8)
imshow(mat2gray(I2_synt))
subplot(3,3,9)
imshow(mat2gray(I3_synt))


