%% FUNCTION FOR ANALYTICALLY FINDING THE THRESHOLD VALUE %%
function [threshold]=ELE888_LAB1_2(Training_Data) 
D = Training_Data;
[M,N]=size(D);  
f=D(:,1);  % feature samples -- 1 for sepal length; 2 for sepal width
la=D(:,N); % class labels
xMin = min(f); %smallest feature value
xMax = max(f); %largest feature value
i=0;


for xTest=xMin:0.01:xMax %find all g(x) for the associated x value
%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Prior probabilities:');
Pr1 = length(find(la(:)==1))/length(D)
Pr2 = length(find(la(:)==2))/length(D)
%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Mean & Std for class 1 & 2');
m11 =  mean(f(find(la(:)==1)));
std11 = std(f(find(la(:)==1)));
 
m12 =  mean(f(find(la(:)==2)));
std12= std(f(find(la(:)==2)));


disp(['Conditional probabilities for x=' num2str(xTest)]);
cp11= (1/(sqrt(2*pi)*std11)*exp(-(1/2)*((xTest-m11)/std11)^2))
% use the above mean, std and the test feature to calculate p(x/w1)

cp12= (1/(sqrt(2*pi)*std12)*exp(-(1/2)*((xTest-m12)/std12)^2))
% use the above mean, std and the test feature to calculate p(x/w2)

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

disp('Posterior prob. for the test feature');

p=(Pr1*cp11)+(Pr2*cp12);

pos11= Pr1*cp11/p; % p(w1/x) for the given test feature value

pos12= Pr2*cp12/p; % p(w2/x) for the given test feature value

posteriors_x= [pos11 pos12]

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

g_x= log(pos11/pos12)+log(Pr1/Pr2)% compute the g(x) for min err rate classifier.
i=i+1;
threshold(i,1)=[g_x];
threshold(i,2)=[xTest];
end