%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ELE 888/ EE 8209: LAB 1: Bayesian Decision Theory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [posteriors_x,g_x]=lab1(x,Training_Data)

% x = individual sample to be tested (to identify its probable class label)
% featureOfInterest = index of relevant feature (column) in Training_Data 
% Train_Data = Matrix containing the training samples and numeric class labels
% posterior_x  = Posterior probabilities
% g_x = value of the discriminant function

D=Training_Data;

% D is MxN (M samples, N columns = N-1 features + 1 label)
[M,N]=size(D);    
 
f=D(:,1); % feature samples -- 1 for sepal length; 2 for sepal width
la=D(:,N); % class labels


%% %%%%Prior Probabilities%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hint: use the commands "find" and "length"

disp('Prior probabilities:');
Pr1 = length(find(la(:)==1))/length(D)
Pr2 = length(find(la(:)==2))/length(D)

%% %%%%%Class-conditional probabilities%%%%%%%%%%%%%%%%%%%%%%%

disp('Mean & Std for class 1 & 2');
m11 =  mean(f(find(la(:)==1)));
std11 = std(f(find(la(:)==1)));
 
m12 =  mean(f(find(la(:)==2)));
std12= std(f(find(la(:)==2)));


disp(['Conditional probabilities for x=' num2str(x)]);
% use the above mean, std and the test feature to calculate p(x/w1)
cp11= (1/(sqrt(2*pi)*std11)*exp(-(1/2)*((x-m11)/std11)^2))
% use the above mean, std and the test feature to calculate p(x/w2)
cp12= (1/(sqrt(2*pi)*std12)*exp(-(1/2)*((x-m12)/std12)^2))

%% %%%%%%Compute the posterior probabilities%%%%%%%%%%%%%%%%%%%%

disp('Posterior prob. for the test feature');

p=(Pr1*cp11)+(Pr2*cp12);

pos11= Pr1*cp11/p; % p(w1/x) for the given test feature value

pos12= Pr2*cp12/p; % p(w2/x) for the given test feature value

posteriors_x= [pos11 pos12]

%% %%%%%%Discriminant function for min error rate classifier%%%

disp('Discriminant function for the test feature');

g_x= log(pos11/pos12)+log(Pr1/Pr2)% compute the g(x) for min err rate classifier.

