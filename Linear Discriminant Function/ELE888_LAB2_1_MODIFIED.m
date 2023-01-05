% USE CASE: load irisdata.mat; [a, g, k_iterations] =
% ELE888_LAB2_MODIFIED(irisdata_features)
function [a,g,k]=ELE888_LAB2_MODIFIED(Training_Data)

D=Training_Data;
[M,N]=size(D); 
setA=D(1:50,2:3);
setB=D(51:100,2:3);

% Partition sets A and B to generate training and testing sets
trainSet=[setA(1:15,1:2);setB(1:15,1:2)]
testSet=[setA(16:50,1:2);setB(16:50,1:2)];

k=0; % gradient descent algorithm iteration
% a=[-100 60 -30]'; % part 5 intiv
% a=[10 -80 -23]'; % part 5 initv
a=[0 0 1]'; % initaliziation of solution vector
theta=0;
% eta = 10; % part 5 learning rate
% eta = 0.0001; % part 5 learning rate
eta=0.01;
% augmented and normalized test data sets
aug_testSet=[ones(70,1),testSet(:,1:2)]';
norm_testSet=[aug_testSet(1:3,1:35),-1*aug_testSet(1:3,36:70)];
%augmented and normallized training data sets
aug_trainSet=[ones(30,1),trainSet(:,1:2)]';
norm_trainSet=[aug_trainSet(1:3,1:15),-1*aug_trainSet(1:3,16:30)];
Jpa = [];

% Gradient descent approach algorithm using while as "do" is not available 
% in matlab
 while 1 
     k=k+1;
     % a(k) = (a^t)y
     J=a'*norm_trainSet;
     gradJ=0;
     Jpa(k) = 0;
        for i=1:length(J)
                if (J(i)<=0)
                 % Sum only the misclassified samples using condition 
                 % g(x) < 0 for each iterration
                 gradJ=gradJ+(-norm_trainSet(1:3,i));
                 % Get the Perceptron function through the summation of all 
                 % the misclassified samples in the current iteration. 
                 Jpa(k) = Jpa(k) + (-1)*J(i);
                end   
        end
     % Adjust a based on error correction using gradient descent   
     a=a-eta*gradJ;
       % check if there is a significant change in current iteration a
       if (abs(eta*gradJ)<=theta)
         break;
       % if above condition does not happen, run for 300 iterations only
       elseif(k>=300)
         break;
       end
 end

% Classification test for the final a value and the augmented test data set
g=a'*aug_testSet
misclassA=length(find(g(1:35)<0))
misclassB=length(find(g(36:70)>0))
errorRate=(misclassA+misclassB)/length(testSet)

% Determine values for LDF g(x)
for x=0:(max(trainSet)+1)
    w=x+1;
y(w)=-a(2)/a(3)*x-a(1)/a(3);
lineData(w,1)=y(w);
lineData(w,2)=x;
end

%% Plots for part 2 and 6
% Sketch the training data set, the LDF g(x) result from the calculated 
% weight factor and the Perceptron Criterion Function Jpa.

% Used for graphs in part 6 - Plot training dataset.
figure
subplot(2,1,1);
plot(trainSet(:,1),trainSet(:,2),".");
hold on;
plot(D(1:15,2),D(1:15, 3),"bs");
hold on;
plot(D(51:65,2),D(51:65, 3),"rs");
hold on;

% Used for graphs in part 2. Uncomment below and comment above to plot the
% testing set to visually see the accuracy of the LDF. 
% plot(testSet(:,1),testSet(:,2),".");
% hold on;
% plot(D(16:50,2),D(16:50, 3),"bs");
% hold on;
% plot(D(66:100,2),D(66:100, 3),"rs");
% hold on;
plot(lineData(:,2),lineData(:,1)); % Sketch LDF g(x)
xlabel('Sepal Width (x2)');
ylabel('Petal Length (x3)');
title('Training Data for Iris Setosa vs. Iris Versicolour (30% Training)');
legend('Training set samples','Iris Setosa','Iris Versicolour', 'g(x)');
% legend('Testing set samples','Iris Setosa','Iris Versicolour', 'g(x)');

% Plot Perceptron Criterion Function J_p(a)
n = (1:k);
subplot(2,1,2);
plot(n, Jpa);
title("Perceptron Criterion Function J_p(a) for Iris Setsoa vs. Iris Versicolour (30% Training)");
xlabel("k iterations");
ylabel("J_p(a)");
