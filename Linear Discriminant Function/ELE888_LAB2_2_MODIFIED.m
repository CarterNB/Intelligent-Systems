
function [a,g,k]=ELE888_LAB2_2_MODIFIED(Training_Data)

D=Training_Data;
[M,N]=size(D); 
setA=D(1:50,2:3);
setB=D(51:100,2:3);

testSet=[setA(1:15,1:2);setB(1:15,1:2)];
trainSet=[setA(16:50,1:2);setB(16:50,1:2)];
k=0;
a=[0 0 1]';
theta=0;
eta=0.01;
%augmented and normalized training data sets
aug_trainSet=[ones(70,1),trainSet(:,1:2)]';
norm_trainSet=[aug_trainSet(1:3,1:35),-1*aug_trainSet(1:3,36:70)];
%augmented and normallized test data sets
aug_testSet=[ones(30,1),testSet(:,1:2)]';
norm_testSet=[aug_testSet(1:3,1:15),-1*aug_testSet(1:3,16:30)];

%Gradient descent approach using while as "do" is not available in matlab
 while 1 
     k=k+1;
     J=a'*norm_trainSet;
     gradJ=0;
     Jpa(k) = 0;
        for i=1:length(J)
                if (J(i)<=0)
                 gradJ=gradJ+(-norm_trainSet(1:3,i));
                 % Get the Perceptron function through the summation of all 
                 % the misclassified samples in the current iteration.                  
                 Jpa(k) = Jpa(k) + (-1)*J(i);
                end   
        end
     a=a-eta*gradJ;
       if (abs(eta*gradJ)<=theta)
         break;
       elseif(k>=300)
         break;
       end
 end
%Classification test for the final a value and the augmented test data set
g=a'*aug_testSet
misclassA=length(find(g(1:15)<0))
misclassB=length(find(g(16:30)>0))
errorRate=(misclassA+misclassB)/length(testSet)

for x=0:(max(trainSet)+1)
    w=x+1;
y(w)=-a(2)/a(3)*x-a(1)/a(3);
lineData(w,1)=y(w);
lineData(w,2)=x;
end
figure
subplot(2,1,1);
plot(trainSet(:,1),trainSet(:,2),".");
hold on;
plot(D(16:50,2),D(16:50, 3),"bs");
hold on;
plot(D(66:100,2),D(66:100, 3),"rs");
hold on;
plot(lineData(:,2),lineData(:,1));
xlabel('Sepal Width (x2)');
ylabel('Petal Length (x3)');
title('Training Data for Iris Setosa vs. Iris Versicolour');
legend('Training set samples','Iris Setosa','Iris Versicolour', 'g(x)');

n = (1:k);
subplot(2,1,2);
plot(n, Jpa);
title("Perceptron Criterion Function J_p(a) for Iris Setsoa vs. Iris Versicolour (70% Training)");
xlabel("k iterations");
ylabel("J_p(a)");
