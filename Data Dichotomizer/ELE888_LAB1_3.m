D = trainingSet;
[M,N]=size(D);
f=D(:,2);  % feature samples -- 1 for sepal length; 2 for sepal width
la=D(:,N); % class labels -- different flower types
x1 = sort(f(1:50)); %Data for Setosa
x2 = sort(f(51:100)); %Data for Versicolour
cp11 = (1/(sqrt(2*pi)*std(x1)))*exp(-(1/2)*((x1 - mean(x1))/std(x1)).^2) %Conditional Probability for Setosa 
cp12 = (1/(sqrt(2*pi)*std(x2)))*exp(-(1/2)*((x2 - mean(x2))/std(x2)).^2) %Conditional Probability for Versicolour
plot(x1, cp11,x2,cp12); %Plot of the probability density vs the feature values
