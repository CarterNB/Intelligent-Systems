clear
load wine.data

%organize and initialize data from wine.data
w_data(:,1)=wine(:,2);
w_data(:,2)=wine(:,3);
%classifier targets
w_data(:,3)=wine(:,1);

for i=1:length(w_data)
    if w_data(i,3)==1
        w_data(i,3)=1;
    else
        w_data(i,3)=-1;
    end
end

%Normalize and Scale all data such that m=0 and std=1
x1=[w_data(1:59,1); w_data(131:178,1)];
x1m=mean(x1);
x1d=std(x1);
x1=(x1-x1m)/x1d;
x2=[w_data(1:59,2); w_data(131:178,2)];
x2m=mean(x2);
x2d=std(x2);
x2=(x2-x2m)/x2d;
targ=[w_data(1:59,3); w_data(131:178,3)];

n=length(x1);
eta=0.01;
theta=0.001;
d=2;
nh=2;
c=1;
r=0;
gradJ = [];

%initializing random values for the weight vector Wij
for i=1:(d*nh)
    tmp1=-1/(sqrt(d));
    tmp2=1/(sqrt(d));
    wij(i)=(tmp2-tmp1).*rand(1,1)+tmp1;
end

%initializing random values for the weight vector Wkj
for i=1:(c*nh)
    tmp1=-1/(sqrt(nh));
    tmp2=1/(sqrt(nh));
    wkj(i)=(tmp2-tmp1).*rand(1,1)+tmp1;
end

while 1
    r=r+1;
    m=0;
    del_wij=zeros([4 1]);
    del_wkj=zeros([2 1]);
    while 1
        m=m+1;
        %finding output given current weight vector for each data point
        x=[x1(m); x2(m)];
        net1=wij(1)*x(1)+wij(2)*x(2);
        net2=wij(3)*x(1)+wij(4)*x(2);
        y1=tanh(net1);
        y2=tanh(net2);
        netz=wkj(1)*y1+wkj(2)*y2;
        net=[net1;net2;netz];
        Z(m)=tanh(net(3));
        
        %Calculate sensitivity at output
        delk=(targ(m)-Z(m))*(1-(tanh(net(3)))^2);

        %Calculate sensitivity at hidden layer
        delj(1)=(1-(tanh(net(1)))^2)*wkj(1)*delk;
        delj(2)=(1-(tanh(net(2)))^2)*wkj(2)*delk;

        %find the change of the weight for each pattern
        del_wkj(1)=del_wkj(1)+eta*y1*delk;
        del_wkj(2)=del_wkj(2)+eta*y2*delk;
        
     
        del_wij(1)=del_wij(1)+eta*x(1)*delj(1);
        del_wij(2)=del_wij(2)+eta*x(2)*delj(1);
        del_wij(3)=del_wij(3)+eta*x(1)*delj(2);
        del_wij(4)=del_wij(4)+eta*x(2)*delj(2);
       
        if(m==n)
            break;
        end
    end
    
    %update weights after all patterns have been processed 
    wij(1)=wij(1)+del_wij(1);
    wij(2)=wij(2)+del_wij(2);
    wij(3)=wij(3)+del_wij(3);
    wij(4)=wij(4)+del_wij(4);
    
    wkj(1)=wkj(1)+del_wkj(1);
    wkj(2)=wkj(2)+del_wkj(2);
    
    %finding gradJ
    for i=1:n
        x=[x1(i); x2(i)];
        net1=wij(1)*x(1)+wij(2)*x(2);
        net2=wij(3)*x(1)+wij(4)*x(2);
        y1=tanh(net1);
        y2=tanh(net2);
        netz=wkj(1)*y1+wkj(2)*y2;
        net=[net1;net2;netz];
        Z(i)=tanh(net(3));
        Jp(i)=(targ(i)-Z(i))^2;
    end
    Jw(r)=(0.5)*sum(Jp);
    if r==1
        gradJ(r) = Jw(r);
    else
        gradJ(r) = Jw(r-1)-Jw(r);
    end
    %if stop condition is met exit
    if (abs(gradJ(r))<=theta)
        break;
    end
    if (r>500)
        break;
    end
        
end

% determining boundary equations based on weights
% y intercept at zero because no bias - data clustered around zero
x_axis = -5:0.5:5;
s1 = -wij(1)/wij(2); 
b1 = s1*x_axis;
s2 = -wij(3)/wij(4);
b2 = s2 * x_axis;


% plotting decision boundaries 
figure(1);
plot(x1(1:59),x2(1:59),'m.'); 
hold on;
plot(x1(60:107),x2(60:107),'k.');
plot(x_axis, b1);
hold on;
plot(x_axis, b2, 'm');
xlabel('x_1');
ylabel('x_2');
legend('class 1','class 2','Bound 1', 'Bound 2');


figure(2);
%DISPLAY THE LEARNING CURVE AND PRINT EPOCH COUNT
k = 1:length(abs(gradJ));
plot(k, abs(gradJ));
title("Learning curve for XOR operation");
xlabel("Number of Epochs for Convergence");
ylabel("J(W)");
axis tight;
r=r

%TESTING ALL DATA USING FINAL WEIGHTS
for i=1:length(x1)
        x=[x1(i); x2(i)];
        net1=wij(1)*x(1)+wij(2)*x(2);
        net2=wij(3)*x(1)+wij(4)*x(2);
        y1=tanh(net1);
        y2=tanh(net2);
        netz=wkj(1)*y1+wkj(2)*y2;
        net=[net1;net2;netz];
        testout(i)=tanh(net(3));
end

final = [];

for i=1:length(testout)
    if testout(i) > 0
        final(i) = ceil(testout(i));
    else
        final(i) = floor(testout(i));
    end
end

%SUCCESS RATE CHECKING
for i=1:length(final)
    if targ(i)==final(i)
        check(i)=1;
    else
        check(i)=0;
    end
end
sRate=length(find(check==1))/length(check)
