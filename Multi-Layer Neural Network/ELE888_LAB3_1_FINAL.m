clear

% trainingSet=[-1 -1 -1;
%              -1  1  1;
%               1 -1  1;
%               1  1 -1];


% training set samples order of (2, 3, 1, 4) for plotting convenience
trainingSet=[-1  1  1;
              1 -1  1;
             -1 -1 -1;
              1  1 -1];          
          
%initializing data and constants
x1=trainingSet(:,1);
x2=trainingSet(:,2);
targ=trainingSet(:,3);
n=length(x1);
eta=0.1;
theta=0.001;
d=2;
nh=2;
c=1;
r=0;
gradJ = [];

%initializing random values for the weight vector Wij
for i=1:(nh*(d+1))
    tmp1=-1/(sqrt(d));
    tmp2=1/(sqrt(d));
    wij(i)=(tmp2-tmp1).*rand(1,1)+tmp1;
end

%initializing random values for the weight vector Wkj
for i=1:(nh+1)
    tmp1=-1/(sqrt(nh));
    tmp2=1/(sqrt(nh));
    wkj(i)=(tmp2-tmp1).*rand(1,1)+tmp1;
end

% Network learning protocol - Batch backpropagation algorithm
while 1
    % increment epoch (r)
    r=r+1;
    % initialize amount of patterns and weight changes from 
    % patterns to zero.
    m=0;
    del_wij=zeros([6 1]);
    del_wkj=zeros([3 1]);
    while 1
        m=m+1;
        %finding output given current weight vector for each data point
        x=[x1(m); x2(m)];
        % output to hidden layer
        net1=1*wij(1)+wij(2)*x(1)+wij(3)*x(2);
        net2=1*wij(4)+wij(5)*x(1)+wij(6)*x(2);
        y1=tanh(net1);
        y2=tanh(net2);
        % output layer calculation
        netz=1*wkj(1) + wkj(2)*y1+wkj(3)*y2;
        net=[net1;net2;netz];
        Z(m)=tanh(net(3));
        
        %Calculate sensitivity at output
        delk=(targ(m)-Z(m))*(1-(tanh(net(3)))^2);
        %Calculate sensitivity at hidden layer
        delj = zeros(1,2);
        delj(1)=(1-(tanh(net(1)))^2)*wkj(2)*delk;
        delj(2)=(1-(tanh(net(2)))^2)*wkj(3)*delk;

        %find the change of the weight for each pattern
        del_wkj(1)=del_wkj(1)+eta*1*delk;
        del_wkj(2)=del_wkj(2)+eta*y1*delk;
        del_wkj(3)=del_wkj(3)+eta*y2*delk;
     
        del_wij(1)=del_wij(1)+eta*1*delj(1);
        del_wij(2)=del_wij(2)+eta*x(1)*delj(1);
        del_wij(3)=del_wij(3)+eta*x(2)*delj(1);
        del_wij(4)=del_wij(4)+eta*1*delj(2);
        del_wij(5)=del_wij(5)+eta*x(1)*delj(2);
        del_wij(6)=del_wij(6)+eta*x(2)*delj(2);
        
        if(m==n)
            break;
        end
    end
    
    % update weights after all patterns have been processed (after batch)
    wij = wij + del_wij';
    wkj = wkj + del_wkj';
    
    Jp(r) = 0.5 * (((targ - Z')') * (targ - Z'));
    if r==1
        gradJ(r) = Jp(r);
    else
        gradJ(r) = Jp(r-1)-Jp(r);
    end

    if (abs(gradJ(r))<=theta)
        break;
    end
end

% determining boundary equations based on weights
y_in1 = wij(1)/wij(3); 
s1 = wij(1)/wij(3);
x_axis = -4:0.01:4;
boundary1 = s1 * x_axis + y_in1;

y_in2 = wij(4)/wij(6);
s2 = wij(5)/wij(6);
boundary2 = y_in2 + s2*x_axis;

% plotting decision boundaries
figure(1);
plot(x1(1:2),x2(1:2),'rs');
hold on;
plot(x1(3:4),x2(3:4),'bs');
plot(x_axis, boundary1);
hold on;
plot(x_axis, boundary2);
xlabel('x_1');
ylabel('x_2');
legend('class 1','class 2','Bound 1', 'Bound 2');

figure(2);
% DISPLAY THE LEARNING CURVE AND PRINT EPOCH COUNT
k = 1:length(Jp);
subplot(2,1,1);
plot(k, Jp);
title("Error Criterion Plot J(W)");
xlabel("Number of Epochs for Convergence");
ylabel("J(W)");
subplot(2,1,2);
plot(k, abs(gradJ));
title("Gardient Error Criterion Plot \nablaJ(W)");
xlabel("Number of Epochs for Convergence");
ylabel("\nablaJ(W)");
axis([1 length(Jp) -0.5 3]);

final = [];

% error checking classification
for i=1:length(Z)
    if Z(i) > 0
        final(i) = ceil(Z(i));
    else
        final(i) = floor(Z(i));
    end
end

disp(Z)
disp(final)

% SUCCESS RATE CHECKING
for i=1:length(final)
    if targ(i)==final(i)
        check(i)=1;
    else
        check(i)=0;
    end
end
sRate=length(find(check==1))/length(check)