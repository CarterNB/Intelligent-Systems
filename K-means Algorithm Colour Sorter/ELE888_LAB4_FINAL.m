close all
clear all

I=imread('house.tiff');
X=reshape(I, 256*256, 3);
X=double(X);
R = [1,0,0];
G = [0,1,0];
B = [0,0,1];
figure, plot3(X(:,1), X(:,2), X(:,3),'.','Color',[0, 0, 0])
xlabel('R');
ylabel('G');
zlabel('B');
title('Unlabeled Sample Data of house.tiff Pixels in RGB Space')

c=2;

% for i=1:(c)
%     for j=1:3
%      tmp1=0
%      tmp2=256
%      M_INIT(i,j)=(tmp2-tmp1).*rand(1,1)+tmp1;
%      M_INIT(i,j)= round(M_INIT(i,j),2)
%     end
% end

M= [18.230000000000	    156.020000000000	211.680000000000
    98.100000000000	    65.3500000000000	38.2200000000000];
last_M=zeros(size(M));
Jplot=[];
N1=0;
M1_prev = M(1,:);
M2_prev = M(2,:);
while (1)
    N1=N1+1;
    last_M=M;
    J=zeros(size(X,1),c);
    for i=1:1:c
        J3 = (X - repmat(M(i,:), size(X,1),1));
        J3 = sum(J3.^2, 2);
        J(:,i)=J3;
    end
   
    Jplot=[Jplot sum(min(J(:,1),J(:,2)))];

    [~,C]=min(J,[],2);
    for i=1:1:c
        C_2=(C==i);
        M(i,:) = sum(X(C_2,:))/sum(C_2);
       
    end
    X_label=zeros(size(X));
    for i=1:1:c
        C_2=(C==i);
        X_label=X_label + repmat(M(i,:), size(X,1), 1) .* repmat(C_2, 1, size(X,2));
    end
    M1_prev = [M1_prev; M(1,:)];
    M2_prev = [M2_prev; M(2,:)];
    if (last_M==M) 
        break;
    end   
    if (N1>50) 
        break;
    end  

end
%PART i
figure 
plot(Jplot);
xlabel('Iteration');
ylabel('J');
title('Error Criterion J for each iteration (c = 2)')
%Part ii
figure 
plot3(M1_prev(:,1), M1_prev(:,2), M1_prev(:,3),'-o');
hold all
plot3(M2_prev(:,1), M2_prev(:,2), M2_prev(:,3),'-o');
title('Plot of cluster means (c = 2)')
xlabel('R');
ylabel('G');
zlabel('B');

%Part iii
C_1=~C_2;
figure
val1 = X(C_1,:); 
val2 = X(C_2,:);
plot3(val1(:,1),val1(:,2),val1(:,3),'.','Color', M(1,:)/256);
hold all
plot3(val2(:,1),val2(:,2),val2(:,3),'.','Color', M(2,:)/256);
title('Labeled data samples in RGB space (c = 2)')
xlabel('R');
ylabel('G');
zlabel('B');

%PART iv
figure
X_label = zeros(size(X));
for i = 1:1:c
    C_2 = (C==i);
    X_label = X_label + repmat(M(i,:), size(X,1), 1) .* repmat(C_2, 1, size(X,2));
end
X_label = reshape(X_label, size(I,1),size(I,2),3);

subplot(1,2,1);
imshow(I)
title('Original Image')
subplot(1,2,2);
imshow(X_label/256)
title('Colour Labeled Image For c=2')

%%%%%%%%%%%%%%%%%%%PART B%%%%%%%%%%%%%%%%%%%%%%%%%%
c=5;

% for i=1:(c)
%     for j=1:3
%      tmp1=0
%      tmp2=256
%      M_INIT(i,j)=(tmp2-tmp1).*rand(1,1)+tmp1;
%      M_INIT(i,j)= round(M_INIT(i,j),2)
%     end
% end

%Initial mean randomly generated using above equation
M_INIT1=[
        192.320000000000	65.3000000000000	129.530000000000
        178.960000000000	228.070000000000	245.580000000000
        140.090000000000	35.4900000000000	38.2200000000000
        65.9200000000000	215.220000000000	65.1000000000000
        208.460000000000	62.3400000000000	237.890000000000];

M_INIT2=[
        19.4200000000000	13.8100000000000	135.880000000000
        199.470000000000	239.110000000000	33.2600000000000
        145.620000000000	120.160000000000	3.05000000000000
        86.3000000000000	41.5200000000000	203.340000000000
        79.6700000000000	135.300000000000	42.4100000000000];

last_M=zeros(size(M));

N2=0;
while (1)
    N2=N2+1;
    last_M=M_INIT1;
    J=zeros(size(X,1),c);
    for i=1:1:c
        J3 = (X - repmat(M_INIT1(i,:), size(X,1),1));
        J3 = sum(J3.^2, 2);
        J(:,i)=J3;
    end

    [~,CB1]=min(J,[],2);
    for i=1:1:c
        C_5=(CB1==i);
        M_INIT1(i,:) = sum(X(C_5,:))/sum(C_5);
       
    end
    X_label=zeros(size(X));
    for i=1:1:c
        C_5=(CB1==i);
        X_label=X_label + repmat(M_INIT1(i,:), size(X,1), 1) .* repmat(C_5, 1, size(X,2));
    end
    if (last_M==M_INIT1) 
        break;
    end   
    if (N1>50) 
        break;
    end  

end

figure
X_label = reshape(X_label, size(I,1),size(I,2),3);
subplot(1,2,1);
imshow(I)
title('Original Image')
subplot(1,2,2);
imshow(X_label/256)
title('Colour Labeled Image For c=5 (1st Version)')

% FIGURE BELOW WON'T DISPLAY AND IDK WHY -- fixed %

figure
for i = 1:1:c
    C_5 = (CB1==i);
    Xi = X(C_5, :);
    plot3(Xi(:,1), Xi(:,2), Xi(:,3),'.','Color', M_INIT1(i,:)/256)
    hold all
end
title('Labeled data samples in RGB space for c=5 (1st Version)')
xlabel('R');
ylabel('G');
zlabel('B');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
last_M=zeros(size(M));

N3=0;

while (1)
    N3=N3+1;
    last_M=M_INIT2;
    J=zeros(size(X,1),c);
    for i=1:1:c
        J3 = (X - repmat(M_INIT2(i,:), size(X,1),1));
        J3 = sum(J3.^2, 2);
        J(:,i)=J3;
    end

    [~,CB2]=min(J,[],2);
    for i=1:1:c
        C_5=(CB2==i);
        M_INIT2(i,:) = sum(X(C_5,:))/sum(C_5);
    end
    X_label=zeros(size(X));
    for i=1:1:c
        C_5=(CB2==i);
        X_label=X_label + repmat(M_INIT2(i,:), size(X,1), 1) .* repmat(C_5, 1, size(X,2));
    end
    if (last_M==M_INIT2) 
        break;
    end   
    if (N2>50) 
        break;
    end  

end

figure
X_label = reshape(X_label, size(I,1),size(I,2),3);
subplot(1,2,1);
imshow(I)
title('Original Image')
subplot(1,2,2);
imshow(X_label/256)
title('Colour Labeled Image For c=5 (2nd Version)')

figure
for i = 1:1:c
    C_5 = (CB2==i);
    Xi = X(C_5, :);
    plot3(Xi(:,1), Xi(:,2), Xi(:,3),'.','Color', M_INIT2(i,:)/256)
    hold all
end
title('Labeled data samples in RGB space for c=5 (2nd Version)')
xlabel('R');
ylabel('G');
zlabel('B');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PART C %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = size(X,1);
XB1 = 0;

for i = 1:1:c
    C_5 = (CB1==i);
    Xi = X(C_5, :);
    mindist = sort(sum((M_INIT1- repmat(M_INIT1(i,:), c, 1)).^2, 2).^.5);
    %% mindist(2) is min, non zero value
    XB1 = XB1 + sum(sum((Xi - repmat(M_INIT1(i,:), size(Xi,1), 1)).^2, 2).^.5) / mindist(2);
end

disp('Xie-Beni (XB) Index for Version 1:')
XB1 = XB1 / N

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
XB2 = 0;

for i = 1:1:c
    C_5 = (CB2==i);
    Xi = X(C_5, :);
    mindist = sort(sum((M_INIT2 - repmat(M_INIT2(i,:), c, 1)).^2, 2).^.5);
    %% mindist(2) is min, non zero value
    XB2 = XB2 + sum(sum((Xi - repmat(M_INIT2(i,:), size(Xi,1), 1)).^2, 2).^.5) / mindist(2); 
end

disp('Xie-Beni (XB) Index for Version 2:')
XB2 = XB2 / N