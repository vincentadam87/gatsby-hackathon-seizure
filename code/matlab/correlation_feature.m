%% Extract_Correlation_Feature
%Extract correlation across channels and make pretty plots

window_width=300;
sliding_width=50;

T_max=size(data,2);
K_max=ceil(1+(T_max-window_width)/sliding_width);

Num_histogram=51;
histogram_all=zeros(Num_histogram,K_max);

for k=1:K_max
    x=data(:,sliding_width*(k-1)+(1:window_width));
    C=corr(x');     % Measure correlation across channels over the sliding window;
    C=triu(C,1);    C(C==0)=NaN;   %Remove the diagonal & Lower triangular portion
    histogram_all(:,k)=hist(C(:),linspace(-1,1,Num_histogram)); % Histogram
end

figure,imagesc(histogram_all)
