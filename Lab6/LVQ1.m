% Function to apply LVQ1 algorithm to a dataset
%  data is the entire dataset in a matrix
%  k is the number of prototypes per class
%  n is the learning rate
%  tmax is the minimum number of epochs to consider the error stable

function LVQ1(data, k, n, tmax)

    % We have 2 classes so we have 2*k prototypes
    k = k*2;
    
    % Size of the whole dataset
    N = size(data, 1);
    
    points_class1 = data(1:N/2,:);
    points_class2 = data((N/2 + 1):end,:);
    
    prototypes = [datasample(points_class1, k/2) ; datasample(points_class2, k/2)];

    % Plot initial state of the data
    figure
    plot(points_class1(:, 1), points_class1(:, 2), 'b.', 'MarkerSize', 10)
    hold on
    plot(prototypes(1:k/2, 1), prototypes(1:k/2, 2), 'bo', 'MarkerFaceColor','g', 'MarkerSize', 10)
    plot(points_class2(:, 1), points_class2(:, 2), 'r.', 'MarkerSize', 10)
    plot(prototypes(k/2+1:end, 1), prototypes(k/2+1:end, 2), 'ro', 'MarkerFaceColor','g', 'MarkerSize', 10)
    title('Initial dataset with prototypes')
    xlabel('x')
    ylabel('y')
    legend('Class 1', 'Prototypes of class 1', 'Class 2', 'Prototypes of class 2')
    hold off
    
    % List of the error in each epoch
    error = [];
    t = 1;
    equal_epochs = 0;
    % Loop over the epochs
    while equal_epochs ~= tmax
        % Add one element to the error list
        error = [error 0];
        
        % We shuffle the indices
        shuffled_indexes = randperm(N);
        
        for idx = shuffled_indexes
            distances = pdist2(data(idx,:),prototypes);
            [~, p] = min(distances);
            
            
            % If the winner belongs to class 1
            if p <= k/2
                % If the data point belongs to class 1
                if idx <= N/2
                    % Move it closer
                    prototypes(p,:) = prototypes(p,:) - (n*(prototypes(p,:) - data(idx,:)));
                % If the data point belongs to class 2
                else
                    % Move it further
                    prototypes(p,:) = prototypes(p,:) + (n*(prototypes(p,:) - data(idx,:)));
                end
            % If the winner belongs to class 2
            else
                % If the data point belongs to class 1
                if idx <= N/2
                    % Move it further
                    prototypes(p,:) = prototypes(p,:) + (n*(prototypes(p,:) - data(idx,:)));
                % If the data point belongs to class 2
                else
                    % Move it closer
                    prototypes(p,:) = prototypes(p,:) - (n*(prototypes(p,:) - data(idx,:)));
                end
            end
        end
        
        for i = 1:N
            distances = pdist2(data(i,:),prototypes);
            [~, p] = min(distances);
            
            if i <= N/2
                if p > k/2
                    error(t) = error(t) + 1;
                end
            else
                if p <= k/2
                    error(t) = error(t) + 1;
                end
            end
        end
        
        error(t) = (error(t)*100)/N;
        
        if t > 1 && error(t-1) == error(t)
            equal_epochs = equal_epochs + 1;
        else
            equal_epochs = 0;
        end
        
        t = t + 1;
        
    end
    
    % Plot final state of the data
    figure
    plot(points_class1(:, 1), points_class1(:, 2), 'b.', 'MarkerSize', 10)
    hold on
    plot(prototypes(1:k/2, 1), prototypes(1:k/2, 2), 'bo', 'MarkerFaceColor','g', 'MarkerSize', 10)
    plot(points_class2(:, 1), points_class2(:, 2), 'r.', 'MarkerSize', 10)
    plot(prototypes(k/2+1:end, 1), prototypes(k/2+1:end, 2), 'ro', 'MarkerFaceColor','g', 'MarkerSize', 10)
    title('Final dataset with the prototypes')
    xlabel('x')
    ylabel('y')
    legend('Class 1', 'Prototypes of class 1', 'Class 2', 'Prototypes of class 2')
    hold off
    
    % Plot quantization error or all epochs
    figure
    plot(1:numel(error), error, 'k')
    xlabel('Epoch')
    ylabel('Error (%)')
    title('Error plot')
    ylim([1,100])
    hold off
    
end
