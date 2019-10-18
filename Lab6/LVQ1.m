% Function to apply LVQ1 algorithm to a dataset
%  data is the entire dataset in a matrix
%  k is the number of prototypes
%  n is the learning rate
%  tmax is the number of epochs
function LVQ1(data, k, n, tmax)

    N = size(data, 1);
    prototypes = datasample(data, k);
    points_class1 = data(1:50, :);
    points_class2 = data(51:end, :);
    
    % Plot initial state of the data
    figure
    plot(data(:, 1), data(:, 2), 'b.')
    hold on
    plot(prototypes(:, 1), prototypes(:, 2), 'ro', 'MarkerFaceColor','r')
    title('Initial dataset with the prototypes')
    xlabel('x')
    ylabel('y')
    legend('Data points', 'Prototypes')
    hold off
    
    cost_function = zeros(1, tmax);
    
    for t = 1:tmax
        
        shuffled_indexes = randperm(N);
        cost = 0;
        
        for idx = shuffled_indexes
            distances = pdist2(data(idx,:),prototypes);
            [d, p] = min(distances);
            
            for i = 1:k
                if i == p
                    prototypes(p,:) = prototypes(p,:) + n*(data(idx,:) - prototypes(i,:));
                else
                    prototypes(p,:) = prototypes(p,:) - n*(data(idx,:) - prototypes(i,:));
                end
                
            end
            
            
            cost = cost + d.^2;
        end
        
        % Build title for the plot
        p_title = 'Data set with prototypes with K=';
        p_title = strcat(p_title, sprintf("%d", k));
        p_title = strcat(p_title, ', LR=');
        p_title = strcat(p_title, sprintf("%.2f", n));
        p_title = strcat(p_title, ', t=');
        p_title = strcat(p_title, sprintf("%d", t));
        
        % Plot the data for the current epoch
        figure
        plot(data(:, 1), data(:, 2), 'b.')
        hold on
        plot(prototypes(:, 1), prototypes(:, 2), 'ro', 'MarkerFaceColor','r')
        title(p_title)
        xlabel('x')
        ylabel('y')
        legend('Data points', 'Prototypes')
        hold off
        
        cost_function(t) = cost;
        
    end
    
    % Plot final state of the data
    figure
    plot(data(:, 1), data(:, 2), 'b.')
    hold on
    plot(prototypes(:, 1), prototypes(:, 2), 'ro', 'MarkerFaceColor','r')
    title('Final dataset with the prototypes')
    xlabel('x')
    ylabel('y')
    legend('Data points', 'Prototypes')
    hold off
    
    % Build title for the plot
    p_title = 'Quantization error over epochs with K=';
    p_title = strcat(p_title, sprintf("%d", k));
    p_title = strcat(p_title, ', LR=');
    p_title = strcat(p_title, sprintf("%.2f", n));
    % Plot quantization error or all epochs
    figure
    plot(1:tmax, cost_function, 'k')
    xlabel('Epoch')
    ylabel('Quantization error')
    title(p_title)
    xlim([1, tmax])
    hold off
    
end
