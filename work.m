%% ===============================================================
%  GOLD PRICE MODELING (OCTAVE COMPATIBLE VERSION)
%  Models:
%     - Linear Regression (Polyfit degree 1)
%     - Polynomial Regression (Polyfit degree 3)
%  Note: SVM and Random Forest removed (Not available in Octave)
%% ===============================================================

clc; clear; close all;

%% 1. Load Dataset (Manual CSV parsing for Octave)
filename = 'annual_gold_rate.csv';
if exist(filename, 'file') ~= 2
    error('File not found: %s. Please upload it to Colab files.', filename);
end

fid = fopen(filename);
% Scan file: String (Date), Float (USD)
C = textscan(fid, '%s %f', 'Delimiter', ',', 'HeaderLines', 1);
fclose(fid);

dateStrings = C{1};
prices = C{2};

% Extract Year manually from "yyyy-MM-dd" string
years = zeros(length(dateStrings), 1);
for i = 1:length(dateStrings)
    % Take first 4 chars as year
    years(i) = str2double(dateStrings{i}(1:4));
end

% Remove NaNs if any
validIdx = ~isnan(prices) & ~isnan(years);
years = years(validIdx);
prices = prices(validIdx);

disp(['Loaded rows: ', num2str(length(years))]);

%% 2. Train/Test Split (80/20)
% Manual split since cvpartition is missing
rand('seed', 42); % Set seed for reproducibility
n = length(years);
idx = randperm(n);

nTrain = floor(0.8 * n);
trainIdx = idx(1:nTrain);
testIdx = idx(nTrain+1:end);

Xtrain = years(trainIdx);
Ytrain = prices(trainIdx);

Xtest = years(testIdx);
Ytest = prices(testIdx);

%% 3. Train Models (Using Polyfit)

% --- Linear Model (Degree 1) ---
pLinear = polyfit(Xtrain, Ytrain, 1);

% --- Polynomial Model (Degree 3) ---
pPoly3 = polyfit(Xtrain, Ytrain, 3);

%% 4. Evaluate Models

% Predict on Test Data
yPredLinear = polyval(pLinear, Xtest);
yPredPoly3  = polyval(pPoly3, Xtest);

% Define helper for metrics
calcR2 = @(y, yhat) 1 - sum((y - yhat).^2) / sum((y - mean(y)).^2);
calcMAE = @(y, yhat) mean(abs(y - yhat));
calcRMSE = @(y, yhat) sqrt(mean((y - yhat).^2));

fprintf('\nPerformance on Test Set:\n');
fprintf('--------------------------------------------------\n');
fprintf('Model      | R2       | MAE      | RMSE\n');
fprintf('--------------------------------------------------\n');

% Linear Metrics
r2_lin = calcR2(Ytest, yPredLinear);
mae_lin = calcMAE(Ytest, yPredLinear);
rmse_lin = calcRMSE(Ytest, yPredLinear);
fprintf('Linear     | %.4f   | %.2f   | %.2f\n', r2_lin, mae_lin, rmse_lin);

% Poly3 Metrics
r2_poly = calcR2(Ytest, yPredPoly3);
mae_poly = calcMAE(Ytest, yPredPoly3);
rmse_poly = calcRMSE(Ytest, yPredPoly3);
fprintf('Poly3      | %.4f   | %.2f   | %.2f\n', r2_poly, mae_poly, rmse_poly);

%% 5. Pick Best Model
if r2_poly > r2_lin
    bestName = 'Polynomial (Deg 3)';
    bestP = pPoly3;
else
    bestName = 'Linear';
    bestP = pLinear;
end

fprintf('\nBest Model: %s\n', bestName);

%% 6. Plotting
yearsAll = (min(years):max(years))';
predAll = polyval(bestP, yearsAll);

figure;
plot(years, prices, 'o', 'MarkerFaceColor', 'b'); hold on;
plot(yearsAll, predAll, 'r-', 'LineWidth', 2);
xlabel('Year');
ylabel('Gold Price (USD)');
title(['Gold Price Prediction - ' bestName]);
legend('Actual Data', 'Prediction');
grid on;
% Save plot because Colab doesn't always show popups nicely
print('prediction_plot.png', '-dpng');
disp('Plot saved as prediction_plot.png');

%% 7. Prediction Function
predictYear = @(yr) polyval(bestP, yr);

fprintf('\nPrediction for 2025: %.2f USD\n', predictYear(2025));
fprintf('Prediction for 2030: %.2f USD\n', predictYear(2030));