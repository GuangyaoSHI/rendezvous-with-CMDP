function [P] = powerStocWeight(varargin)
% Returns electrical power requirement as function of velocity s.t. wind
% and weight stochasticity
% For any kinematic simulation used to evaluate the planner, I recommend
% only querying this power value once per UAS leg and using the result as
% the average power requirement for that leg.
nargin = length(varargin);
if nargin > 4
    error('windspeed:TooManyInputs','only 4 optional inputs')
end

% tgtV = target velocity in m/s, expected cruise is about 14 m/s
% meanW = average weight in kg
% stdW = standard deviation of weight in kg
% aG = characteristic velocity for
% bG = shape function of veloicty distribution, from https://wind-data.ch/tools/weibull.php
% heading = wind heading angle m/s, randomly sampled unless provided by user

%optargs = {14 rand*2*pi 2.35 0.15 4.2534 6}; %default parameters
optargs = {14 rand*2*pi 2.3 0.05 1.5 3};


optargs(1:nargin) = varargin;
[tgtV, heading, meanW, stdW, aG, bG] = optargs{:};

% sample weight using random normal distribution of weight
W = stdW*randn+meanW;

% solve for true airspeed by adding in weibull wind distribution with
% equally random heading direction of wind
% simplifying assumption is that only wind tangential to UAS heading affects power
V = abs(tgtV + wblrnd(aG,bG)*cos(-heading));

%empirical fit of relationship between velocity, weight, and power (85% electrical efficiency)
b = [-88.7661477109457;3.53178719017177;-0.420567520590965;0.0427521866683907;107.473389967445;-2.73619087492112];
P = b(1)+b(2)*V+b(3)*V.^2+b(4)*V.^3+b(5)*W+b(6)*V.*W;



end


