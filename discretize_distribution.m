keySet = {'Jan','Feb','Mar','Apr'};
valueSet = [327.2 368.2 197.6 178.4];
M = containers.Map(keySet,valueSet)

intervals = {}
for i=100:10:300
    intervals{end+1} = [i, i+10];
end

nums = zeros(size(intervals));

distribution = containers.Map(intervals, nums);