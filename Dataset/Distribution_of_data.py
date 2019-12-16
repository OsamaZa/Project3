from pylab import *

redwine = [10,53,681,638,199,18]
red = [3,4,5,6,7,8]

whitewine = [20,163,1457,2198,880,175,5]
white = [3,4,5,6,7,8,9]

figure("Distribution of Quality Score for Red Wine")
title("Distribution of Quality Score for Red Wine")
xticks(red, ('3','4','5','6','7','8'))
xlabel("Quality score")
ylabel("Number of datapoints")
bar(red,redwine)

figure("Distribution of Quality Score for White Wine")
title("Distribution of Quality Score for White Wine")
xticks(white, ('3','4','5','6','7','8','9'))
xlabel("Quality score")
ylabel("Number of datapoints")
bar(white,whitewine)

show()
