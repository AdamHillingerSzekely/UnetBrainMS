# code for Mann-Whitney U test
from scipy.stats import mannwhitneyu
# Take batch 1 and batch 2 data as per above example
batch_1 =[0.5856,	0.588,	0.5891,	0.5857,	0.5874















]


batch_2 =[0.5832,	0.5833,	0.583,	0.584,	0.586



]
  
# perform mann whitney test
stat1, p_value1 = mannwhitneyu(batch_1, batch_2)
print('Statistics=%.2f, p=%.10f' % (stat1, p_value1))

#stat2, p_value2 = mannwhitneyu(batch_1, batch_3)
#print('Statistics=%.2f, p=%.10f' % (stat2, p_value2))
# Level of significance
alpha = 0.05
# conclusion
if p_value1 < alpha:
    print('Reject Null Hypothesis (Significant difference between two samples)')
else:
    print('Do not Reject Null Hypothesis (No significant difference between two samples)')

#if p_value2 < alpha:
#    print('Reject Null Hypothesis (Significant difference between two samples)')
#else:
#    print('Do not Reject Null Hypothesis (No significant difference between two samples)')