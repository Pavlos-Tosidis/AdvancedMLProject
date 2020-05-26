from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


def random_undersampler(X, y):
    print(sorted(Counter(y).items()))
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print(sorted(Counter(y_resampled).items()))