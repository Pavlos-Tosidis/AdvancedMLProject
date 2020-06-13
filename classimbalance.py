from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE


def random_undersampler(X, y):
    #print('Class balance before random_undersampler: '+str(sorted(Counter(y).items())))
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    #print('Class balance after random_undersampler: '+str(sorted(Counter(y_resampled).items())))
    return X_resampled, y_resampled


# a lot slower than random undersampling and SMOTE
def tomek_links(X,y):
    #print(sorted(Counter(y).items()))
    tml = TomekLinks()
    X_resampled, y_resampled = tml.fit_resample(X, y)
    #print(sorted(Counter(y_resampled).items()))
    return X_resampled, y_resampled
    
    
def smote(X,y):
    #print('Class balance before SMOTE: '+str(sorted(Counter(y).items())))
    smo = SMOTE(random_state=0)
    X_resampled, y_resampled = smo.fit_resample(X, y)
    #print('Class balance after SMOTE: '+str(sorted(Counter(y_resampled).items())))
    return X_resampled, y_resampled