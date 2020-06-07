import numpy as np
from sklearn.linear_model import LogisticRegression
from costcla.metrics import cost_loss
from costcla.models import BayesMinimumRiskClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
cost_list = [3, 4, 2, 6, 5, 7]
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# Cost minimization on Binary Relevance
def cost_sensitive(X_train, y_train, X_test, y_test, classifier):
    total_cost = 0
    for cnt, label in enumerate(label_names):
        #clf = LogisticRegression(C=12.0)
        #clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True))
        clf = classifier
        print('Processing {} with no cost minimization:'.format(label))
        y_train_b = 1-y_train[:, [cnt]]
        y_test_b = 1-y_test[:, [cnt]]

        cost = cost_list[cnt]
        fp = np.full((y_test_b.shape[0], 1), cost)
        fn = np.full((y_test_b.shape[0], 1), 1)
        tp = np.zeros((y_test_b.shape[0], 1))
        tn = np.zeros((y_test_b.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))

        model = clf.fit(X_train, y_train_b)
        test_pred = model.predict(X_test)
        loss = cost_loss(y_test_b, test_pred, cost_matrix)
        total_cost += loss
        print("Cost :%d" % loss)
    print('Total Cost was: ', total_cost)
    print('\n')
    total_cost_no_calib = 0
    for cnt, label in enumerate(label_names):
        #clf = LogisticRegression(C=12.0)
        #clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True))
        clf = classifier
        print('Processing {} with no calibration:'.format(label))
        y_train_b = 1-y_train[:, [cnt]]
        y_test_b = 1-y_test[:, [cnt]]

        cost = cost_list[cnt]
        fp = np.full((y_test_b.shape[0], 1), cost)
        fn = np.full((y_test_b.shape[0], 1), 1)
        tp = np.zeros((y_test_b.shape[0], 1))
        tn = np.zeros((y_test_b.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))

        model = clf.fit(X_train, y_train_b)
        prob_test = model.predict_proba(X_test)
        bmr = BayesMinimumRiskClassifier(calibration=False)
        test_pred = bmr.predict(prob_test, cost_matrix)

        loss = cost_loss(y_test_b, test_pred, cost_matrix)
        total_cost_no_calib += loss
        print("Cost :%d" % loss)
    print('Total Cost was: ', total_cost_no_calib)
    print('\n')
    total_cost_costcla = 0
    for cnt, label in enumerate(label_names):
        #clf = LogisticRegression(C=12.0)
        #clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True))
        clf = classifier
        print('Processing {} with costcla calibration on training set:'.format(label))
        y_train_b = 1-y_train[:, [cnt]]
        y_test_b = 1-y_test[:, [cnt]]

        cost = cost_list[cnt]
        fp = np.full((y_test_b.shape[0], 1), cost)
        fn = np.full((y_test_b.shape[0], 1), 1)
        tp = np.zeros((y_test_b.shape[0], 1))
        tn = np.zeros((y_test_b.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))

        model = clf.fit(X_train, y_train_b)
        prob_train = model.predict_proba(X_train)
        bmr = BayesMinimumRiskClassifier(calibration=True)
        bmr.fit(y_train_b, prob_train)
        prob_test = model.predict_proba(X_test)
        test_pred = bmr.predict(prob_test, cost_matrix)

        loss = cost_loss(y_test_b, test_pred, cost_matrix)
        total_cost_costcla += loss
        print("Cost :%d" % loss)
    print('Total Cost was: ', total_cost_costcla)
    print('\n')
    from sklearn.calibration import CalibratedClassifierCV
    total_cost_nsigmoid = 0
    for cnt, label in enumerate(label_names):
        #clf = LogisticRegression(C=12.0)
        #clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True))
        clf = classifier
        print('Processing {} with nsigmoid calibration on training set:'.format(label))
        y_train_b = 1-y_train[:, [cnt]]
        y_test_b = 1-y_test[:, [cnt]]

        cost = cost_list[cnt]
        fp = np.full((y_test_b.shape[0], 1), cost)
        fn = np.full((y_test_b.shape[0], 1), 1)
        tp = np.zeros((y_test_b.shape[0], 1))
        tn = np.zeros((y_test_b.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))
        cc = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
        model = cc.fit(X_train, y_train_b)
        prob_test = model.predict_proba(X_test)
        bmr = BayesMinimumRiskClassifier(calibration=False)
        test_pred = bmr.predict(prob_test, cost_matrix)

        loss = cost_loss(y_test_b, test_pred, cost_matrix)
        total_cost_nsigmoid += loss
        print("Cost :%d" % loss)
    print('Total Cost was: ', total_cost_nsigmoid)
    print('\n')

    total_cost_isotonic = 0
    for cnt, label in enumerate(label_names):
        #clf = LogisticRegression(C=12.0)
        #clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True))
        clf = classifier
        print('Processing {} with isotonic calibration on training set:'.format(label))
        y_train_b = 1-y_train[:, [cnt]]
        y_test_b = 1-y_test[:, [cnt]]

        cost = cost_list[cnt]
        fp = np.full((y_test_b.shape[0], 1), cost)
        fn = np.full((y_test_b.shape[0], 1), 1)
        tp = np.zeros((y_test_b.shape[0], 1))
        tn = np.zeros((y_test_b.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))
        cc = CalibratedClassifierCV(clf, method="isotonic", cv=3)
        model = cc.fit(X_train, y_train_b)
        prob_test = model.predict_proba(X_test)
        bmr = BayesMinimumRiskClassifier(calibration=False)
        test_pred = bmr.predict(prob_test, cost_matrix)

        loss = cost_loss(y_test_b, test_pred, cost_matrix)
        total_cost_isotonic += loss
        print("Cost :%d" % loss)
    print('Total Cost was: ', total_cost_isotonic)
    print('\n')

    # minimize sum i to n -(w0 * log(yhat_i) * y_i + w1 * log(1 – yhat_i) * (1 – y_i))
    from sklearn.utils.class_weight import compute_class_weight
    total_cost_cw = 0
    for cnt, label in enumerate(label_names):
        if classifier == LogisticRegression(C=12.0):
            clf = LogisticRegression(C=12.0, class_weight='balanced')
        else:
            clf = make_pipeline(StandardScaler(with_mean=False), SVC(gamma='auto', probability=True, class_weight='balanced'))
        print('Processing {} with class weighting:'.format(label))
        y_train_b = 1-y_train[:, [cnt]]
        y_test_b = 1-y_test[:, [cnt]]
        weighting = compute_class_weight('balanced', [0, 1], y_train_b.reshape(y_train_b.shape[0],))
        print('Weighting is: ', weighting)
        cost = cost_list[cnt]
        fp = np.full((y_test_b.shape[0], 1), cost)
        fn = np.full((y_test_b.shape[0], 1), 1)
        tp = np.zeros((y_test_b.shape[0], 1))
        tn = np.zeros((y_test_b.shape[0], 1))
        cost_matrix = np.hstack((fp, fn, tp, tn))

        model = clf.fit(X_train, y_train_b)
        test_pred = model.predict(X_test)
        loss = cost_loss(y_test_b, test_pred, cost_matrix)
        total_cost_cw += loss
        print("Cost :%d" % loss)
    print('Total Cost was: ', total_cost_cw)
    print('\n')

    print('*'*10+'*'*10)
    print('{:>20} | {:>20} | {:>20} | {:>20} | {:>20} | {:>20} |'
          .format('No weight min', 'no calibration', 'costcla', 'nsigmoid', 'isotonic', 'class_weighting'))
    print('{:>20} | {:>20} | {:>20} | {:>20} | {:>20} | {:>20} |'
          .format(total_cost, total_cost_no_calib, total_cost_costcla, total_cost_nsigmoid, total_cost_isotonic, total_cost_cw))
    print('*'*10+'*'*10)
