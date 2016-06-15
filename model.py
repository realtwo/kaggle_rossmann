import preprocess
from sklearn.metrics import mean_squared_error
from time import time

seed = 53

def performance_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)**0.5


def fit_and_predict(clf, x_train, y_train, x_test):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def evaluate_model(clf, x_train, y_train, x_test, y_test):
    start = time()
    y_pred = fit_and_predict(clf, x_train, y_train, x_test)
    end = time()      
    print '----------------------'
    print clf
    print "Trained model in {:.4f} seconds".format(end - start) 
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=y_pred))


def basic_training(x_train, x_test, y_train, y_test):
  
    print 'Training basic models... '

    # Baseline 1: use mean
    print '----------------------'
    print 'Baseline: mean sales'
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=x_test['MeanSales']))

    # Baseline 2: use median
    print '----------------------'
    print 'Baseline: median sales'
    print "RMSE: {}".format(performance_metric(y_true=y_test, y_pred=x_test['MedianSales']))

    # Learning models
    from sklearn.ensemble import RandomForestRegressor
    clf = RandomForestRegressor(random_state=seed)
    evaluate_model(clf, x_train, y_train, x_test, y_test)

    print 'Done!'  

def tuning():
    pass




def main():  
    x_all, y_all = preprocess.build_feature_label()
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, 
                                            test_size=0.3, random_state=seed)

    basic_training(x_train, x_test, y_train, y_test)
  

if __name__ == "__main__":
    main()

    

