from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import BoostMain
import matplotlib.pyplot as plt

def load_feature(path):
    df =  pd.read_csv(path, index_col=None, header=None, delimiter=" ")
    df.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    return df

def load_label(path):
    return pd.read_csv(path, index_col=None, header=None, delimiter=" ")


def main(model_nums=100):
    T = model_nums
    train_feature_path = "./adult_dataset/adult_train_feature.txt"
    train_label_path = "./adult_dataset/adult_train_label.txt"

    train_full_feature = load_feature(train_feature_path)
    train_full_label = load_label(train_label_path)
    feature_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    train_full_feature["label"] = train_full_label

    a = 0
    b = len(train_full_feature)//5
    c = 2*len(train_full_feature)//5
    d = 3*len(train_full_feature)//5
    e = 4*len(train_full_feature)//5
    f = 5*len(train_full_feature)//5
    scale = [a, b, c, d, e, f]
    auc = []
    for i in range(5):
        train_feature = train_full_feature.iloc[list(range(0,scale[i]))+list(range(scale[i+1],scale[-1])),:]
        test_feature = train_full_feature.iloc[scale[i]:scale[i+1],:]
        test_label = train_full_label.iloc[scale[i]:scale[i+1],:].values

        models = list()
        feature_names_used = list()
        for i in range(T):
            train_feature_names_used_onetime = sorted(random.sample(feature_names, 4))
            tmp = train_feature.sample(frac=0.5, replace=True)
            train_label_used_onetime = tmp[["label"]]
            train_feature_used_onetime = tmp[train_feature_names_used_onetime]
            decisiontree_model = DecisionTreeClassifier(criterion="entropy", splitter="best")
            decisiontree_model.fit(train_feature_used_onetime, train_label_used_onetime)
            models.append(decisiontree_model)
            feature_names_used.append(train_feature_names_used_onetime)
        
        predictions = np.zeros(len(test_label))
        for model, names in zip(models,feature_names_used):
            prediction = model.predict(test_feature[names])
            predictions = predictions+prediction
        else:
            predictions = predictions/len(models)

        auc.append(roc_auc_score(test_label, predictions))
    else:
        return auc

def test(model_nums):
    T = model_nums
    train_feature_path = "./adult_dataset/adult_train_feature.txt"
    train_label_path = "./adult_dataset/adult_train_label.txt"

    test_feature_path = "./adult_dataset/adult_test_feature.txt"
    test_label_path = "./adult_dataset/adult_test_label.txt"

    train_full_feature = load_feature(train_feature_path)
    train_full_label = load_label(train_label_path)
    test_full_feature = load_feature(test_feature_path)
    test_full_label = load_label(test_label_path)
    feature_names = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    train_full_feature["label"] = train_full_label
    test_full_feature["label"] = test_full_label

    train_feature = train_full_feature
    test_feature = test_full_feature
    test_label = test_full_label.values

    models = list()
    feature_names_used = list()
    for _ in range(T):
        train_feature_names_used_onetime = sorted(random.sample(feature_names, 4))
        tmp = train_feature.sample(frac=0.5, replace=True)
        train_label_used_onetime = tmp[["label"]]
        train_feature_used_onetime = tmp[train_feature_names_used_onetime]
        decisiontree_model = DecisionTreeClassifier(criterion="entropy", splitter="best")
        decisiontree_model.fit(train_feature_used_onetime, train_label_used_onetime)
        models.append(decisiontree_model)
        feature_names_used.append(train_feature_names_used_onetime)
        
    predictions = np.zeros(len(test_label))
    for model, names in zip(models,feature_names_used):
        prediction = model.predict(test_feature[names])
        predictions = predictions+prediction
    else:
        predictions = predictions/len(models)

    return roc_auc_score(test_label, predictions)

if __name__ == "__main__":
    adaboostAUC = []
    randomForestAUC = []
    x = list(range(1, 101))
    for i in range(1, 101): 
        print("第"+str(i)+"次迭代")
        adaboostAUC.append(np.array(BoostMain.main(i)).mean())
        randomForestAUC.append(np.array(main(i)).mean())

    
    plt.title('AUC Result Analysis')
    plt.plot(x, adaboostAUC, color='green', label='adaboostAUC')
    plt.plot(x, randomForestAUC, color='red', label='randomForestAUC')

    plt.legend() # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('AUC')
    plt.show()
    bestadanum = adaboostAUC.index(max(adaboostAUC))+1
    bestrandomForestnum = randomForestAUC.index(max(randomForestAUC))+1
    print(bestadanum)
    print(bestrandomForestnum)
    print(BoostMain.test(bestadanum))
    print(test(bestrandomForestnum))
