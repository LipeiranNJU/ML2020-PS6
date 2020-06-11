from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def load_feature(path):
    df =  pd.read_csv(path, index_col=None, header=None, delimiter=" ")
    df.columns = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n"]
    return df

def load_label(path):
    return pd.read_csv(path, index_col=None, header=None, delimiter=" ")

def pred(models, feature, αs):
    αs = np.array(αs)
    αs = αs/αs.sum()
    rows = feature.shape[0]
    returned = np.zeros((rows))
    for model, α in zip(models, αs):
        returned += model.predict_proba(feature)[:,1] * α
    
    return returned
    
def train_model_once(train_label, train_feature, weight):
    decisiontree_model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=1)
    decisiontree_model.fit(train_feature, train_label, sample_weight=weight)
    pred = decisiontree_model.predict(train_feature)
    train_label = train_label.reshape(pred.shape)
    listdiff = np.array([1 if x[0] != x[1] else 0 for x in zip(list(pred), list(train_label))])
    ε = np.dot(listdiff, weight)
    α = 1/2*np.log((1-ε)/ε)
    new_weight = weight*np.exp(-α*pred*train_label.reshape(pred.shape))
    new_weight = new_weight/new_weight.sum()
    return decisiontree_model, new_weight, ε, α

def main(model_nums=100):
    T = model_nums
    train_feature_path = "./adult_dataset/adult_train_feature.txt"
    train_label_path = "./adult_dataset/adult_train_label.txt"
    test_feature_path = "./adult_dataset/adult_test_feature.txt"
    test_label_path = "./adult_dataset/adult_test_label.txt"

    train_full_feature = load_feature(train_feature_path)
    train_full_label = load_label(train_label_path)


    a = 0
    b = len(train_full_feature)//5
    c = 2*len(train_full_feature)//5
    d = 3*len(train_full_feature)//5
    e = 4*len(train_full_feature)//5
    f = 5*len(train_full_feature)//5
    scale = [a, b, c, d, e, f]
    auc = []
    for i in range(5):
        train_feature = train_full_feature.iloc[list(range(0,scale[i]))+list(range(scale[i+1],scale[-1])),:].values
        train_label = train_full_label.iloc[list(range(0,scale[i]))+list(range(scale[i+1],scale[-1])), :].values

        test_feature = train_full_feature.iloc[scale[i]:scale[i+1],:].values
        test_label = train_full_label.iloc[scale[i]:scale[i+1],:].values

        train_label[train_label==0] = -1
        test_label[test_label==0] = -1

        weight = np.full(len(train_feature), 1/len(train_feature))
        models = []
        αs = []
        for _ in range(T):
            model, weight, ε, α = train_model_once(train_label=train_label, train_feature=train_feature, weight=weight)
            models.append(model)
            αs.append(α)
            # print(i)
            if ε > 0.5:
                break
            else:
                pass


        predict = pred(models, test_feature, αs)
        auc.append(roc_auc_score(test_label, predict))
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


    train_feature = train_full_feature.values
    train_label = train_full_label.values

    test_feature = test_full_feature.values
    test_label = test_full_label.values

    train_label[train_label==0] = -1
    test_label[test_label==0] = -1

    weight = np.full(len(train_feature), 1/len(train_feature))
    models = []
    αs = []
    for _ in range(T):
        model, weight, ε, α = train_model_once(train_label=train_label, train_feature=train_feature, weight=weight)
        models.append(model)
        αs.append(α)
        if ε > 0.5:
            break
        else:
            pass

    predict = pred(models, test_feature, αs)
    return roc_auc_score(test_label, predict)


if __name__ == "__main__":
    for i in range(1000, 1001, 1):
        print(np.mean(main(model_nums=i)))