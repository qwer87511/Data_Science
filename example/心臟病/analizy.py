import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# # from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as clf_rtree
# # from sklearn.model_selection import train_test_split as sk_split
from sklearn import tree
# # from sklearn.metrics import confusion_matrix
# # from sklearn.metrics import roc_curve, auc
import pydotplus
import shap
# # import matplotlib
# # # 調整圖片大小
# # plt.figure(figsize=(16,4),dpi=50)

# def visualization_roc(tpr, fpr):
#     plt.yticks(fontsize = 2)
#     plt.plot(fpr, tpr,c =" .3")
#     plt.title('ROC curve')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.grid(True)
#     plt.savefig("test-output/roc.png")
#     plt.show()

def scatter_plot(data,x,y):
    # print(x,y)
    plt.scatter(x = data[x], y = data[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.show()
    plt.close()
    
def bar_plot(data,feature):
    target1 = data[data["target"]==1]
    target0 = data[data["target"]==0]
    group1 = target1.groupby(feature1).size()
    group0 = target0.groupby(feature0).size()
    bar_width = 0.3
    # fig, ax = plt.subplots()
    A = plt.bar(0,
                group1.values, 
                bar_width,
                alpha=.4,
                label="1") 
    B = plt.bar(0.2,
                group0.values,
                bar_width,
                alpha=.4,
                label="0")
                plt.ylabel("Mean score")
    plt.xlabel("Subject")
    plt.title("Test Scores by Contry")
    plt.xticks(index+.3 / 2 ,("Math","Reading","Science"))
    plt.legend() 
    plt.grid(True)
    # plt.show()
    # print(group.values)
    # plt.bar(x = group.index, height = group.values, width = 0.3)
    # plt.xlabel(feature)
    plt.show()


def analizy(data):
    # print(data.corr()["target"])

    desc = data.describe()
    for i in data.columns:
        bar_plot(data,i)
# # 讀取資料
def Read_data(path):
    data=pd.read_csv(path)
    data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
    feature = data[data.columns[:-1]]
    target = data[data.columns[-1]]
    return feature, target

# # 權重的影響
# def var_influence(model, test_feature, path):
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(test_feature)
#     shap.summary_plot(shap_values[1], test_feature, plot_type="bar")
#     shap.summary_plot(shap_values[1], test_feature)

    
# # 將random forest tree畫出來
def draw_tree(model, column, path):
    estimator = model.estimators_[5]
    # 將樹輸出成dot檔
    dot_data = tree.export_graphviz(estimator,filled=True,feature_names=list(column))
    # 讀取dot檔並繪製出來
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(path+"random_classification_tre_out.png")

# def t_f_postive(test_target,pre_target):
#     # true,false positve tpr fpr 計算
#     tn, fp, fn, tp  = confusion_matrix(test_target,pre_target).ravel()
#     fpr, tpr, thresholds = roc_curve(test_target,pre_target)
#     print('sensitivy: ',tp/(tp+fn))
#     Auc = auc(fpr, tpr)
#     print(Auc)
#     visualization_roc(tpr,fpr)
    


def main():
    path = 'heart.csv'
    feature, target = Read_data(path)
#     save_path = "test-output/"
#     # 資料切分
#     train_feature, test_feature, train_target, test_target = sk_split(feature, target, test_size=0.3, random_state=10) # 以隨機的方式資料分割 並給隨機種子固定隨機模式
    
#     # 模型選擇
#     model = clf_rtree(max_depth=5,min_samples_leaf=4,n_estimators=10,random_state=10) # 限制決策樹最大深度 避免overfitting 
#     model.fit(train_feature, train_target)
#     # 模型預測
#     pre_target = model.predict(test_feature)
    
#     # true false positive sensitive ROC 
#     t_f_postive(test_target,pre_target)
    
#     # accurancy 正確率
#     scor = model.score(test_feature,test_target)
#     print('scor: ',scor)

#     # 各feature 的影響程度
#     var_influence(model, test_feature, save_path)
    
#     # 畫隨機森林樹出來

#     column = train_feature.columns
#     draw_tree(model , column , save_path)
    
    

if __name__=="__main__":
    main()

