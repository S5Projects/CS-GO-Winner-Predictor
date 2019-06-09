import pandas as pd
import numpy as np
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    GridSearchCV,
    learning_curve
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
 
sns.set_style('darkgrid')
forest = RandomForestClassifier(n_estimators=100)


# データの読み込み
demo_df = pd.read_csv("data.csv")
demo_df = demo_df.drop("map",axis=1).replace("CT",True).replace("T",False) #CTが勝利した場合をTrueとする
predict_df = pd.read_csv("predict.csv")
predict_df = predict_df.drop("map",axis=1).replace("CT",True).replace("T",False)

#plt.scatter(demo_df["CT_EV"], demo_df["T_EV"])
#plt.show()

# 学習
forest = forest.fit(demo_df.drop(["Winside"],axis=1), demo_df["Winside"])
test_data = predict_df.drop(["Winside"],axis=1).values

print("---実際のデータ---")
print(predict_df["Winside"])
print("---予想データ---")
print(forest.predict(test_data))

index = 0
accuracy = []

for item in predict_df["Winside"]:
    if predict_df["Winside"][index] == forest.predict(test_data)[index]:
        print("ラウンド" + str(index + 1) + "の予想されたデータは正しいです！")
        accuracy.append(True)
    else:
        print("ラウンド" + str(index + 1) + "の予想されたデータは正しくありません...")
        accuracy.append(False)
    index += 1

accurate = 0
for item in accuracy:
    if item == True:
        accurate = accurate+1

rate = '{:0.2%}'.format(accurate / len(accuracy))
print("正答率は" + rate + "です")
input("")