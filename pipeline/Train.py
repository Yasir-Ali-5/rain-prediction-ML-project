from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib as j

def train_model(X,y,a,b):
    
    score = {}

    RF = RandomForestClassifier(
    n_estimators = 50,       # number of trees (default is 100)
    max_depth=25,           # limit depth to avoid overfitting
    min_samples_split=10,   # minimum samples to split a node
    min_samples_leaf = 4,
    max_features = 'log2'
    )
    RF.fit(X,y)
    score["RF"] = RF.score(a,b)

    knn = KNeighborsClassifier()
    knn.fit(X,y)
    score["knn"] = knn.score(a,b)

    svm = SVC(kernel="poly")
    svm.fit(X,y)
    score["svm"] = svm.score(a,b)

    LR = LogisticRegression(max_iter=10000)
    LR.fit(X,y)
    score["LR"] = LR.score(a,b)
    
    return score



def save_model(model):
    j.dump(model, "models/model.pkl")