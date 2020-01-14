from sklearn.tree import (
    DecisionTreeClassifier,
    export_graphviz,
    )
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    cancer = load_breast_cancer()
    print(cancer.keys())

    X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                        cancer.target,
                                                        stratify=cancer.target, test_size=1.0/5)
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    export_graphviz(tree, out_file="tree.dot", class_names=cancer.target_names,
                    feature_names=cancer.feature_names, impurity=False,
                    filled=True)

    with open("tree.dot") as reader:
        dot_file = reader.read()


