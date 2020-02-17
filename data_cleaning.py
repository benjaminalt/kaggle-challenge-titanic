import pandas as pd


def impute_by_column(data: pd.DataFrame, target: str, col:str) -> pd.DataFrame:
    means = data.groupby(col)[target].mean()
    data = data.join(means, on=col, rsuffix="Mean")
    data[target].fillna(data[target + "Mean"], inplace=True)
    del data[target + "Mean"]
    return data


def impute_age(data: pd.DataFrame) -> pd.DataFrame:
    # Impute age. What is the best predictor for age? --> Pclass
    # print("=" * 10)
    # print("Correlation with age: ")
    # for x in filter(lambda x: x != "Age", data.columns):
    #     corr = data.where(pd.notna(data["Age"]))[["Age", x]].corr().iloc[0,1]
    #     print("{}: {:.3f}".format(x, corr))
    return impute_by_column(data, "Age", "Pclass")


def impute_fare(data: pd.DataFrame) -> pd.DataFrame:
    # Impute fare. What is the best predictor for fare? --> Pclass
    # print("=" * 10)
    # print("Correlation with fare: ")
    # for x in filter(lambda x: x != "Fare", data.columns):
    #     corr = data.where(pd.notna(data["Fare"]))[["Fare", x]].corr().iloc[0,1]
    #     print("{}: {:.3f}".format(x, corr))
    return impute_by_column(data, "Fare", "Pclass")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Categorical to binary / one-hot
    data["Sex"] = (data["Sex"] == "male").astype(int)
    data["HasCabin"] = (~data["Cabin"].isna()).astype(int)
    data = pd.concat(
        [data.drop("Embarked", axis=1), pd.get_dummies(data["Embarked"], dummy_na=True, prefix="Embarked")], axis=1)

    # Drop columns not used for regression
    del data["PassengerId"]
    del data["Name"]
    del data["Ticket"]
    del data["Cabin"]

    data = impute_age(data)
    data = impute_fare(data)

    return data
