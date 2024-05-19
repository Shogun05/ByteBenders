import streamlit as st
import os
import pandas as pd
from PIL import Image
from imageRecog import FaceRecognitionSystem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import GridSearchCV

def main():
    st.set_page_config(page_title= "My Webpage",page_icon=":tada:", layout="wide")
    st.title("""Loan repayment""")

    names= gender =age = sector = occupation = None
    values = []

    st.title("Document Verification")
    print("verify called")
    all_jpegs=False
    # File uploader with the accept parameter to restrict to jpeg files
    uploaded_files = st.file_uploader("Upload your documents", type=["jpeg","jpg"], accept_multiple_files=False)
    name = st.text_input("Enter your name: ")
    
    box_name1=['Male','Female','Others']
    gender=st.radio("Select your gender:",box_name1)
    
    age=st.number_input("Age:")
    income=st.number_input("Income:")

    box_name1=['Own', 'Rent']
    st.write("What would you describe your house ownership as: ")
    house_ownership = st.radio("Select one: ", box_name1)
    
    st.write('Do you have previous borrowed loan amount?')
    box_name1=['Yes','No']
    loan=st.radio("Select one:",box_name1)
    if loan == "Yes":
        borrowedloan=st.number_input("Enter previous loan amount:")
        dueamt=st.number_input("Due Amount of previous loan :")
#    credit_amt=st.number_input('Credit amnt:')
    box_name1 = ["radio/TV", "education", "furniture/equipment", "car", "business", "domestic appliances", "repairs", "vacation/others"]
    st.write('Purpose:')
    purpose=st.radio("Select one:", box_name1)

    loan_amt = st.number_input("Enter the loan amount required: ")
    term=st.number_input('Enter the term of loan (in months):')

    if st.button("SUBMIT"):
        if all([name, gender, age, purpose]) and uploaded_files.type=="image/jpeg":
            file_details = {"FileName":uploaded_files.name,"FileType":uploaded_files.type}
            st.write(file_details)
            with open(os.getcwd()+"\\temp.jpg","wb") as f: 
                if(uploaded_files.type=='image/jpeg'):
                    f.write(uploaded_files.getbuffer())         
                    st.success("Saved File")
                    with open("done.cool", 'w') as fg:
                        pass
            st.success("Form submitted successfully!")
        else:
            st.error("Please enter all the fields.")
    
    if income<5000:
        job = 0
    elif income>5000 and income < 10000:
        job = 1
    elif income>10000 and income < 15000:
        job = 2
    else:
        job = 3

    values={"field":None, "Age":age, "Sex":gender, "Job":job, "Housing":house_ownership, "Saving accounts":"little", "Checking account":"little", "Credit amount":2000, "Duration":term, "Purpose":purpose,"Risk":"good", "loan_amt":loan_amt}
    return values

if __name__=="__main__":
    values = main()
    print(st.session_state)
    if True:
        os.remove("done.cool")
        input_image_path = "C:/Users/Shogun-Knight/Documents/final/temp.jpg"
        encoded_images_dir = "C:/Users/Shogun-Knight/Documents/final/images"
        faceRecogSystem = FaceRecognitionSystem(encoded_images_dir, input_image_path, encoded_images_dir+"/cropped_face.jpg")
        faceRecogSystem.getImages()
        st.success("User verified! Welcome!")

        #implementing decision trees
        #largely inspired by: https://github.com/alicenkbaytop/German-Credit-Risk-Classification
        #on the same dataset
        dataf = pd.read_csv("german_credit_data.csv", index_col=0)
        dataf.head()
        #new_row = pd.DataFrame(values)
        #dataf = pd.concat([dataf, new_row], ignore_index=True)
        #print(dataf)
        dataf.isnull().sum().sort_values(ascending=False)
        dataf.duplicated().sum()
        cols = dataf.columns.to_list()
        for col in cols:
            set_values = dataf[col].unique()
            numeric = pd.api.types.is_numeric_dtype(dataf[col])
            if numeric:
                set_values = np.sort(set_values)
        risk_values={'radio/TV':0.08,'education':0.1,'furniture/equipment':0.28,'car':0.15,'business':0.18,'domestic appliances':0.1,'repairs':0.22,'vacation/others':0.36}
        dataf["Saving accounts"] = dataf["Saving accounts"].fillna("none")
        dataf["Checking account"] = dataf["Checking account"].fillna("none")
        dataf["Purpose"] = dataf["Purpose"].map(risk_values)
        #values.map(risk_values)

        numeric_features = dataf.select_dtypes(include=[int, float]).columns.to_list()
        categorical_features = list(set(dataf.drop("Risk", axis=1).columns) - set(numeric_features))

        dataf["Sex"].value_counts(normalize=True)
        labels = ("student", "young", "adult", "senior")
        groups = pd.cut(dataf["Age"], labels=labels, bins=(18, 24, 30, 60, 120), ordered=True)
        dataf["Age group"] = groups

        dataf["Age_gt_median"] = dataf["Age"].map(lambda x: (x >= dataf["Age"].median()).astype(int))
        dataf["Duration_gt_median"] = dataf["Duration"].map(lambda x: (x >= dataf["Duration"].median()).astype(int))
        dataf["Credit_amount_gt_median"] = dataf["Credit amount"].map(lambda x: (x >= dataf["Credit amount"].median()).astype(int))
        df_pre = dataf.copy()

        def get_outliers(df, feature, iqr_threshold=1.35):
            q1 = np.percentile(df[feature], 25)
            q3 = np.percentile(df[feature], 75)
            iqr = q3 - q1
            return df[(df[feature] < (q1 - iqr_threshold * iqr)) | (df[feature] > (q3 + iqr_threshold * iqr))]

        numeric_features = df_pre.select_dtypes(include=[int, float]).columns.to_list()
        for feature in numeric_features:
            nouts = len(get_outliers(df_pre, feature))

        onehot = OneHotEncoder(drop="first", sparse_output=False)
        onehot_features = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Risk", "Age group"]
        X_onehot = onehot.fit_transform(df_pre[onehot_features])
        df_onehot = pd.DataFrame(data=X_onehot, columns=onehot.get_feature_names_out(onehot.feature_names_in_))

        df_pre = dataf.drop(onehot_features, axis=1)
        df_pre = pd.concat((df_pre, df_onehot), axis=1)
        
        df_pre["Risk_bad"] = (df_pre["Risk_good"] + 1) % 2
        df_pre.drop(["Risk_good"], axis=1, inplace=True)

        to_scale_columns = ["Age", "Credit amount", "Duration"]
        transformer = ColumnTransformer(transformers=[
            ("robust_scaler", RobustScaler(), to_scale_columns)
        ])
        df_pre.loc[:, to_scale_columns] = transformer.fit_transform(df_pre)

        X = df_pre.drop("Risk_bad", axis=1)
        y = df_pre["Risk_bad"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
            
        model = DecisionTreeClassifier(random_state=240)
        scoring = "recall"
        kfold = StratifiedKFold(n_splits=10,  random_state=42,shuffle=True)
        _scores = cross_val_score(model, X_train, y_train, scoring=scoring, cv=kfold, n_jobs=-1)

        msg = "%s has an average score of %.3f ± %.3f" % ("CART", np.mean(_scores), np.std(_scores))
        scores = [_scores]

        scores_df = pd.DataFrame(data=np.array(scores), index=["CART"]).reset_index().rename(columns=dict(index="model"))
        scores_df = pd.melt(scores_df, id_vars=["model"], value_vars=np.arange(0, 10)).rename(columns=dict(variable="fold", value="score"))

        model = DecisionTreeClassifier(random_state=240)
        gridSearch = GridSearchCV(
            model,
            param_grid={
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": np.arange(2, 6),
                "min_samples_split": np.arange(2, 100, 10),
                "max_features": ["sqrt", "log2", None]
            },
            scoring="recall",
            cv=kfold,
            n_jobs=-1
        )
        gridSearch.fit(X_train, y_train)
        model = gridSearch.best_estimator_
        probas = model.predict(X_test)

        if not probas[0]:
            print("You are not eligible for a loan")
        #if(probas[0]==1):
        #alpha = AlphaCalculator(values)
        #print((1+alpha.calculate_alpha())*values["loan_amt"])

