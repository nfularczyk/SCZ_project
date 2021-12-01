from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.utils import resample
import csv
import os

#Based on https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py

def concate_features(start_1, end_1, start_2, end_2, dframe):
    d_1 = dframe.iloc[:,start_1:end_1]
    #print(d_1)
    d_2 = dframe.iloc[:,start_2:end_2]
    #print(d_2)
    data = pd.concat([d_1, d_2], axis=1)
    #print(data)
    return data
    
def write_to_csv(save_filename, iters, results):
    with open(save_filename, 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(["Classifier", "Mean Accuracy over "+str(iters)+" iterations", "Standard Deviation"])
        for row in results:
            writer.writerow(row)
    #writer.writerow([""])
    #for row in results[0]:
        #writer.writerow(row)
 
#inPath = os.path.join(os.getcwd(), "..", "..", "r")
#outPath = inPath+"/radiomics"
#savePath = inPath+"/concat_perm_r417"
#savePath = inPath + "/temp"
#old_file=outPath+"/Mix_radiomics_features.csv"
#new_file=outPath+"/TUBB_radiomics_new.csv"
#file_name = outPath+"/TUBB_DS_radiomics.csv"
#HC_save_name = outPath+"/HC_TUBB_results.csv"
#SCZ_save_name = outPath+"/SCZ_TUBB_results.csv"
#file_name_1 = outPath+"/TUBB_radiomics_new.csv"
file_name_1 = os.path.join(os.getcwd(),"..","..","..","","r","radiomics","","TUBB_radiomics_new.csv")
#HC_save_name = outPath+"/TUBB_DMSOresults.csv"
#SCZ_save_name = outPath+"/TUBB_CHIRresults.csv"
#file_name = outPath+"/MAP_radiomics.csv"
#HC_save_name = outPath+"/MAP_HCresults.csv"
#SCZ_save_name = outPath+"/MAP_SCZresults.csv"
#DMSO_save_name = outPath+"/MAP_DMSOresults.csv"
#CHIR_save_name = outPath+"/MAP_CHIRresults.csv"
#file_name_2 = outPath+"/FGF_radiomics.csv"
file_name_2 = os.path.join(os.getcwd(),"..","..","..","","r","radiomics","","FGF_radiomics.csv")
#HC_save_name = outPath+"/FGF_HCresults.csv"
#SCZ_save_name = outPath+"/FGF_SCZresults.csv"
#DMSO_save_name = outPath+"/FGF_DMSOresults.csv"
#CHIR_save_name = outPath+"/FGF_CHIRresults.csv"
#file_name_Nav = outPath+"/NAV_radiomics.csv"
#HC_save_name = savePath+"/TUBB_FGF_HC.csv"
#SCZ_save_name = savePath+"/TUBB_FGF_SCZ.csv"
#DMSO_save_name = savePath+"/TUBB_FGF_DMSO.csv"
#CHIR_save_name = savePath+"/TUBB_FGF_CHIR.csv"
HC_save_name = os.path.join(os.getcwd(),"..","..","..","","r","","concat_r417","","TUBB_FGF_HC.csv")
SCZ_save_name = os.path.join(os.getcwd(),"..","..","..","","r","","concat_r417","","TUBB_FGF_SCZ.csv")
DMSO_save_name = os.path.join(os.getcwd(),"..","..","..","","r","","concat_r417","","TUBB_FGF_DMSO.csv")
CHIR_save_name = os.path.join(os.getcwd(),"..","..","..","","r","","concat_r417","","TUBB_FGF_CHIR.csv")
#old_data=pd.read_csv(old_file)
#new_data=pd.read_csv(new_file)
df_1 = pd.read_csv(file_name_1)
df_2 = pd.read_csv(file_name_2)
df_2 = df_2.drop(['LABEL'], axis=1)
#df_2 = df_2.copy()
#print(df_TUBB.LABEL)
#print(df_FGF.columns)
#count1 = old_data.LABEL.value_counts()
#count2 = new_data.LABEL.value_counts()
#print(new_data["LABEL"].values)
#new_data["LABEL"].replace([5,6],[2,3],inplace=True)
#count = df_TUBB.LABEL.value_counts()
#print(count)

df = pd.concat([df_1, df_2], axis=1)
#df = df.copy()
#print(df)
df_ids= df_1.ID
to_remove=[]
for index, value in df_ids.items():
    #find which set the data point belongs to
    temp_s=value.split('-')[0]
    #find the datapoint number
    temp_n=value.split()[2]


    if (temp_s == "Set 2" and temp_n == "417"):
    	#print(index)
    	#print(temp_s)
    	#print(temp_n)
    	#print(value)
    	to_remove.append(index)
#print(to_remove)
#print(len(to_remove))
df = df.drop(to_remove)
#print(df)
df = df.reset_index(drop=True)
#print(df)

start_1= 98
end_1 = start_1+14
start_2 = end_1+97
end_2 = start_2+14

#print(df)
#print(df.LABEL)
#print(new_data["LABEL"].values)
#df = pd.concat([old_data, new_data])
#count3 = df.LABEL.value_counts()
#print(count1)
#print(count2)
#print(count3)
HC_df = df[(df.LABEL==0) | (df.LABEL==1)]
SCZ_df = df[(df.LABEL==5) | (df.LABEL==6)]
DMSO_df = df[(df.LABEL==0) | (df.LABEL==5)]
CHIR_df = df[(df.LABEL==1) | (df.LABEL==6)]

HC_count = HC_df.LABEL.value_counts()
SCZ_count = SCZ_df.LABEL.value_counts()
DMSO_count = DMSO_df.LABEL.value_counts()
CHIR_count = CHIR_df.LABEL.value_counts()

print(HC_count)
print(SCZ_count)
print(DMSO_count)
print(CHIR_count)
#HC_df = HC_df.copy()
#DMSO_df = DMSO_df.copy()
#CHIR_df = CHIR_df.copy()
#SCZ_df = SCZ_df.copy()
#HC_df = df[(df.LABEL==0) | (df.LABEL==3)]
#SCZ_df = df[(df.LABEL==5) | (df.LABEL==6)]
#SCZ_df = df[(df.LABEL==5) | (df.LABEL==8)]
HC_labels=HC_df["LABEL"].values
SCZ_labels=SCZ_df["LABEL"].replace([5,6],[0,1]).values
DMSO_labels=DMSO_df["LABEL"].replace([0,5],[0,1]).values
CHIR_labels=CHIR_df["LABEL"].replace([1,6],[0,1]).values

HC_data=concate_features(start_1, end_1, start_2, end_2, HC_df)
SCZ_data=concate_features(start_1, end_1, start_2, end_2, SCZ_df)
DMSO_data=concate_features(start_1, end_1, start_2, end_2, DMSO_df)
CHIR_data=concate_features(start_1, end_1, start_2, end_2, CHIR_df)
#print(HC_data)
#print(SCZ_data)
#print(DMSO_data)
#print(CHIR_data)


HC_data=HC_data.to_numpy()
SCZ_data=SCZ_data.to_numpy()
DMSO_data = DMSO_data.to_numpy()
CHIR_data = CHIR_data.to_numpy()
C=1
tol=1e-4

datasets = [[HC_data, HC_labels], [SCZ_data, SCZ_labels], [DMSO_data, DMSO_labels], [CHIR_data, CHIR_labels]]
classifiers= [
    LogisticRegression(C=C, penalty='l1', tol=tol, class_weight='balanced', solver='saga', max_iter=10000),
    LogisticRegression(C=C, penalty='l2', tol=tol, class_weight='balanced', solver='lbfgs', max_iter=10000),
    SVC(kernel='linear',class_weight='balanced'),
    LogisticRegression(C=C, penalty='l1', tol=tol, solver='saga', max_iter=10000),
    LogisticRegression(C=C, penalty='l2', tol=tol, solver='lbfgs', max_iter=10000),
    SVC(kernel='linear')]
names = ["Balanced L1 Regression", "Balanced L2 Regression", "Balanced Linear SVM", "L1 Regression",
         "L2 Regression", "Linear SVM"]

HC_results = [["Iteration", "Classifier","Accuracy on Test Set", "Confusion Matrix"]]
SCZ_results = [["Iteration", "Classifier","Accuracy on Test Set", "Confusion Matrix"]]
DMSO_results = [["Iteration", "Classifier","Accuracy on Test Set", "Confusion Matrix"]]
CHIR_results = [["Iteration", "Classifier","Accuracy on Test Set", "Confusion Matrix"]]
results=[HC_results, SCZ_results, DMSO_results, CHIR_results]
num_iter = 100

for i in range(num_iter):
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        #print(ds_cnt)
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
        #print('train -  {}   |   test -  {}'.format(np.bincount(y_train), np.bincount(y_test)))
        #print(len(y_train))
        #print(len(y_test))

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            y_pred=clf.predict(X_test)
            conf = confusion_matrix(y_test, y_pred)
            #print(name)
            #print(score)
            #print(conf)
            results[ds_cnt].append([str(i), name, score, conf])

results[0].append("")
results[1].append("")
results[2].append("")
results[3].append("")
HC_final_results=[]
SCZ_final_results=[]
DMSO_final_results=[]
CHIR_final_results=[]
final_results=[HC_final_results, SCZ_final_results, DMSO_final_results, CHIR_final_results]
for i in range(len(classifiers)):
    j=i+1
    entries=[j]
    HC_acc_i=[]
    SCZ_acc_i=[]
    DMSO_acc_i=[]
    CHIR_acc_i=[]
    for k in range(num_iter-1):
        j=j+len(classifiers)
        entries.append(j)
    #print(entries)
    #print(len(results[0]))
    #print(results[0])
    for l in entries:
        #print(l)
        #print(results[0][l][2])
        #print(results[0][l])
        HC_acc_i.append(results[0][l][2])
        SCZ_acc_i.append(results[1][l][2])
        DMSO_acc_i.append(results[2][l][2])
        CHIR_acc_i.append(results[3][l][2])
    final_results[0].append([names[i], np.mean(HC_acc_i), np.std(HC_acc_i)])
    final_results[1].append([names[i], np.mean(SCZ_acc_i), np.std(SCZ_acc_i)])
    final_results[2].append([names[i], np.mean(DMSO_acc_i), np.std(DMSO_acc_i)])
    final_results[3].append([names[i], np.mean(CHIR_acc_i), np.std(CHIR_acc_i)])
                        
write_to_csv(HC_save_name, num_iter, final_results[0])
write_to_csv(SCZ_save_name, num_iter, final_results[1])
write_to_csv(DMSO_save_name, num_iter, final_results[2])
write_to_csv(CHIR_save_name, num_iter, final_results[3])
