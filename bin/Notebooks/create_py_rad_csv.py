import pandas as pd
import numpy as np
import os

outPath = os.path.join(os.getcwd(), "..", "..", "r")
file_name=outPath+"/data.csv"
save_name_0=outPath+"/py_rad_TUBB.csv"
save_name_1=outPath+"/py_rad_FGF.csv"
save_name_2=outPath+"/py_rad_NAV.csv"
save_name_3=outPath+"/py_rad_MAP.csv"
df=pd.read_csv(file_name)
#im_type=df["Image_TYPE"].values
print(df)

# Display class counts
count1 = df.ClassLABEL.value_counts()
print(count1)
#data=df.iloc[:, 26:]
#data=df.iloc[:,44:]
#data=df.iloc[:,44:66]
#data=df.iloc[:,66:82]
#data=df.iloc[:,82:98]
#data=df.iloc[:,98:]
#df=df.sort_values(by=['Image_TYPE'])
#print(df)
#print(data2)
#df_TUBB = df[df.ImageTYPE==2]
#print(df_TUBB)
# Display class counts
#count2 = df_TUBB.ClassLABEL.value_counts()
#print(count2)
#HC_DMSO = df_TUBB[df_TUBB.ClassLABEL==0]
#print(HC_DMSO)
#HC_CHIR = df_TUBB[df_TUBB.ClassLABEL==1]
#HC_SGK = df_TUBB[df_TUBB.ClassLABEL==3]
#SCZ_DMSO = df_TUBB[df_TUBB.ClassLABEL==5]
#SCZ_CHIR = df_TUBB[df_TUBB.ClassLABEL==6]
#SCZ_SGK = df_TUBB[df_TUBB.ClassLABEL==8]

#df_FGF = df[df.ImageTYPE==0]
#print(df_FGF)
# Display class counts
#count2 = df_FGF.ClassLABEL.value_counts()
#print(count2)
#HC_DMSO = df_FGF[df_FGF.ClassLABEL==0]
#print(HC_DMSO)
#HC_CHIR = df_FGF[df_FGF.ClassLABEL==1]
#HC_SGK = df_TUBB[df_TUBB.ClassLABEL==3]
#SCZ_DMSO = df_FGF[df_FGF.ClassLABEL==5]
#SCZ_CHIR = df_FGF[df_FGF.ClassLABEL==6]
#SCZ_SGK = df_TUBB[df_TUBB.ClassLABEL==8]

#df_NAV = df[df.ImageTYPE==1]
#print(df_NAV)
# Display class counts
#count2 = df_NAV.ClassLABEL.value_counts()
#print(count2)
#HC_DMSO = df_NAV[df_NAV.ClassLABEL==0]
#print(HC_DMSO)
#HC_CHIR = df_NAV[df_NAV.ClassLABEL==1]
#HC_SGK = df_TUBB[df_TUBB.ClassLABEL==3]
#SCZ_DMSO = df_NAV[df_NAV.ClassLABEL==5]
#SCZ_CHIR = df_NAV[df_NAV.ClassLABEL==6]
#SCZ_SGK = df_TUBB[df_TUBB.ClassLABEL==8]

df_MAP = df[df.ImageTYPE==3]
print(df_MAP)
# Display class counts
count2 = df_MAP.ClassLABEL.value_counts()
print(count2)
HC_DMSO = df_MAP[df_MAP.ClassLABEL==0]
#print(HC_DMSO)
HC_CHIR = df_MAP[df_MAP.ClassLABEL==1]
#HC_SGK = df_TUBB[df_TUBB.ClassLABEL==3]
SCZ_DMSO = df_MAP[df_MAP.ClassLABEL==5]
SCZ_CHIR = df_MAP[df_MAP.ClassLABEL==6]
#SCZ_SGK = df_TUBB[df_TUBB.ClassLABEL==8]




# Combine classes
data = pd.concat([HC_DMSO, HC_CHIR, SCZ_DMSO, SCZ_CHIR])
print(data)
count3 = data.ClassLABEL.value_counts()
print(count3)
data.to_csv(save_name_3)
