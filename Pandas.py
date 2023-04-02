#Pandas Exercises
##################################################

#let's import the libraries
import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Task 1:  Identify the Titanic dataset from the Seaborn library.
#########################################
df = sns.load_dataset("titanic")
df.shape
df.head()

#########################################
# Task 2:  Find the number of male and female passengers in the Titanic dataset described above.

df["sex"].unique()
df["sex"].value_counts()  #we use value_counts() to find out how many there are

#########################################
# Task 3:  Find the number of unique values for each column.

df["sex"].nunique()
df.nunique()

#########################################
# Task 4:  Find the unique values of the variable pclass.
df["pclass"].unique()

#########################################
# Task 5:  Find the number of unique values of pclass and parch variables.
df[["pclass","parch"]].nunique()

#########################################
#Task 6:  Check the type of the embarked variable. Change its type to category. Check the repetition type.

str(df["embarked"].dtype)   # dtype is object

df["embarked"] = df["embarked"].astype("category")  # we use astype() to change the type

#########################################
# Task 7:  Show all the sages of those with embarked value C.
df[df["embarked"]=="C"].head()

df.loc[df["embarked"]=="C",df.columns]

#other alternatives
df[df["embarked"].astype("str").str.contains("C")]
df.query("embarked=='C'")

#########################################
# Task 8:  Show all the sages of those whose embarked value is not S.

df[df["embarked"]!="S"].head()
df[~(df["embarked"]=="S")].head()
df.loc[df["embarked"]!="S",df.columns]

#########################################
# Task 9:  Show all the information for female passengers younger than 30 years old.
df[(df["age"]<30) & (df["sex"]=="female")].head()

#########################################
# Task 10:  Show the information of passengers whose Fare is over 500 or 70 years old.
df[(df["fare"]>500) |  (df["age"]> 70 )].head()

#########################################
# Task 11:  Find the sum of the null values in each variable.
df.isnull().sum()

#########################################
# Task 12:  drop the who variable from the dataframe.
df.drop("who", axis=1,inplace=True)  #inplace=True if we do this, it deletes it permanently
df.head()

#########################################
# Task 13:  Fill the empty values in the deck variable with the most repeated value (mode) of the deck variable.

df["deck"].mode()[0]

df["deck"].value_counts()

df["deck"] = df["deck"].fillna(df["deck"].mode()[0])
df["deck"].isnull().sum()

#########################################
# Task 14:  Fill the empty values in the age variable with the median of the age variable.
df["age"].fillna(df["age"].median(), inplace=True)
df["age"].isnull().sum()

#########################################
# Task 15:  Find the sum, count, mean values of the Pclass and Gender variables of the survived variable.
#groupby a alıcaz
df.groupby(["pclass","sex"]).agg({"survived": ["sum","count","mean"]})

#########################################
# Task 16:  Write a function that returns 1 for those under 30, 0 for those above or equal to 30.
# Create a variable named age_flag in the titanic data set using the function you wrote. (use apply and lambda constructs)

# create a function
def age_30(age):
    if age < 30:
        return 1
    else:
        return 0

df["age_flag"] = df["age"].apply(lambda x: age_30(x))
df.head()

df["age_flag_2"] = df["age"].apply(lambda x:  1 if x<30 else 0)
df.head()

#########################################
# Task 17:  Define the Tips dataset from the Seaborn library.
df = sns.load_dataset("tips")
df.shape
df.head()

#########################################
# Task 18:  Find the sum, min, max and average of the total_bill values according to the categories (Dinner, Lunch) of the Time variable.
df.groupby("time").agg({"total_bill": ["sum","min","max","mean"]})

#########################################
# Task 19:  Find the sum, min, max and average of total_bill values by days and time.
df.groupby(["time","day"]).agg({"total_bill": ["sum","min","max","mean"]})


#########################################
# Task 20:  Find the sum, min, max and average of the total_bill and type values of the female customers, according to the day, for the lunch time.
df[(df["time"]=="Lunch") & (df["sex"]=="Female")].groupby("day").agg({ "total_bill": ["sum","min","max","mean"],
                                                            "tip": ["sum","min","max","mean"]})


df[(df["time"]=="Lunch") & (df["sex"]=="Female")].groupby("day").agg(["min","max","mean"])[["total_bill","tip"]]

#########################################
# Task 21:  What is the average of orders with size less than 3 and total_bill greater than 10?

df[(df["size"]<3) & (df["total_bill"]>10)]["total_bill"].mean()
df.loc[(df["size"]<3) & (df["total_bill"]>10),"total_bill"].mean()

#########################################
# Task 22:  Create a new variable called total_bill_tip_sum. Let him give the sum of the total bill and tip paid by each customer.
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()

df["total_bill_tip_sum_2"] = [row[0] + row[1] for row in df.values]
df.head()

df["total_bill_tip_sum_3"] = [str(row[0]) + "_" + str(row[1]) for row in df.values]
df.head()

#########################################
# Task 23: Find the average of the total_bill variable for men and women separately.
# Create a new total_bill_flag variable in which those below the averages you found are given 0, those above and equal to one.
# Attention!! For those who are female, the average found for women will be taken into account, while for male, the average found for men.
# start by typing a function that takes gender and total_bill as parameters. (If-else conditions will be included)

# the average of the total bill for men and women
f_avg = df[df["sex"] == "Female"]["total_bill"].mean()
m_avg = df[df["sex"] == "Male"]["total_bill"].mean()

def func(sex,total_bill):
    if sex== "Female":
        if total_bill < f_avg:
            return 0
        else:
            return 1
    else:
        if total_bill < m_avg:
            return 0
        else:
            return 1

df["total_bill_flag"] = df[["sex","total_bill"]].apply(lambda x : func(x["sex"],x["total_bill"]),axis=1)
df.head()

#########################################
# Task 24: Using the total_bill_flag variable, observe the number of those who are below and above the average by gender.

df["total_bill_flag"].value_counts()
df.groupby(["sex","total_bill_flag"]).agg({"total_bill_flag": "count"})

#########################################
# Görev 25: Sort the total_bill_tip_sum variable from largest to smallest and assign the first 30 people to a new dataframe.
new_df = df.sort_values("total_bill_tip_sum",ascending=False)[:30]  #sacending=False büyükten küçüğe sıralar
new_df.head()
new_df.head()
new_df.shape
