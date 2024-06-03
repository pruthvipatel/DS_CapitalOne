# %%
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import statsmodels as sm
import scipy
from scipy.stats import fisher_exact
from pycaret.classification import *

# %%
# def convert_line_to_dict(line):
#     return json.loads(line)

# data = []
# with open('../transactions.txt', 'r') as f:
#     for line in f:
#         data.append(convert_line_to_dict(line))

# df = pd.DataFrame(data)
# df.to_parquet('transactions.parquet')

# %%
transactions_df = pd.read_parquet('transactions.parquet')

# %%
transactions_df.head(2).T

# %%
transactions_df.shape

# %%
transactions_df.dtypes

# %%
datetime_cols = ['transactionDateTime','accountOpenDate','dateOfLastAddressChange','currentExpDate']

transactions_df[datetime_cols] = transactions_df[datetime_cols].apply(lambda x: pd.to_datetime(x))

# %%
transactions_df.sort_values(by=['transactionDateTime'], inplace=True)

# %%
def create_histplot(transactions_df, column_name, bins = 30,log_transform = False, convert_counts = False):

  if log_transform:
    plotting_variable = np.log10(transactions_df[column_name])
  else:
    plotting_variable = transactions_df[column_name]

  if convert_counts:
    plotting_variable = plotting_variable.value_counts()



  sns.histplot(plotting_variable, bins=bins, color='grey')


  min_val = plotting_variable.min()
  max_val = plotting_variable.max()
  mean_val = plotting_variable.mean()
  median_val = plotting_variable.median()

  if log_transform:
    print(f'Actual Min, Mean, Median and Max values are : {10**min_val:.3f},{10**mean_val:.3f},{10**median_val:.3f}, and {10**max_val:.3f}')
  else:
    print(f'Min, Mean, Median and Max values are : {min_val:.3f},{mean_val:.3f},{median_val:.3f}, and {max_val:.3f}')


  plt.axvline(x=min_val, color='red', linestyle='dashed', linewidth=2)
  plt.axvline(x=mean_val, color='green', linestyle='dashed', linewidth=2)
  plt.axvline(x=median_val, color='blue', linestyle='dashed', linewidth=2)
  plt.axvline(x=max_val, color='purple', linestyle='dashed', linewidth=2)


  plt.title(f'{column_name} Histogram')
  plt.xlabel(f'{column_name}')
  plt.ylabel('Frequency')


  plt.show()


# %%
create_histplot(transactions_df, 'currentBalance', bins = 10, log_transform=False, convert_counts=True)

# %%
create_histplot(transactions_df, 'transactionAmount', bins = 100)

# %% [markdown]
# ### REVERSED TRANSACTION

# %% [markdown]
# Reversed Transaction in this case is identified by checking for the following. For a given:
# 1. Account Number
# 2. Merchant Name
# 3. Transaction Amount
# 
# If there is a transaction type 'PURCHASE', followed by Transaction Type 'REVERSAL'. Then, the 'PURCHASE' transaction is marked as is_reversed = True
# 
# 

# %%

transactions_df['shifted_transactionType'] = transactions_df.groupby(['accountNumber','merchantName','transactionAmount']).transactionType.shift(-1)
transactions_df['is_reversed'] = False
transactions_df.loc[(transactions_df.transactionType == 'PURCHASE') & (transactions_df.shifted_transactionType == 'REVERSAL'),'is_reversed'] = True

# %% [markdown]
# ### MULTIPLE TRANSACTIONS

# %%
transactions_df['diff_transactionDateTime'] = transactions_df.groupby(['accountNumber','merchantName','transactionAmount']).transactionDateTime.diff()
transactions_df['is_multiple'] = False

# %%
transactions_df.loc[transactions_df.diff_transactionDateTime < timedelta(minutes=10),'is_multiple'] = True

# %% [markdown]
# ### VALIDATION OF REVERSED TRANSACTIONS
# 
# For transactions having 1 time unit shifted Transaction Type = 'REVERSAL', there is assumed to have existed a 'PURCHASE' transactionType. On verifying, this is not the case.
# There are 'ADDRESS_VERIFICATION' transactions and blank transactions having a following 'REVERSAL'.
# 
# Since, these are not reversed transactions in true sense, these are not marked as is_reversed
# 
# Additionally, the transactionAmount for the shifted transactionType as 'REVERSAL' but not is_reversed have a very low mean transactionAmount and majority of the transactions are worth $0. Hence, these are ignored

# %%
transactions_df[~transactions_df.is_reversed & (transactions_df.shifted_transactionType == 'REVERSAL')].transactionType.value_counts()

# %%
transactions_df.transactionType.value_counts()

# %%
transactions_df.is_reversed.value_counts()

# %%
transactions_df[~transactions_df.is_reversed & (transactions_df.shifted_transactionType == 'REVERSAL')].transactionAmount.plot.hist(bins=30)

# %% [markdown]
# ### TOTAL DOLLAR AMOUNT OF REVERSED TRANSACTIONS

# %%
total_reversedTransactionAmount = transactions_df[transactions_df.is_reversed].transactionAmount.sum()

# %%
total_reversedTransactionAmount

# %%
total_multipleTransctionAmount = transactions_df[transactions_df.is_multiple].transactionAmount.sum()

# %%
total_multipleTransctionAmount

# %%
transactions_df.groupby(['is_reversed','is_multiple']).size()

# %%
transactions_df.columns

# %% [markdown]
# ## EDA
# 
# To explore:
# 
# 1. Merhcant Time and Space distribution
#    - acqCountry
#    - merchantCountryCode
#    - merchantCity
#    - merchantState
#    - merchantZip
# 2. Time of the year for these transactions
# 3. Do they have a pattern of specific merchant info
#    - PosOnPremises
#    - enteredCVV
#    - exirationDateKeyInMatch
#    - posEntryMode
#    - posConditionCode
#    - merchantCategoryCode
# 4. Account activity information
#    - dateOfLastAddressChange
#    - recurringAuthInd?
#    - currentExpDate
#    - accountOpenDate
# 5. Correlation between
#    - cardCVV and enteredCVV match and isFraud
#    - expirationDateKeyInMatch and isFraud
#    - posOnPremise to isFraud
# 
# 
# 

# %%
is_reversed_flag = transactions_df.is_reversed == True
is_multiple_flag = transactions_df.is_multiple == True

# %% [markdown]
# ### EDA on Fraud and Multiple Transactions
# 
# 1. How correlated are reversed and multiplte transactions
# 2. How is the merchant distribution of reversed and multiple transactions
# 3. How much is the credit limit for both the transactions

# %%

reversed_multiple_transactions_crosstab = pd.crosstab(transactions_df.is_reversed, transactions_df.is_multiple)


# %%
sns.heatmap(reversed_multiple_transactions_crosstab, annot=True, fmt='g')
plt.title('Reversed vs Multiple Transactions Crosstab')
plt.xlabel('is_multiple')
plt.ylabel('is_reversed')
plt.show()


# %%

table = sm.stats.contingency_tables.Table(reversed_multiple_transactions_crosstab) # might throw module error. In case of error, try using import statsmodels, then sm.stats.contingency_tables.Table(...)
result = table.test_nominal_association()
print(f'Test gives Chi-Square statistic = {result.statistic} and p-value = {result.pvalue:.3f} .\n Ar Alpha = 0.05, the result is {"significant" if result.pvalue < 0.05 else "not significant"}')

# %% [markdown]
# Chi Square Test of independence gives a very low p-value, meaning that the reversed and mulitple transactions are correlated. However, given the imbalanced nature of the categorical features, I performed a Fisher Exact Test to see the odds ratio of the cases where both the categories are aligned vs when they are not aligned

# %%
transactions_df.transactionType.value_counts()

# %%
# Fisher's Exact Test

odds_ratio, p_value = fisher_exact(reversed_multiple_transactions_crosstab)

print('Odds Ratio:', odds_ratio)
print('p-value:', p_value)

# %% [markdown]
# Odds ratio indicates that the reversed and multiple transactions are more likely to have different values (True-False, False-True) vs same values (True-True, False-False)
# with this result being 1 (confirm and contract equally) with a very low probability.
# 
# Meaning both of these types of unusual transactions are less likely to occur simultaneously

# %% [markdown]
# ### Merchant Distribution of Reversed Transactions, Multiple Transactions

# %%
def bar_plot(filtering_variable, target_variable, count_threshold = 100):
  target_variable_distribution = transactions_df[transactions_df[filtering_variable]][target_variable].value_counts()
  target_variable_distribution[target_variable_distribution > count_threshold].plot.bar()
  plt.ylabel(f'Count of {filtering_variable}')
  plt.show()

# %%
bar_plot('is_reversed','merchantName')

# %%
bar_plot('is_reversed','merchantCategoryCode')

# %%
bar_plot('is_multiple','merchantName')

# %%
bar_plot('is_multiple','merchantCategoryCode')

# %% [markdown]
# The 2 plots show that most of the top drivers of multiple and reversed transaction have very common list of merchant names, notably the ride sharing, ecommerce, ecards and movie theaters
# 

# %%
create_histplot(transactions_df[transactions_df.is_reversed], 'transactionAmount', bins = 100)

# %%
create_histplot(transactions_df[transactions_df.is_multiple], 'transactionAmount', bins = 100)

# %%
create_histplot(transactions_df, 'transactionAmount', bins = 100)

# %% [markdown]
# ## FRAUD DETECTION

# %%
transactions_df.dtypes

# %%
transactions_df.shape

# %%
transactions_df.apply(lambda x: (x == '').mean())

# %% [markdown]
# In this data, following are the columns grouped by their column types:
# 
# Indicator Columns
# 1. accountNumber
# 2. customerId
# 
# 
# Numeric Columns
# 1. creditLimit
# 2. availableMoney
# 3. transactionAmount
# 4. currentBalance
# 
# 
# Datetime Columns
# 1. transactionDateTime
# 2. currentExpDate
# 3. accountOpenDate
# 4. dateOfLastAddressChange
# 
# Categorical Columns
# 1. merchantName
# 2. acqCountry
# 3. merchantCategoryCode
# 4. merchantCountryCode
# 5. transactionType
# 6. merchantCity
# 7. merchantState
# 8. merchantZip
# 9. cardPresent
# 10. posOnPremises
# 11. ExpirationDateKeyInMatch
# 12. is_reversed
# 13. is_multiple
# 
# String Columns
# 1. cardCVV
# 2. enteredCVV
# 3. cardLast4Digits
# 4. echoBuffer?
# 5. recurringAuthInd
# 
# 
# Target Column
# 1. isFraud
# 
# Derived Columns
# 1. is_reversed (Categorical)
# 2. is_multiple (Categorical)
# 3. shifted_transactionType (Categorical)
# 4. diff_transactionDateTime (timeDelta)
# 
# 
# Some of the columns are completely blank. This is missing information which is not useful to predict fraudulent transactions and hence, removed from further analysis

# %%
transactions_df_cleaned = transactions_df.copy()

# %%
blank_columns = transactions_df.columns[transactions_df.apply(lambda x: (x == '').mean()) == 1]
transactions_df_cleaned.drop(columns = blank_columns, inplace = True)

# %%
numeric_columns = transactions_df_cleaned.select_dtypes(include=['float64']).columns.tolist()
categorical_columns = ['merchantName','acqCountry','merchantCategoryCode','merchantCountryCode',\
'transactionType','merchantCity','merchantState','merchantZip','cardPresent','posOnPremises',\
'expirationDateKeyInMatch','is_reversed','is_multiple']

categorical_columns = [x for x in categorical_columns if x not in blank_columns]


# %% [markdown]
# ## Correlation with Target

# %%

def get_point_biserial_correlation(df, numeric_column, categorical_target):


    point_biserial_target = df[categorical_target].apply(lambda x: 1 if x else 0)


    correlation, p_value = scipy.stats.pointbiserialr(df[numeric_column], point_biserial_target)

    print(f'{numeric_column} vs {categorical_target} -- POINT BISERIAL CORRELATION = {correlation:.3f}, p-value = {p_value:.3f}')



# %%
for i in numeric_columns:
  get_point_biserial_correlation(transactions_df_cleaned, i, 'isFraud')

# %%
def cramers_v(df, col, categorical_target):


    confusion_matrix = pd.crosstab(df[col], df[categorical_target])
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    try:
      print(f"{col} CRAMER'S V CORRELATION {np.sqrt(phi2corr / min(k-1, r-1)):.3f}")
    except:
      print(f"{col} CRAMER'S V CORRELATION NOT COMPUTABLE")




# %%
for i in categorical_columns:
  cramers_v(transactions_df_cleaned, i,'isFraud')

# %% [markdown]
# Since the correlation of any single variable (categorical or numeric) is not practically large enough to predict the isFraud target, the classification model needs to be a combination of the features and possibly interaction of features
# 
# Low correlation values could also be due to the large imbalance in the data (98.4% no vs 1.6% yes), this data needs to be balanced before creating a classification model

# %%
transactions_df.columns

# %%
transactions_df_cleaned.set_index('accountNumber', inplace=True)
transactions_df_cleaned.drop(['customerId'], axis=1, inplace=True)

# %%


# %%
exp_clf101 = setup(data = transactions_df_cleaned, target = 'isFraud', session_id=123)

# %%



