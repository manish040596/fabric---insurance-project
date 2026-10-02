#!/usr/bin/env python
# coding: utf-8

# ## Notebook 1
# 
# New notebook

# In[1]:


# Welcome to your new notebook
# Type here in the cell editor to add code!
from pyspark.sql.functions import *

# ===============================
# 1. LOAD RAW DATA (Bronze Layer)
# ===============================
df_customers = spark.read.parquet('abfss://insurance@onelake.dfs.fabric.microsoft.com/insurance_lakehouse.Lakehouse/Files/Bronze/customer/customers.parquet')
df_policies  = spark.read.parquet('abfss://insurance@onelake.dfs.fabric.microsoft.com/insurance_lakehouse.Lakehouse/Files/Bronze/policy/policies.parquet')
df_claims    = spark.read.parquet('abfss://insurance@onelake.dfs.fabric.microsoft.com/insurance_lakehouse.Lakehouse/Files/Bronze/claim/claims.parquet')


# In[2]:


display(df_customers.limit(10))


# In[3]:


display(df_policies.limit(10))


# In[4]:


display(df_claims.limit(10))


# # Cleaning the customer data

# In[5]:


display(df_customers.limit(5))


# In[6]:


df_customers_clean = (
    df_customers
    .dropna(subset=["cust_id"])
    .withColumn("cust_id", col("cust_id").cast("int"))
    .withColumn("CustomerName", initcap(trim(col("Full Name"))))
    .withColumn("Date_of_Birth", to_date(col("Date_of_Birth")))
    .withColumn("Contact", when(col("Contact").isNull(), "Not Provided").otherwise(col("Contact")))
    .drop("Full Name")
    .dropDuplicates()
)
display(df_customers_clean)


# # Cleaning the policy data

# In[7]:


display(df_policies.limit(10))


# In[13]:


df_policies_clean = (
    df_policies
    .withColumn("policy_Type", initcap(trim(col("policy_Type"))))
    .withColumn("status", initcap(trim(col("status"))))
    .withColumn("Start_Date", to_date(col("Start_Date"), "dd/MM/yyyy"))
    .withColumn("Coverage_Amount", when(col("Coverage_Amount") == "not available", None)
                .otherwise(col("Coverage_Amount").cast("double")))
    .withColumn("cust_id", col("cust_id").cast("int"))
    .dropna(subset=["cust_id", "policy_id"])
)
display(df_policies_clean)


# # Cleaning the Claim data

# In[15]:


display(df_claims.limit(5))


# In[16]:


df_claims_clean = (
    df_claims
    .withColumn("claim_date", to_date(col("claim_date")))
    .withColumn("claim_amount", col("claim_amount").cast("double"))
    .withColumn("status", initcap(trim(col("status"))))
    .dropna(subset=["policy_id"])
)
display(df_claims_clean)


# # create delta silver table for this

# In[27]:


df_customers_clean.write.mode("overwrite").format("delta").saveAsTable("silver_customers")
df_policies_clean.write.mode("overwrite").format("delta").saveAsTable("silver_policies")
df_claims_clean.write.mode("overwrite").format("delta").saveAsTable("silver_claims")


# In[30]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %%sql 
# select * from insurance_lakehouse.silver_customers


# # gold tables 

# In[31]:


df_joined = (
    df_claims_clean.alias("c")
    .join(df_policies_clean.alias("p"), col("c.policy_id") == col("p.policy_id"), "inner")
    .join(df_customers_clean.alias("cu"), col("p.cust_id") == col("cu.cust_id"), "inner")
)
display(df_joined.limit(5))


# In[33]:


df_aggregated = (
    df_joined.groupBy("cu.cust_id", "cu.CustomerName", "cu.Gender")
    .agg(
        count("c.claim_id").alias("TotalClaims"),
        count(when(lower(col("c.status")) == "approved", True)).alias("ApprovedClaims"),
        count(when(lower(col("c.status")) == "rejected", True)).alias("RejectedClaims"),
        round(sum("c.claim_amount"), 2).alias("TotalClaimAmount"),
        round(avg("c.claim_amount"), 2).alias("AvgClaimAmount"),
        collect_set(lower(col("p.policy_type"))).alias("PolicyTypesArray"),
        round(sum("p.coverage_amount"), 2).alias("TotalCoverageAmount"),
        min("c.claim_date").alias("FirstClaimDate"),
        max("c.claim_date").alias("LastClaimDate")
    )
    .withColumn("PolicyTypes", concat_ws(", ", col("PolicyTypesArray")))
    .withColumn(
        "ClaimToCoverageRatio",
        round((col("TotalClaimAmount") / col("TotalCoverageAmount")) * 100, 2)
    )
    .drop("PolicyTypesArray")
)
display(df_aggregated)

# SAVE the cleaned+enriched silver output
df_aggregated.write.mode("overwrite").format("delta").save("abfss://insurance@onelake.dfs.fabric.microsoft.com/insurance_lakehouse.Lakehouse/Files/gold")


# In[34]:


df_aggregated.write.mode("overwrite").format("delta").saveAsTable("insurance_aggrgate")


# In[35]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %%sql 
# select * from insurance_lakehouse.insurance_aggrgate


# In[ ]:





# # Top 10 High-Value Customers by Claim Amount

# In[36]:


display(spark.sql("""
SELECT CustomerName, TotalClaimAmount
FROM  insurance_lakehouse.insurance_aggrgate
ORDER BY TotalClaimAmount DESC
LIMIT 10
"""))


# # Claim Approval Rate by Gender

# In[37]:


display(spark.sql("""
SELECT Gender,
       ApprovedClaims,
       RejectedClaims,
       ROUND(ApprovedClaims * 100.0 / TotalClaims, 2) AS ApprovalRate
FROM insurance_lakehouse.insurance_aggrgate
"""))


#  # 3. Claim to Coverage Ratio Distribution

# In[38]:


display(spark.sql("""
SELECT CustomerName, ClaimToCoverageRatio
FROM insurance_lakehouse.insurance_aggrgate
WHERE ClaimToCoverageRatio IS NOT NULL
ORDER BY ClaimToCoverageRatio DESC
"""))


# # Claims by Policy Types (Explode for Analysis)

# In[39]:


from pyspark.sql.functions import explode, split

df_policy_split = (
    df_aggregated
    .withColumn("PolicyType", explode(split(col("PolicyTypes"), ",\\s*")))
    .groupBy("PolicyType")
    .agg(count("*").alias("CustomerCount"))
)

display(df_policy_split)


# In[ ]:




