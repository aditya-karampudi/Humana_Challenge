# Humana-Challenge

### 1. Executive Summary 

This analysis focused on providing Humana a better understanding of transportation as a social determinant of health. Our goal was to develop a classification model to identify Medicare members most at the risk for a Transportation Challenge and propose solutions for them to overcome this barrier to accessing care and achieving their best health.

Since our target is a binary feature, we implemented classification models. After thoroughly understanding each feature and its relationship with the target feature - transportation_issues, we reduced the number of features from 826 to 138. The shortlisted features belong to different categories like medical claims, pharmacy claims, CMS, demographics, and condition related.

We used the following technologies/tools:
- Google Cloud Platform as our computation engine
- Tableau for data visualization
- Python for data manipulation and model building
- RStudio (R Programming) for statistical analysis
- H20 driverless AI for training multiple models

After testing various models, we found the Light GBM machine learning model to perform best on both train and validation data. The model resulted in an AUC of 0.77 with 72.5% accuracy on a balanced class data set.

### 2. Background

The social and economic environments in which people live are inextricably related to health and well-being. Research has shown that medical care can be linked to just 20 percent of health, while social and economic factors, such as access to nutritious food, housing status, educational achievement and access to transport, account for 40 percent. 

Transportation is an economic and social factor that shapes people’s daily lives and thus a social determinant of health. It connects people from their origin to their destination, affects land use and shapes our daily lives. It is necessary to access goods, services and activities such as emergency services, health care, adequate food and clothing, education, employment, and social activities. It also can be a vehicle for wellness. Developing affordable and appropriate transportation options, walkable communities, bike lanes, bike-share programs and other healthy transit options can help boost health.
Transportation barriers can affect a person’s access to health care services. These barriers may result in missed or delayed health care appointments, increased health expenditures and overall poorer health outcomes. These statistics highlight the scope of the problem:

- 3.6 million people in the U.S. do not obtain medical care due to transportation barriers
- Regardless of insurance status, due to unavailable transportation, 4% of children (approximately 3 million) in the U.S. miss a health care appointment per year; this includes 9% of children in families with incomes of less than $50,000.
- ransportation is the third most commonly cited barrier to accessing health services for older adults.

There is a growing recognition that improving transportation access and support for members can help improve health outcomes and lower health costs.

### 3. Data Understanding

#### 3.1 Data Summary

To perform our analysis, we were given a training dataset of 69572 rows and 826 columns. The data was collected over a span of 1 year for members enrolled in the Humana Medicare Advantage and Prescription Drug (MAPD) plan. The data included medical claims, pharmacy claims, lab claims, demographics, consumers, credit, clinical conditions, CMS member data, and other member-related features.

#### 3.2 Target Feature

Our goal was to predict if the members are likely to struggle with transportation issues. The target feature was collected as a response to the transportation screening question - “In the past 12 months, has a lack of reliable transportation kept you from medical appointments, meetings, work or from getting things needed for daily living?” This information is captured in the transportation_issues binary feature where 0 indicates the member faced no transportation challenge and 1 indicates the member faced a transportation challenge.  

From the distribution of this feature, we can observe that 85% of the members do not face transportation issues which creates a class imbalance problem and should be taken care of while building the predictive model.  The document also talks about the descriptive analysis of the dataset.

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture1.png?raw=true "Optional Title") 

### 4. Feature Engineering and Exploratory Data Analysis

The data mainly included medical claims, pharmacy claims, lab claims, demographics, consumers, credit, clinical conditions, CMS member data, and other member-related features. This section discusses the inferences, features generated and interesting observations or patterns that are derived in the above-mentioned categories. Since, the data is augmented, the derived conclusions are limited and may not represent the real-world scenario. 

Instead of directly using all the features, we did thorough research and chose the features which we could understand, explain and relate to the final target feature. We believe that this will result in a richer data set that can support the model with fewer parameters.

There are two key types of features across multiple categories:
a.	IND - Binary indicator for different scenarios (Yes - 1/ No – 0)
b.	PMPM – Utilization per member per month calculated using total utilization by the number of months a person stayed with Humana in the past 12 months.

#### 4.1 Medical Claims

Medical claims have 3 sub-categories of features:

##### 4.1.1	CCS (Clinical Classifications Software) Procedure Codes

These features break down diagnosis and procedures into manageable categories that make statistical analysis and reporting easier. The official CCS website helped us to understand that the feature definition of these codes corresponds to the diagnostic code of the CCS and not the processes of the CCS. The data includes 21 such CCS codes. We dropped ccsp_034_ind and ccsp_0120 codes as they had no data. Since these member diagnoses can have an effect on the extent to which members may face transportation problems, we have used few of the features for our model building.

New features: 
- A binary indicator - whether the member used any of the CCS service
- Total number of diagnoses services a member has availed

Descriptive statistics:
- 15% of the members had superficial injury and contusion (ccsp_239_ind), and 30% of them faced transportation issues.
- 72% of the members availed at least one of the CCSP services (out of 21) and 15% of them had transportation issues.
- The plot below shows the frequency of top services used in CCSP category

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture2.png?raw=true "Optional Title") 
 	
##### 4.1.2	BETOS (Berenson-Eggers Type of Service) Procedure codes 

BETOS offers information on one of the seven types of services used to track member spending and is used for medical expenditure growth analysis. The type of tests conducted on the member provides insights into the medical conditions that may affect chances of facing transportation issues. There are seven test types (Evaluation and Management, Procedures, Imaging, Tests, Durable Medical Equipment, Other) out of which procedures and imaging does not contain any data. We used some of the default features along with newly curated ones to aggregate data from lower to a higher hierarchy.

New features:
-	ind_sum - number of services the member used out of all the services
-	pmpm_sum - sum of PMPM for all the services the member used in each category.
-	yes_no - binary indicator - did the member use any service in each category
-	yes_no_sum - total services a member used across all the categories
-	pmpm_total_sum - aggregated PMPM of a member across all the categories

Descriptive statistics:
-	Member id d97e73MOSe86T74L0Y5A8Ifb who does not face transportation issues has used 16 services in 5 categories that summed to 15.27 PMPM.
-	Evaluation category has maximum PMPM sum of 58935
 
 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture3.png?raw=true "Optional Title") 
 
##### 4.1.3	Utilization by category

These features give information about visits (ambulance, urgent care, emergency rooms, physician office, outpatient), hospital admit count and days for chronic, acute and maternity health conditions. The features are spread across two categories - medical and total. We have only focused on the total category as it covers both medical and non-medical categories.

New Features:
-	Binary indicator - if a person utilized any type of services (derived by pmpm value)

Descriptive statistics:
-	pmpm cost is high for outpatient and physician services rather than ambulance service
-	13% of members used ambulance service and 30% of them faced transportation issues
-	Total amount of pmpm spent on physician services across all the members is 43983

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture4.png?raw=true "Optional Title") 

#### 4.2 Pharmacy Claims

These features provide details about 100 different drugs used by the members. It also provides information on prescription and delivery of drugs - whether the prescription was branded or generic or whether the drugs were mailed. After discussing with a pharma expert, we categorized all the drugs based on their health/illness purpose into 18 categories.

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture5.png?raw=true "Optional Title") 

New features:
- ind_sum: Aggregated all the indicators in each group
- pmpm_sum: Aggregated pmpm in each group
- yes_no - binary indicator - did the member use drug in any of the 18 categories
- yes_no_sum - Total categories of drugs used 

As we had a lot of features, we clubbed the newly generated features with the original data and ran an XGboost algorithm to identify the most relevant features for model building. They are listed in the appendix.

Descriptive statistics: 
-	The person id 225de3M2OSd6ea8T54LYA1I9 used 15 out of 18 drug categories
-	Member id 2Mb6OcdS9fTc711a24LdYAI5 used 33 different types of drugs 
-	Average pmpm count is higher for cardiovascular and antibiotics drugs than others.
-	The plots below display pmpm cost per member across top categories
 
 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture6.png?raw=true "Optional Title") 
 
#### 4.3 Demographics or Consumer Data

A lot can be derived from the personal information of a member and socio-demographic information that can have a practical effect on the likelihood of facing transport problems. Due to the presence of high NULL values and the inability of imputation models on member level data, we discarded many features that we believe would be useful and selected age, sex, current smoker, former smoker, rural and urban category features from the data.

New features:
-	Binary indicator - urban or not (combined multiple levels of data)

Descriptive statistics:
-	Mean age of members facing transportation issues (66 years) is distinctively different from members who don't (71 years). Ideally, we would expect older people to face transportation issues.

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture7.png?raw=true "Optional Title") 

- Percentage of members (in different categories) facing transportation issues:

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture8.png?raw=true "Optional Title") 

#### 4.4 Condition Related Features

##### 4.4.1 Behavioral Health Condition Indicators

Mental health is essential to a person’s well-being. These indicators shed light on health conditions like bipolar disorder, alcohol abuse, substance abuse, tobacco use disorder, major depressive disorder, post-traumatic stress disorder and other anxiety disorders which can be directly linked to the possibility of having transportation issues. The graph below displays the total number of members present in each behavioral category:

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture9.png?raw=true "Optional Title") 

New features:
-	Binary indicator - whether the member faced any behavioral issue
-	Aggregation - total number of behavioral issues faced by the member

Descriptive Statistics:
-	There are more behavioral issues for dementia and other anxiety disorders
-	Three members (MO5S536T49e24f3L027Y37AI, bM4OfbS1Te347339LY13AIac and d4MOST0L18c896Y7A9Idc7d2) have all of the behavioral issues

##### 4.4.2 CCI (Charlson Comorbidity Index), FCI (Functional Comorbidity Index) and DCSI (Diabetes Complication Severity Index)

The CCI scores are used to categorize the severity of comorbidity into three grades: mild (CCI scores of 1–2) moderate (CCI scores of 3–4) and severe (CCI scores ≥5). The FCI scores range from 0 to 18 and DCSI scores range from 0 to 13.

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture10.png?raw=true "Optional Title") 
    
##### 4.4.3 MCC Diagnosis Code Categories
Major clinical categories are created to bin the type of diagnosis a member received. According to the description file presented by Humana there are 28 different categories and there are ample members who received more than one type of clinical diagnosis. These provide insights to the type of diagnosis which in turn helps to find the members who are most probable of facing transportation issues. For example, if a member is diagnosed with spine issues, he/she might be facing the transportation issue. 

New Features:
As the data is granular and contains the specific type of diagnosis claimed by the member, we have created new features based on each category.
-	Aggregating all the features in each category
-	Binary indicator - whether the person availed diagnosis in each of the category or not

Descriptive statistics:
-	Highest pmpm for a member is 73.03 (Member ID 18M71O8S48eT87dLY81A12I0)
-	Highest diagnosis count is 82 for member fM3O00b0S6a12dTLYA032bIe
-	The tree map below displays pmpm count greater than 40k across multiple MCC categories

![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture11.png?raw=true "Optional Title") 
 
#### 4.5 CMS Features 
The Centers for Medicare & Medicaid Services is a federal agency that oversees major healthcare programs. The organization seeks to provide better treatment, access to coverage, and enhanced health CMS features for the healthcare system. It provides interesting details on risk factors and indicators of members' eligibility for various programs. 

New features: 
The CMS features are divided into two categories CMS diagnosis and CMS risk score. We have created two set of features for each category:
-	Aggregation - total number of services availed by a member in each category
-	Binary - Whether a member has availed a service in at least one category
We replaced NULL values in some of the numeric features with mean and categorical columns with mode of the column of training data on both training and validation data.
 
 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture12.png?raw=true "Optional Title") 
 
### 5. Model Building

#### 5.1 Handling class imbalance
The dataset is severely imbalanced with only 14.6% of members struggling with transportation issues. Models built with imbalanced data can lead to biased results. To enhance the performance of the model and correctly classify the minority class (members facing transportation issues), we inspected the following artificial data generation techniques.

1.	Under sampling: Removing some observations of the majority class.
2.	Oversampling: Adding more observation of the minority class using duplication
3.	SMOTE (Synthetic Minority Oversampling Technique): Similar to oversampling, uses nearest neighbors algorithm to generate new and synthetic data.

Oversampling worked best for our dataset without causing bias-variance problems.

#### 5.2 Model Building

While choosing the machine learning model, we considered the accuracy vs. interpretability trade-off. Since we are dealing with healthcare and finance data, we wanted to build a model which emphasizes on introspection, causal effects, and processes.

We followed a sophisticated well-designed process to build our model. After understanding each feature, we conducted exploratory data analysis, identified some interesting patterns and shortlisted useful features for our classification model. To prepare the data for model building, we encoded the categorical features and standardized the numeric features. 

Instead of simply splitting the dataset into train and test, we used stratified k-fold cross validation to keep the same percentage of samples of target class for both training and validation. This significantly reduces bias and variance as we use most of the data for both fitting and validation.

We pipelined the features into machine learning and statistical models to check the performance and choose the best model for prediction. We had a tradeoff between choosing different models, either to choose Whitebox models for clear understanding of impact of each feature on target feature or to choose Blackbox models which have high predictability and less understandability of features impact- to this end we did both. We have implemented LightGBM as the machine learning model and used the generalized linear model with binary distribution (logistic regression) to validate ML interpretations. We believe this has struck a balance of complexity and interpretability. 

Light GBM is almost 7 times faster than XGBOOST and is a much better approach when dealing with large datasets. Light GBM is a fast, distributed, high-performance gradient boosting framework based on a decision tree algorithm, used for ranking, classification and many other machine learning tasks. The parameters used for the final model are colsample_bytree=0.8(Subsample ratio of columns when constructing each tree). learning_rate=0.02(The amount that the weights are updated during training), max_depth=8(Maximum tree depth for base learners), n_estimators=74(Number of boosted trees to fit), reg_lambda=1.0(L2 regularization term on weights), subsample=0.7(subsample ratio of the training instance). The Logistic model summarizes the coefficients and standard errors of how a dependent feature varies when an independent feature is changed by a unit. The interpretation is in terms of log-odds and can be easily converted to probabilities.

 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture13.png?raw=true "Optional Title") 
 
 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture14.png?raw=true "Optional Title") 
 
 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture15.png?raw=true "Optional Title") 
 
The Shapley plot gives clear output on the effect of each feature and the correlation. We can clearly see the cms_total_partd_payment_amt, cms_disabled_ind, ccsp_239_ind and betos_01a_ct_betos have high and positive impact on transportation issue (The high comes from the red color and positive impact shows from the X-axis). With increase in the value present in these columns the chances of a member facing the transportation issue goes up. On the other hand, est_age, ccsp_220_ind have negative correlation.

### 6. Insights and Recommendations

 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture16.png?raw=true "Optional Title") 
 
- cms_total_partd_payment is the payment made to private insurers for delivering prescription drug benefits to Medicare beneficiaries. This feature has a strong positive influence on transport problems. If the sum is higher than $200, the chances of having transportation difficulties would increase. This can be further supported by the statistical model, for every $100 increase in cms_total_partd_payment_amt, the log odds of transportation issue increase by 0.254. 

***Recommendation:*** Humana, with the aid of pharmacy partners, can assist patients spending more than $200, by delivering the prescription medications to their home / work along with FFS (fee for service) options by partnering with Uber/Lyft. This will alleviate the transportation efforts made by patients and improve the level of service offered to individual customers.

 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture17.png?raw=true "Optional Title") 

- From the plot above, we can observe that the probability of predicting transportation_issues dependent feature increases with decrease in age and its more evident when age is less than 60. The statistical model output also aligns with this pattern i.e. log-odds reduces by 0.03 when age increases by a year. The Medicare services are provided to members whose age is greater than 65, certain young people with disabilities and people with end-stage renal disease. 

***Recommendation:*** Younger members with disabilities and members with serious health conditions are at larger risk than few senior members among Medicare Beneficiaries. Providing disability services like in-home treatment will be beneficial.	

 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture18.png?raw=true "Optional Title") 

- Betos_01a_pmpm_ct_betos is the ambulance service code in the barenson-eggers type of service codes. The plot on the right clearly shows that the members who have pmpm count greater than 1, have higher chance of facing transportation issues and the log-odds increases by 0.189 with increase in betos_01a_pmpm_ct_betos by 1. The possible reason behind this can be that there is no reliable mode of transportation in their locality or the severity of their issue is high which needs frequent ambulance service.

***Recommendation:*** Contact members to know the issue for which the ambulance service is used and embrace telehealth services, where the severity of the problem is low.  

 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture19.png?raw=true "Optional Title") 
 
-	Functional Comorbidity Issues (FCI), is calculated on 18 aspects which reveal the overall health status of a person. From the plot, it is evident that members having score greater than 5 have higher chance of facing transportation issues and with every increase in score, the log odds of having transportation issues increases by 0.013.

***Recommendation:*** The clear distinction of threshold helps to identify members and  introduce Health Plans/ fitness plans. These Health literacy initiatives can be achieved by partnering with fitness services.

 ![](https://github.com/netisheth/Humana-Challenge/blob/master/Pictures/Picture20.png?raw=true "Optional Title") 

- There is an interesting pattern with interaction of cms_risk_adj_payment_rate_b_amt and cms_diasbled_ind. If a member is disabled then the increase in predictability of transportation is almost zero irrespective of the increase in payment_rate_b (as most of the red points are on the zero line), but if the member is not disabled and the cms_risk_adj_payment_rate_b_amt is greater than 500 then the predictability increases.

-	Increase in member awareness about the effects of various problems like the superficial and contusion injury (ccsp_239) which was one of the strong predictor for transportation issue. By studying more about this issue, we can educate members through pop-up notification or share it on the main website. 
-	Calculate in-house treatments for disabled members, to find optimal way of service.
-	Predicting members who might use ambulatory services, would help in pre-admission and mitigate possible risk of transportation.
-	The need for virtual connection is more needed than ever due to Covid. Creating a platform for members to meet the Primary care Physician (PCP) virtually might help reduce the emergency ambulance rides.

### 7.	Conclusion

SDOH is now being incorporated in Insurance and public health domain. With guidelines from CMS, various programs can be initiated based on the insights generated from the above analysis. By mitigating the transportation issue faced by its members, Humana will be able to solve one of the Social Determinants of health which in turn would not only improve health status and affordability of its members, but also cut-costs, reduce spending and support communities. By Identifying a small number of the high-risk members responsible for majority of healthcare costs, reaching, and engaging with those members is critical for improved outcomes. Since, transportation is a broader/holistic issue dependent on many economic factors. The increase in personal data combined with machine learning has potential to identify the vulnerable members.
