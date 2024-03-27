# EcoVadis-Churn-Prediction
 
## Notable insights:
The following information was found from data analysis that was applied to the 'Assignment (churn prediction).xlsx' file

 ### Preliminary analysis:
    * Dataset size: (10,000 x 15)

    * Feature types:
        RowNumber             int64
        CustomerId            int64
        Surname              object
        CreditScore           int64
        Country              object
        Gender               object
        Age                   int64
        Tenure                int64
        CustomerFeedback     object
        Balance (EUR)       float64
        NumberOfProducts      int64
        HasCreditCard         int64
        IsActiveMember        int64
        EstimatedSalary     float64
        Exited                int64

### Data Cleaning:

    CustomerFeedback is the only column with NaN values (a totla of 6982). This 
    missing data appears independent of whether or not the user has left the 
    system and so will be filled with empty speech marks to prevent NaN errors
    during model training. 

    * There are no duplicated results in the customer feedback. 

Data Analysis:

* The average customer has been with the company for 5 years and holds a 
    balance of around 76,000 euros with a credit score of 650. The majority of 
    users have a credit card and use at least one product. 

    Command:

        customer_num = customer_data[['CreditScore' , 'Age', 'Tenure', 'Balance (EUR)', 'NumberOfProducts', 'HasCreditCard', 'IsActiveMember', 'EstimatedSalary']]
        
        print (customer_num.describe())
    
* The expected salary a user has does not seem to affect whether or not they 
    are more likely to exit or not. It is only in the salary range >170,000 that 
    the ratio of those leaving trends consistently below 20%. 
   
    Graph: estimated_salary.png
    
    Command: 
            
            customer_data['EstimatedSalary'] = customer_data['EstimatedSalary'].round(decimals=-4)
            data_analysis.compare_mode_against_existed(customer_data, "EstimatedSalary")


* It is expected that the longer the customer is with the service the more
    they are devoted to staying and not exiting. This only seems to become
    the case at the 10 year mark with the ratio of staying vs leaving
    remaining similar from years 1-9. 
    
    Graph: tenure.png
    
    Command: 
        
        data_analysis.compare_mode_against_existed(customer_data, "Tenure")

* Customers are more likely to leave the server once their balance becomes >0. This is 
        likely due to the fact that those without money in the account or "ghost" users who
        do not use the service (logic only works if product is free)
        
    Graph: binary_balance.png
        
    Command:
            
            customer_data['Balance (EUR)'].values[customer_data['Balance (EUR)'].values > 0] = 1
            data_analysis.compare_mode_against_existed(customer_data, "Balance (EUR)", title, save_name)

* It is interesting that regardless of how active a user is with a balance of 0, they are just as
        likely to stay as they are to leave. 
        
    Graph:        zero_balance_active.png
    
    Command:
    
            customer_data['Balance (EUR)'].values[customer_data['Balance (EUR)'].values > 0] = 1
            data_analysis.compare_mode_against_existed(customer_data, "IsActiveMember", title, save_name)

        (in compare_mode_against_existed)

            customer_stayed = cust_data[(cust_data['Balance (EUR)'] == 0) & (cust_data['Exited']==0)]
            customer_exited = cust_data[(cust_data['Balance (EUR)'] == 0) & (cust_data['Exited']==1)]