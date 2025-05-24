import os
# Sample Dataset Schema/Examples
sample_schema_examples = [
    {
        "dataset_name": "Customer.csv",
        "columns": ["UserID", "SessionCount", "age", "PctSessionPurchase", "ClicksPerSession", "ItemsClickedPerSession", "AvgTimePerClick", "AvgTimePerSession", "AvgPriceClicked", "PctSessionClickDiscount", "PctSessionPurchaseDiscount"],
        "description": "Customer dataset with demographic info and purchase behavior like clickstreams in online store."
    },
    {
        "dataset_name": "NYC-BikeShare-2015-2017-combined.csv",
        "columns": ["Trip Duration", "Start Time", "Stop Time", "Start Station ID", "Start Station Name", "Start Station Latitude", "Start Station Longitude", "End Station ID", "End Station Name", "End Station Latitude", "End Station Longitude", "Bike ID", "User Type", "Birth Year", "Gender", "Trip_Duration_in_min"],
        "description": "NYC bike share dataset with trip and user details from 2015–2017."
    },
    {
        "dataset_name": "StoreTraffic.csv",
        "columns": ["date", "store_id", "store_innum", "store_innum_before_two_weeks", "last_7d_store_innum_mean_shift_14d", "last_7d_store_innum_std_shift_14d", "last_14d_store_innum_mean_shift_14d", "last_14d_store_innum_std_shift_14d", "last_28d_store_innum_mean_shift_14d", "last_28d_store_innum_std_shift_14d", "mall_id", "city_id", "mall_innum", "last_7d_mall_innum_mean_shift_14d", "last_7d_mall_innum_std_shift_14d", "last_14d_mall_innum_mean_shift_14d", "last_14d_mall_innum_std_shift_14d", "last_28d_mall_innum_mean_shift_14d", "last_28d_mall_innum_std_shift_14d", "month", "day_of_week", "is_weekend", "is_legal_holiday", "is_workday", "is_work_tmw", "holiday_level", "holiday_length", "th_holiday", "holiday_type", "holiday_name", "days_to_last_legal_holiday", "days_to_next_legal_holiday", "days_to_last_release_date", "days_to_next_release_date", "days_to_last_sale_date", "days_to_next_sale_date"],
        "description": "Store foot traffic dataset including holidays, weather, and promotion effects."
    },
    {
        "dataset_name": "TelecomChurn.csv",
        "columns": ["CustomerID", "Gender", "SeniorCitizen", "Partner", "Dependents", "Tenure", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"],
        "description": "Telecom customer churn dataset with service details and churn labels."
    }
]

def get_similar_schema(user_columns):
    # Return the example with the most column intersection
    best = None
    best_score = 0
    for ex in sample_schema_examples:
        score = len(set(user_columns) & set(ex['columns']))
        if score > best_score:
            best = ex
            best_score = score
    return best 