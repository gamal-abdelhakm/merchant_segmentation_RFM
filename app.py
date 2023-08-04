import pandas as pd
from sklearn.cluster import KMeans
import streamlit as st

# Load the data from Cleaned_Data_Merchant_Level.csv
data = pd.read_csv('updated.csv')


def get_recommendations(user_id):
    # Get the cluster of the given User_Id
    user_cluster = data[data['User_Id'] == user_id]['Cluster'].iloc[0]
    
    # Get all users in the same cluster
    users_in_cluster = data[data['Cluster'] == user_cluster]
    
    # Group by Category and calculate the total number of transactions for each category
    category_transactions = users_in_cluster.groupby('Category In English')['Trx_Rank'].sum().reset_index()
    
    # Sort categories based on transaction frequency in descending order
    category_transactions = category_transactions.sort_values(by='Trx_Rank', ascending=False)
    
    # Get the top two categories
    top_categories = category_transactions['Category In English'].head(2)
    
    recommendations = []
    
    for category in top_categories:
        # Get all transactions for the top category
        category_data = users_in_cluster[users_in_cluster['Category In English'] == category]
        
        # Group by Merchant and calculate the total number of transactions for each Merchant
        merchant_transactions = category_data.groupby('Mer_Id')['Trx_Rank'].sum().reset_index()
        
        # Sort merchants based on transaction frequency in descending order
        merchant_transactions = merchant_transactions.sort_values(by='Trx_Rank', ascending=False)
        
        # Get the top three merchants
        top_merchants = merchant_transactions['Mer_Id'].head(5).tolist()
        
        recommendations.append((category, top_merchants))
    
    return recommendations

def main():
    st.title("Customer Recommendation System")
    st.write("Enter a User_Id to get personalized recommendations.")
    
    # Input user ID
    user_id = st.number_input("Enter User_Id", min_value=0, step=1, value=0)
    
    # Convert user_id to int, as Streamlit returns it as float
    user_id = int(user_id)
    
    if st.button("Get Recommendations"):
        recommendations = get_recommendations(user_id)
        st.subheader(f"Recommendations for User_Id {user_id}:")
        
        for idx, (category, merchants) in enumerate(recommendations, 1):
            st.write(f"{idx}. Category: {category}, Top Merchants: {merchants}")

if __name__ == "__main__":
    main()
