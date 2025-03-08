import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Smart Shopper Recommendations",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .merchant-card {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1565C0;
    }
    .metric-label {
        font-size: 1rem;
        color: #546E7A;
    }
</style>
""", unsafe_allow_html=True)

# Cache the data loading to improve performance
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('updated.csv')
        
        # Convert date columns if they exist
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
                
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to get merchant names
@st.cache_data
def get_merchant_name(df, mer_id):
    merchant_names = df[df['Mer_Id'] == mer_id]['Merchant_Name'].unique()
    if len(merchant_names) > 0:
        return merchant_names[0]
    return f"Merchant {mer_id}"

# Enhanced recommendation function
def get_recommendations(df, user_id, top_n_categories=3, top_n_merchants=5):
    if user_id not in df['User_Id'].values:
        return None, None, None
    
    # Get the cluster of the given User_Id
    user_cluster = df[df['User_Id'] == user_id]['Cluster'].iloc[0]
    
    # Get all users in the same cluster
    users_in_cluster = df[df['Cluster'] == user_cluster]
    
    # Get this user's data
    user_data = df[df['User_Id'] == user_id]
    
    # Group by Category and calculate metrics
    category_transactions = users_in_cluster.groupby('Category In English').agg({
        'Trx_Rank': 'sum',
        'User_Id': 'nunique'
    }).reset_index()
    
    # Calculate popularity score (combining transaction frequency and user count)
    category_transactions['Popularity_Score'] = (
        0.7 * category_transactions['Trx_Rank'] / category_transactions['Trx_Rank'].max() + 
        0.3 * category_transactions['User_Id'] / category_transactions['User_Id'].max()
    )
    
    # Sort categories based on popularity score in descending order
    category_transactions = category_transactions.sort_values(by='Popularity_Score', ascending=False)
    
    # Get the top categories
    top_categories = category_transactions['Category In English'].head(top_n_categories).tolist()
    
    # User's preferred categories
    user_preferred = user_data.groupby('Category In English')['Trx_Rank'].sum().reset_index()
    user_preferred = user_preferred.sort_values(by='Trx_Rank', ascending=False)
    user_top_categories = user_preferred['Category In English'].head(top_n_categories).tolist()
    
    recommendations = []
    
    for category in top_categories:
        # Get all transactions for the category
        category_data = users_in_cluster[users_in_cluster['Category In English'] == category]
        
        # Group by Merchant and calculate metrics
        merchant_transactions = category_data.groupby('Mer_Id').agg({
            'Trx_Rank': 'sum',
            'User_Id': 'nunique'
        }).reset_index()
        
        # Calculate merchant popularity score
        merchant_transactions['Popularity_Score'] = (
            0.7 * merchant_transactions['Trx_Rank'] / merchant_transactions['Trx_Rank'].max() + 
            0.3 * merchant_transactions['User_Id'] / merchant_transactions['User_Id'].max()
        )
        
        # Sort merchants based on popularity score in descending order
        merchant_transactions = merchant_transactions.sort_values(by='Popularity_Score', ascending=False)
        
        # Get the top merchants
        top_merchants = merchant_transactions['Mer_Id'].head(top_n_merchants).tolist()
        
        # Get merchant names if available in the dataset
        merchant_names = []
        for mer_id in top_merchants:
            name = get_merchant_name(df, mer_id)
            merchant_names.append((mer_id, name))
        
        recommendations.append((category, merchant_names))
    
    # Get cluster statistics
    cluster_stats = {
        'size': len(users_in_cluster['User_Id'].unique()),
        'avg_transactions': users_in_cluster.groupby('User_Id')['Trx_Rank'].sum().mean(),
        'top_categories': top_categories,
        'cluster_id': user_cluster
    }
    
    return recommendations, user_top_categories, cluster_stats

# Function to generate visual insights
def generate_insights(df, user_id):
    user_data = df[df['User_Id'] == user_id]
    
    # Transaction history over time
    if 'Transaction_Date' in df.columns:
        # Analyze transaction patterns by time
        user_data['Transaction_Month'] = user_data['Transaction_Date'].dt.month_name()
        monthly_transactions = user_data.groupby('Transaction_Month')['Trx_Rank'].count().reset_index()
        
        # Plot monthly transactions
        fig_monthly = px.bar(
            monthly_transactions, 
            x='Transaction_Month', 
            y='Trx_Rank',
            title='Monthly Transaction Pattern',
            labels={'Trx_Rank': 'Number of Transactions', 'Transaction_Month': 'Month'},
            color_discrete_sequence=['#1E88E5']
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Category distribution
    category_dist = user_data.groupby('Category In English')['Trx_Rank'].sum().reset_index()
    category_dist = category_dist.sort_values(by='Trx_Rank', ascending=False)
    
    fig_category = px.pie(
        category_dist, 
        values='Trx_Rank', 
        names='Category In English',
        title='Spending by Category',
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig_category.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig_category

def main():
    st.markdown("<h1 class='main-header'>🛍️ Smart Shopper Recommendations</h1>", unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.warning("Please ensure 'updated.csv' file is available in the application directory.")
        return
    
    # Check if 'Merchant_Name' column exists, if not add placeholder
    if 'Merchant_Name' not in data.columns:
        data['Merchant_Name'] = data['Mer_Id'].apply(lambda x: f"Merchant {x}")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("User Input")
        
        # Allow user to select input method
        input_method = st.radio("Choose input method:", ["Select from list", "Enter User ID manually"])
        
        if input_method == "Select from list":
            # Get unique sorted user IDs and create dropdown
            user_ids = sorted(data['User_Id'].unique())
            user_id = st.selectbox("Select User ID:", user_ids)
        else:
            # Manual input with validation
            user_id = st.number_input("Enter User ID:", min_value=0, step=1, value=0)
            # Check if the user ID exists
            if user_id not in data['User_Id'].values:
                st.warning("This User ID does not exist in the dataset. Please enter a valid ID.")
                valid_ids = sorted(data['User_Id'].unique())[:10]  # Show some examples
                st.info(f"Examples of valid User IDs: {', '.join(map(str, valid_ids))}")
        
        # Advanced settings
        st.header("Advanced Settings")
        top_n_categories = st.slider("Number of categories to recommend:", 1, 5, 2)
        top_n_merchants = st.slider("Number of merchants per category:", 3, 10, 5)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if user_id in data['User_Id'].values:
            # Get recommendations
            recommendations, user_categories, cluster_stats = get_recommendations(
                data, user_id, top_n_categories, top_n_merchants
            )
            
            if recommendations:
                st.markdown(f"<h2 class='sub-header'>Personalized Recommendations for User {user_id}</h2>", unsafe_allow_html=True)
                
                # Display recommendations in an organized way
                for idx, (category, merchants) in enumerate(recommendations, 1):
                    with st.expander(f"Category {idx}: {category}", expanded=True):
                        for i, (mer_id, merchant_name) in enumerate(merchants, 1):
                            st.markdown(f"""
                            <div class="merchant-card">
                                <strong>#{i}: {merchant_name}</strong> (ID: {mer_id})
                            </div>
                            """, unsafe_allow_html=True)
                
                # Show user's historical preferences
                st.markdown("<h2 class='sub-header'>Your Shopping Patterns</h2>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write("Your preferred shopping categories:")
                for i, category in enumerate(user_categories, 1):
                    st.write(f"{i}. {category}")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show cluster insights
                st.markdown("<h2 class='sub-header'>Shopping Community Insights</h2>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.write(f"You belong to a shopping community of {cluster_stats['size']} users with similar preferences.")
                st.write(f"Average transactions per user in your community: {cluster_stats['avg_transactions']:.1f}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("Could not generate recommendations. Please check the User ID and try again.")
        else:
            st.info("Please enter a valid User ID to get personalized recommendations.")
            
            # Show random insights from the dataset
            st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            
            # Number of users and transactions
            total_users = len(data['User_Id'].unique())
            total_transactions = len(data)
            total_merchants = len(data['Mer_Id'].unique())
            total_categories = len(data['Category In English'].unique())
            
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.markdown("<p class='metric-label'>Total Users</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{total_users:,}</p>", unsafe_allow_html=True)
            with col_b:
                st.markdown("<p class='metric-label'>Transactions</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{total_transactions:,}</p>", unsafe_allow_html=True)
            with col_c:
                st.markdown("<p class='metric-label'>Merchants</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{total_merchants:,}</p>", unsafe_allow_html=True)
            with col_d:
                st.markdown("<p class='metric-label'>Categories</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{total_categories:,}</p>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if user_id in data['User_Id'].values:
            # Display user insights
            st.markdown("<h2 class='sub-header'>Your Shopping Analysis</h2>", unsafe_allow_html=True)
            
            fig_category = generate_insights(data, user_id)
            st.plotly_chart(fig_category, use_container_width=True)
            
            # Display user statistics
            user_stats = data[data['User_Id'] == user_id]
            total_user_transactions = len(user_stats)
            unique_merchants = len(user_stats['Mer_Id'].unique())
            unique_categories = len(user_stats['Category In English'].unique())
            
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"""
                <p class='metric-label'>Your Total Transactions</p>
                <p class='metric-value'>{total_user_transactions}</p>
                <p class='metric-label'>Unique Merchants Visited</p>
                <p class='metric-value'>{unique_merchants}</p>
                <p class='metric-label'>Shopping Categories</p>
                <p class='metric-value'>{unique_categories}</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            # Show popular categories overall
            popular_categories = data.groupby('Category In English')['Trx_Rank'].sum().reset_index()
            popular_categories = popular_categories.sort_values(by='Trx_Rank', ascending=False).head(10)
            
            fig_popular = px.bar(
                popular_categories,
                x='Category In English',
                y='Trx_Rank',
                title='Most Popular Shopping Categories',
                color_discrete_sequence=['#1E88E5']
            )
            fig_popular.update_layout(xaxis_title="Category", yaxis_title="Transaction Volume")
            st.plotly_chart(fig_popular, use_container_width=True)
            
            # Cluster distribution
            cluster_dist = data.groupby('Cluster')['User_Id'].nunique().reset_index()
            cluster_dist.columns = ['Cluster', 'User Count']
            
            fig_cluster = px.pie(
                cluster_dist,
                values='User Count',
                names='Cluster',
                title='User Distribution Across Shopping Communities',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_cluster.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_cluster, use_container_width=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee;">
        <p>Smart Shopper Recommendation System - Making your shopping experience better, one recommendation at a time.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
