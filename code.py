import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import io
import base64
from zipfile import ZipFile

class FolderSplitRecommender:
    def __init__(self, df, file_threshold=500000):
        """Initialize the recommender with the dataframe and threshold."""
        self.df = df
        self.file_threshold = file_threshold
        self.users_exceeding = None
        self.recommendations = {}
        self.service_accounts = {}  # Track service accounts

    def calculate_user_stats(self):
        """Calculate statistics for each user and identify those exceeding threshold."""
        st.write("Calculating user statistics...")
        
        # Filter to only level 1 folders and group by user
        level1_df = self.df[self.df['Level'] == 1]
        
        # Group by user and calculate total file count from level 1 folders only
        self.user_stats = level1_df.groupby('Owner').agg(
            total_file_count=('File Count', 'sum'),
            total_size_mb=('Size (MB)', 'sum'),
            folder_count=('Folder ID', 'count')
        ).sort_values('total_file_count', ascending=False).reset_index()
        
        # Identify users exceeding the threshold
        self.users_exceeding = self.user_stats[self.user_stats['total_file_count'] > self.file_threshold]
        
        return self.users_exceeding

    def identify_nested_folders(self):
        """Identify parent-child relationships between folders."""
        st.write("Identifying nested folder relationships...")
        
        # Add path length for sorting
        self.df['path_length'] = self.df['Path'].str.len()
        self.df = self.df.sort_values('path_length')
        
        # Initialize direct file count with total file count
        self.df['direct_file_count'] = self.df['File Count']
        
        # Create a dictionary for faster lookups
        path_to_idx = {row['Path']: i for i, row in self.df.iterrows()}
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        total_rows = len(self.df)
        progress_bar = st.progress(0)
        
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch = self.df.iloc[start_idx:end_idx]
            
            for _, row in batch.iterrows():
                path = row['Path']
                
                # Skip root paths that don't have parent folders
                if path.count('/') <= 1:
                    continue
                
                # Find parent folder
                parent_path = '/'.join(path.split('/')[:-1]) + '/'
                
                if parent_path in path_to_idx:
                    parent_idx = path_to_idx[parent_path]
                    
                    # Subtract this folder's file count from parent's direct file count
                    self.df.at[parent_idx, 'direct_file_count'] -= row['File Count']
            
            # Update progress bar
            progress_bar.progress(min(end_idx / total_rows, 1.0))
        
        # Ensure direct file counts are not negative
        self.df['direct_file_count'] = self.df['direct_file_count'].clip(lower=0)
        
        return self.df

    def prioritize_folders(self):
        """Prioritize folders for splitting based on the level-based approach."""
        st.write("Prioritizing folders for splitting...")
        
        # First, identify nested folder relationships and calculate direct file counts
        self.identify_nested_folders()
        
        self.recommendations = {}
        self.service_accounts = {}  # Clear any existing service accounts
        
        # Process each user exceeding the threshold
        for _, user_row in self.users_exceeding.iterrows():
            user_email = user_row['Owner']
            total_file_count = user_row['total_file_count']
            
            st.write(f"Processing recommendations for user: {user_email}")
            st.write(f"Total file count (from level 1 folders): {total_file_count:,}")
            
            # Target for the user is exactly the threshold (or less if total is less)
            target_for_user = min(self.file_threshold, total_file_count)
            
            # Get only level 1 folders owned by this user
            # IMPORTANT: We only consider level 1 folders to avoid double-counting
            level1_folders = self.df[(self.df['Owner'] == user_email) & (self.df['Level'] == 1)].copy()
            
            # Sort by file count (ascending) to prioritize keeping smaller folders with original user
            level1_folders = level1_folders.sort_values('File Count')
            
            # Initialize recommendations for this user
            recommendations = {
                'user_email': user_email,
                'total_file_count': int(total_file_count),
                'excess_files': int(total_file_count - target_for_user),
                'recommended_splits': [],
                'service_accounts': {},  # Track service accounts for this user
                'final_file_count': target_for_user
            }
            
            # Folders to keep with original user
            folders_to_keep = []
            current_kept_count = 0
            
            # First pass: Identify folders to keep with original user
            for _, folder in level1_folders.iterrows():
                folder_file_count = folder['File Count']
                
                # If adding this folder keeps us under threshold, keep it
                if current_kept_count + folder_file_count <= target_for_user:
                    folders_to_keep.append(folder['Path'])
                    current_kept_count += folder_file_count
            
            # Calculate files to move
            files_to_move = total_file_count - current_kept_count
            
            st.write(f"Files kept with original user: {current_kept_count:,}")
            st.write(f"Files to move to service accounts: {files_to_move:,}")
            
            # Second pass: Distribute remaining folders to service accounts
            folders_to_move = level1_folders[~level1_folders['Path'].isin(folders_to_keep)]
            
            # Initialize service account tracking
            current_service_account = 1
            service_account_name = f"service_account_{current_service_account}"
            current_service_account_files = 0
            recommendations['service_accounts'][service_account_name] = 0
            
            # Track which folders need partial splitting
            partial_splits = []
            
            # For folders that entirely fit within threshold
            for _, folder in folders_to_move.iterrows():
                folder_path = folder['Path']
                folder_file_count = folder['File Count']
                
                # If this folder fits entirely in a service account
                if folder_file_count <= self.file_threshold:
                    # Check if adding this folder exceeds current service account capacity
                    if current_service_account_files + folder_file_count > self.file_threshold:
                        # Create a new service account
                        current_service_account += 1
                        service_account_name = f"service_account_{current_service_account}"
                        current_service_account_files = 0
                        recommendations['service_accounts'][service_account_name] = 0
                    
                    # Add to current service account
                    recommendations['service_accounts'][service_account_name] += folder_file_count
                    current_service_account_files += folder_file_count
                    
                    # Add to recommended splits
                    recommendations['recommended_splits'].append({
                        'folder_path': folder['Path'],
                        'folder_name': folder['Folder Name'],
                        'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                        'level': int(folder['Level']),
                        'current_file_count': int(folder_file_count),
                        'direct_file_count': int(folder['direct_file_count']),
                        'recommended_files_to_move': int(folder_file_count),
                        'new_owner': service_account_name,
                        'is_partial_split': False
                    })
                    
                    st.write(f"  Assigned folder: {folder_path} ({folder_file_count:,} files) to {service_account_name}")
                else:
                    # This folder needs to be split across multiple service accounts
                    partial_splits.append(folder.to_dict())  # Convert Series to dict
            
            # Handle folders that need to be split across multiple service accounts
            for folder in partial_splits:  # Changed from partial_splits.iterrows() to iterate directly over the list
                folder_path = folder['Path']
                folder_file_count = folder['File Count']
                
                # How many files remain to move from this folder
                remaining_files = folder_file_count
                
                st.write(f"  Splitting large folder: {folder_path} ({folder_file_count:,} files) across multiple service accounts")
                
                # Keep splitting until all files are allocated
                while remaining_files > 0:
                    # Space available in current service account
                    space_available = self.file_threshold - current_service_account_files
                    
                    # Files to move to this service account
                    files_for_this_account = min(space_available, remaining_files)
                    
                    # If no space in current account, create a new one
                    if files_for_this_account == 0:
                        current_service_account += 1
                        service_account_name = f"service_account_{current_service_account}"
                        current_service_account_files = 0
                        recommendations['service_accounts'][service_account_name] = 0
                        
                        # Recalculate files for this new account
                        space_available = self.file_threshold
                        files_for_this_account = min(space_available, remaining_files)
                    
                    # Add to current service account
                    recommendations['service_accounts'][service_account_name] += files_for_this_account
                    current_service_account_files += files_for_this_account
                    
                    # Add to recommended splits
                    recommendations['recommended_splits'].append({
                        'folder_path': folder['Path'],
                        'folder_name': folder['Folder Name'],
                        'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                        'level': int(folder['Level']),
                        'current_file_count': int(folder_file_count),
                        'direct_file_count': int(folder['direct_file_count']),
                        'recommended_files_to_move': int(files_for_this_account),
                        'new_owner': service_account_name,
                        'is_partial_split': True
                    })
                    
                    st.write(f"    Assigned {files_for_this_account:,} files to {service_account_name}")
                    
                    # Update remaining files
                    remaining_files -= files_for_this_account
                    
                    # If service account is full, prepare for next one
                    if current_service_account_files >= self.file_threshold:
                        current_service_account += 1
                        service_account_name = f"service_account_{current_service_account}"
                        current_service_account_files = 0
                        recommendations['service_accounts'][service_account_name] = 0
            
            # Calculate final statistics
            total_files_moved = sum([rec['recommended_files_to_move'] for rec in recommendations['recommended_splits']])
            
            # Verify our math is correct
            if abs(total_files_moved - files_to_move) > 10:  # Allow for rounding errors
                st.warning(f"Calculation discrepancy detected: Files to move ({files_to_move:,}) vs. Total moved ({total_files_moved:,})")
            
            recommendations['total_recommended_moves'] = int(total_files_moved)
            
            # Remove empty service accounts if any
            empty_accounts = [sa for sa, count in recommendations['service_accounts'].items() if count == 0]
            for sa in empty_accounts:
                del recommendations['service_accounts'][sa]
            
            # Save recommendations for this user
            self.recommendations[user_email] = recommendations
            self.service_accounts[user_email] = recommendations['service_accounts']
            
            st.write(f"Final recommendations for {user_email}:")
            st.write(f"Total files before splitting: {total_file_count:,}")
            st.write(f"Files kept with original user: {current_kept_count:,}")
            st.write(f"Total files moved: {total_files_moved:,}")
            st.write(f"Service accounts created: {len(recommendations['service_accounts'])}")
            
            # Show service account distribution
            for sa_name, sa_files in recommendations['service_accounts'].items():
                st.write(f"  {sa_name}: {sa_files:,} files")
            
            # Verify all service accounts are under threshold
            accounts_over_threshold = [sa for sa, count in recommendations['service_accounts'].items() if count > self.file_threshold]
            if accounts_over_threshold:
                st.error(f"Error: {len(accounts_over_threshold)} service accounts are over the threshold!")
            else:
                st.success("All service accounts are below the threshold")
        
        return self.recommendations

    def visualize_recommendations(self):
        """Create visualizations of the recommendations."""
        st.write("Creating visualizations...")
        
        visualizations = {}
        
        # For each user with recommendations
        for user_email, user_recs in self.recommendations.items():
            # Skip if no recommended splits
            if len(user_recs['recommended_splits']) == 0:
                continue
            
            # Create a DataFrame from the recommended splits
            splits_df = pd.DataFrame(user_recs['recommended_splits'])
            
            # Add a column to indicate partial splits
            splits_df['split_type'] = splits_df.apply(
                lambda x: 'Partial Split' if x.get('is_partial_split', False) else 'Complete Split',
                axis=1
            )
            
            # Truncate long folder names for better visualization
            splits_df['short_name'] = splits_df['folder_name'].apply(
                lambda x: x[:15] + '...' if len(str(x)) > 15 else x
            )
            
            user_visualizations = {}
            
            # Increase figure sizes for better readability
            plt.rcParams.update({'font.size': 10})
            
            # Plot recommended files to move by folder - dynamic height based on number of folders
            fig_height = max(8, min(20, len(splits_df) * 0.5))
            fig, ax = plt.subplots(figsize=(18, fig_height))
            
            # Add more left margin for folder names
            plt.subplots_adjust(left=0.3)
            
            # Use horizontal bar chart
            bars = ax.barh(splits_df['short_name'], splits_df['recommended_files_to_move'],
                          color=splits_df['split_type'].map({'Complete Split': 'green', 'Partial Split': 'orange'}))
            
            # Add data labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2,
                      f'{int(width):,}',
                      ha='left', va='center')
            
            ax.set_xlabel('Recommended Files to Move', fontsize=12)
            ax.set_ylabel('Folder Name', fontsize=12)
            ax.set_title(f'Recommended Folder Splits for {user_email}', fontsize=14)
            ax.tick_params(axis='both', labelsize=10)
            ax.legend(handles=[
                plt.Rectangle((0,0),1,1, color='green', label='Complete Split'),
                plt.Rectangle((0,0),1,1, color='orange', label='Partial Split')
            ], fontsize=12)
            
            plt.tight_layout()
            user_visualizations['recommendations'] = fig
            
            # Plot current vs. recommended file count - use abbreviated folder names
            fig, ax = plt.subplots(figsize=(18, fig_height))
            
            # Sort by level and then by file count
            splits_df_sorted = splits_df.sort_values(['level', 'current_file_count'], ascending=[True, False])
            
            x = range(len(splits_df_sorted))
            width = 0.35
            
            # Add more bottom margin for labels
            plt.subplots_adjust(bottom=0.3)
            
            ax.bar(x, splits_df_sorted['current_file_count'], width, label='Current File Count')
            ax.bar(x, splits_df_sorted['recommended_files_to_move'], width,
                 label='Files to Move', alpha=0.7, color='red')
            ax.axhline(y=self.file_threshold, color='green', linestyle='--',
                     label=f'Threshold ({self.file_threshold:,} files)')
            
            ax.set_xlabel('Folder Name', fontsize=12)
            ax.set_ylabel('File Count', fontsize=12)
            ax.set_title(f'Current vs. Recommended File Count for {user_email}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(splits_df_sorted['short_name'], rotation=45, ha='right', fontsize=9)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            user_visualizations['current_vs_recommended'] = fig
            
            # Plot before and after total file count
            fig, ax = plt.subplots(figsize=(12, 8))
            
            labels = ['Before Split', 'After All Splits']
            values = [user_recs['total_file_count'], user_recs['final_file_count']]
            colors = ['#ff9999', '#66b3ff']
            
            ax.bar(labels, values, color=colors)
            ax.axhline(y=self.file_threshold, color='green', linestyle='--',
                     label=f'Threshold ({self.file_threshold:,} files)')
            
            # Add data labels
            for i, v in enumerate(values):
                ax.text(i, v + v*0.02, f'{int(v):,}', ha='center', fontsize=12)
            
            ax.set_ylabel('Total File Count', fontsize=12)
            ax.set_title(f'Before vs. After All Splits for {user_email}', fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            user_visualizations['before_after'] = fig
            
            # Service account distribution visualization
            if user_recs['service_accounts']:
                fig, ax = plt.subplots(figsize=(15, 10))
                
                # Create labels for original user and service accounts
                labels = [f"{user_email}\n(After)"]
                values = [user_recs['final_file_count']]
                
                # Add service accounts
                for sa_name, sa_files in user_recs['service_accounts'].items():
                    labels.append(sa_name)
                    values.append(sa_files)
                
                # Create bars with different colors
                colors = ['#66b3ff'] + ['#ff9999'] * len(user_recs['service_accounts'])
                ax.bar(labels, values, color=colors)
                
                # Add threshold line
                ax.axhline(y=self.file_threshold, color='r', linestyle='--', 
                         label=f'Threshold ({self.file_threshold:,} files)')
                
                # Add data labels
                for i, v in enumerate(values):
                    ax.text(i, v + v*0.02, f'{int(v):,}', ha='center', fontsize=12)
                
                ax.set_ylabel('File Count', fontsize=12)
                ax.set_title(f'File Distribution After Splits for {user_email}', fontsize=14)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
                ax.tick_params(axis='both', labelsize=12)
                ax.legend(fontsize=12)
                
                plt.tight_layout()
                user_visualizations['service_accounts'] = fig
            
            visualizations[user_email] = user_visualizations
        
        # Create a summary visualization for all users
        summary_data = []
        for user_email, user_recs in self.recommendations.items():
            summary_data.append({
                'user_email': user_email,
                'total_file_count': user_recs['total_file_count'],
                'final_file_count': user_recs['final_file_count'],
                'excess_files': user_recs['total_file_count'] - user_recs['final_file_count'],
                'total_recommended_moves': user_recs.get('total_recommended_moves', 0),
                'service_accounts_count': len(user_recs['service_accounts'])
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Plot total file count vs. threshold for each user
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = range(len(summary_df))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], summary_df['total_file_count'], width, label='Before Split')
            ax.bar([i + width/2 for i in x], summary_df['final_file_count'], width, label='After All Splits')
            ax.axhline(y=self.file_threshold, color='r', linestyle='--', label=f'Threshold ({self.file_threshold:,} files)')
            
            # Add data labels
            for i, v in enumerate(summary_df['total_file_count']):
                ax.text(i - width/2, v + v*0.02, f'{int(v):,}', ha='center', va='bottom', rotation=0, fontsize=10)
            
            for i, v in enumerate(summary_df['final_file_count']):
                ax.text(i + width/2, v + v*0.02, f'{int(v):,}', ha='center', va='bottom', rotation=0, fontsize=10)
            
            ax.set_xlabel('User Email', fontsize=12)
            ax.set_ylabel('File Count', fontsize=12)
            ax.set_title('Before vs. After All Splits by User', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['user_email'], rotation=45, ha='right', fontsize=12)
            ax.tick_params(axis='both', labelsize=12)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            visualizations['all_users_before_after'] = fig
            
            # Create a comprehensive visualization showing all users and service accounts
            all_entities = []
            
            # Add original users after split
            for user_email, user_recs in self.recommendations.items():
                all_entities.append({
                    'name': f"{user_email}\n(After)",
                    'files': user_recs['final_file_count'],
                    'type': 'Original User'
                })
                
                # Add service accounts for this user
                for sa_name, sa_files in user_recs['service_accounts'].items():
                    all_entities.append({
                        'name': f"{sa_name}\n(from {user_email})",
                        'files': sa_files,
                        'type': 'Service Account'
                    })
            
            if all_entities:
                entities_df = pd.DataFrame(all_entities)
                entities_df = entities_df.sort_values('files', ascending=False)
                
                fig, ax = plt.subplots(figsize=(18, 10))
                
                # Define colors based on type
                colors = entities_df['type'].map({'Original User': '#66b3ff', 'Service Account': '#ff9999'})
                
                # Create bars
                bars = ax.bar(entities_df['name'], entities_df['files'], color=colors)
                
                # Add threshold line
                ax.axhline(y=self.file_threshold, color='r', linestyle='--', 
                         label=f'Threshold ({self.file_threshold:,} files)')
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{int(height):,}', ha='center', va='bottom', fontsize=10)
                
                ax.set_ylabel('File Count', fontsize=12)
                ax.set_title('File Distribution Across All Users and Service Accounts', fontsize=14)
                ax.set_xticklabels(entities_df['name'], rotation=45, ha='right', fontsize=10)
                ax.tick_params(axis='both', labelsize=10)
                ax.legend(handles=[
                    plt.Rectangle((0,0),1,1, color='#66b3ff', label='Original User'),
                    plt.Rectangle((0,0),1,1, color='#ff9999', label='Service Account')
                ], fontsize=12)
                
                plt.tight_layout()
                visualizations['all_entities'] = fig
        
        return visualizations

    def generate_recommendations_table(self):
        """Generate a summary table of recommendations."""
        summary_data = []
        
        for user_email, user_recs in self.recommendations.items():
            summary_data.append({
                'User': user_email,
                'Before Split': f"{user_recs['total_file_count']:,}",
                'After All Splits': f"{user_recs['final_file_count']:,}",
                'Files to Move': f"{user_recs.get('total_recommended_moves', 0):,}",
                'Service Accounts': len(user_recs['service_accounts']),
                'Status': 'Success' if user_recs['final_file_count'] <= self.file_threshold else
                         'Partial Success'
            })
            
            # Add rows for service accounts
            for sa_name, sa_files in user_recs['service_accounts'].items():
                summary_data.append({
                    'User': sa_name,
                    'Before Split': "0",
                    'After All Splits': f"{sa_files:,}",
                    'Files to Move': f"{sa_files:,}",
                    'Service Accounts': "",
                    'Status': 'Success' if sa_files <= self.file_threshold else 'Exceeds Threshold'
                })
        
        return pd.DataFrame(summary_data)

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        st.write("Starting analysis...")
        
        self.calculate_user_stats()
        
        if len(self.users_exceeding) == 0:
            st.success("No users exceed the threshold. No splits are required.")
            return {}
        
        self.prioritize_folders()
        visualizations = self.visualize_recommendations()
        
        st.write("Analysis complete.")
        return visualizations

def get_download_link(df):
    """Generate a download link for the DataFrame as CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="recommendations.csv">Download CSV</a>'
    return href

def main():
    st.set_page_config(page_title="Folder Split Recommender", layout="wide")
    
    st.title("Folder Split Recommender")
    
    st.markdown("### Analyze folder ownership data and get recommendations for splitting content")
    
    with st.expander("About This Tool", expanded=True):
        st.markdown("""
        This tool analyzes folder ownership data and provides recommendations for splitting content based on file count thresholds.
        
        **Key Features:**
        
        1. Correctly calculates total file count per user by summing **only level 1 folders** (since file counts of level 2, 3, etc. are already included in level 1 counts)
        
        2. Processes all levels sequentially (level 1, then level 2, then level 3, etc.)
        
        3. Continues adding folders from each level until the user's total file count is reduced to 500,000 or less
        
        4. Shows the **total count after all splits** for each user in the "After All Splits" column
        
        5. Assigns files to service accounts (service_account_1, service_account_2, etc.) while ensuring each service account stays under the threshold
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    # Threshold setting
    threshold = st.number_input("File Count Threshold", min_value=1000, value=500000, step=1000,
                              help="Maximum number of files a user should own. Default is 500,000.")
    
    if uploaded_file is not None:
        # Display debugging information
        st.subheader("Debug Information")
        
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            # Display column names for debugging
            st.write("CSV Headers:", df.columns.tolist())
            
            # Display raw data preview
            with st.expander("Raw Data Preview"):
                st.dataframe(df.head())
            
            # Run analysis
            if st.button("Generate Recommendations"):
                try:
                    with st.spinner("Analyzing data and generating recommendations..."):
                        st.write("Creating recommender with threshold:", threshold)
                        recommender = FolderSplitRecommender(df, threshold)
                        
                        st.write("Running analysis...")
                        visualizations = recommender.run_analysis()
                        
                        # Check if any users exceed threshold
                        if recommender.users_exceeding is not None and len(recommender.users_exceeding) > 0:
                            # Display summary table
                            st.subheader("Summary of Recommendations")
                            summary_table = recommender.generate_recommendations_table()
                            st.dataframe(summary_table)
                            
                            # Provide explanation of summary table
                            with st.expander("Understanding the Summary Table"):
                                st.markdown("""
                                **Summary Table Explanation:**
                                
                                - **Before Split**: Total number of files owned by the user before any splits
                                - **After All Splits**: Number of files remaining with the original user after moving folders to service accounts (should be around 500,000)
                                - **Files to Move**: Total number of files that will be moved to service accounts
                                - **Service Accounts**: Number of service accounts created to hold the moved files
                                - **Status**: Success = at or below threshold, Partial Success = reduced but still over threshold
                                
                                The numbers should always add up: Before Split = After All Splits + Files to Move
                                
                                Each service account will have no more than 500,000 files. If a user has more than 500,000 files, we'll create as many service accounts as needed to keep everyone under the threshold.
                                """)
                            
                            # Provide download link for summary table
                            st.markdown(get_download_link(summary_table), unsafe_allow_html=True)
                            
                            # Display overall visualizations
                            st.subheader("Overall Visualizations")
                            
                            if 'all_users_before_after' in visualizations:
                                st.pyplot(visualizations['all_users_before_after'])
                            
                            if 'all_entities' in visualizations:
                                st.pyplot(visualizations['all_entities'])
                            
                            # Display user-specific visualizations
                            for user_email, user_viz in visualizations.items():
                                if user_email in ['all_users_before_after', 'all_entities']:
                                    continue
                                
                                st.subheader(f"Recommendations for {user_email}")
                                
                                # Display before/after visualization
                                st.pyplot(user_viz['before_after'])
                                
                                # Display service account distribution if available
                                if 'service_accounts' in user_viz:
                                    st.pyplot(user_viz['service_accounts'])
                                
                                # Display recommended splits
                                user_recs = recommender.recommendations[user_email]
                                
                                if user_recs['recommended_splits']:
                                    st.write("Recommended Folder Splits:")
                                    
                                    # Create DataFrame for display
                                    splits_df = pd.DataFrame(user_recs['recommended_splits'])
                                    display_df = splits_df[['folder_path', 'level', 'current_file_count',
                                                           'direct_file_count', 'recommended_files_to_move', 'new_owner']]
                                    
                                    display_df = display_df.rename(columns={
                                        'folder_path': 'Folder Path',
                                        'level': 'Level',
                                        'current_file_count': 'Total Files',
                                        'direct_file_count': 'Direct Files',
                                        'recommended_files_to_move': 'Files to Move',
                                        'new_owner': 'New Owner'
                                    })
                                    
                                    # Add split type column
                                    display_df['Split Type'] = splits_df.apply(
                                        lambda x: 'Partial Split' if x.get('is_partial_split', False) else 'Complete Split',
                                        axis=1
                                    )
                                    
                                    st.dataframe(display_df)
                                    
                                    # Display visualizations
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.pyplot(user_viz['recommendations'])
                                    
                                    with col2:
                                        st.pyplot(user_viz['current_vs_recommended'])
                                else:
                                    st.write("No suitable folders found for splitting.")
                            
                            # Create a downloadable ZIP with all results
                            st.subheader("Download All Results")
                            
                            # Create a buffer for the ZIP file
                            zip_buffer = io.BytesIO()
                            
                            with ZipFile(zip_buffer, 'w') as zip_file:
                                # Add summary table
                                zip_file.writestr('summary_table.csv', summary_table.to_csv(index=False))
                                
                                # Add recommendations JSON
                                zip_file.writestr('recommendations.json', json.dumps(recommender.recommendations, default=str, indent=4))
                                
                                # Add user statistics
                                zip_file.writestr('user_stats.csv', recommender.user_stats.to_csv(index=False))
                                
                                # Add users exceeding threshold
                                zip_file.writestr('users_exceeding_threshold.csv', recommender.users_exceeding.to_csv(index=False))
                                
                                # Add service account information
                                zip_file.writestr('service_accounts.json', json.dumps(recommender.service_accounts, default=str, indent=4))
                            
                            # Create download link for ZIP
                            zip_buffer.seek(0)
                            b64 = base64.b64encode(zip_buffer.read()).decode()
                            href = f'<a href="data:application/zip;base64,{b64}" download="folder_split_recommendations.zip">Download All Results (ZIP)</a>'
                            st.markdown(href, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)  # This will show the full stack trace
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)  # This will show the full stack trace

if __name__ == '__main__':
    main()
