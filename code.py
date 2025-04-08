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
        
        # For each user exceeding the threshold
        for _, user_row in self.users_exceeding.iterrows():
            user_email = user_row['Owner']
            total_file_count = user_row['total_file_count']
            excess_files = total_file_count - self.file_threshold
            
            st.write(f"Processing recommendations for user: {user_email}")
            st.write(f"Total file count (from level 1 folders): {total_file_count:,}, Excess files: {excess_files:,}")
            
            # Get all folders owned by this user
            user_folders = self.df[self.df['Owner'] == user_email].copy()
            
            # Initialize recommendations for this user
            recommendations = {
                'user_email': user_email,
                'total_file_count': int(total_file_count),
                'excess_files': int(excess_files),
                'recommended_splits': [],
                'service_accounts': {},  # Track service accounts for this user
                'folders_to_move': []    # Track which folders to move
            }
            
            # Get the maximum folder level for this user
            max_level = user_folders['Level'].max()
            
            # Track folders that have been processed
            processed_folders = set()
            
            # Track folders to move
            folders_to_move = []
            
            # Initialize service account tracking
            current_service_account = 1
            current_service_account_files = 0
            service_account_name = f"service_account_{current_service_account}"
            recommendations['service_accounts'][service_account_name] = 0
            
            # Process all levels sequentially
            for level in range(1, int(max_level) + 1):
                st.write(f"Processing level {level} folders...")
                
                # Get folders at this level
                level_folders = user_folders[user_folders['Level'] == level].copy()
                
                # Sort by file count in descending order to prioritize larger folders
                level_folders = level_folders.sort_values('File Count', ascending=False)
                
                # Find suitable candidates at this level
                candidates = []
                for _, folder in level_folders.iterrows():
                    folder_path = folder['Path']
                    
                    # Skip folders that have already been processed
                    if folder_path in processed_folders:
                        continue
                    
                    folder_file_count = folder['File Count']
                    
                    # Skip folders that are too small to be worth splitting
                    if folder_file_count < 10000:
                        continue
                    
                    # Check if this folder is a good candidate (≤ threshold files)
                    if folder_file_count <= self.file_threshold:
                        candidates.append({
                            'folder_path': folder['Path'],
                            'folder_name': folder['Folder Name'],
                            'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                            'level': int(folder['Level']),
                            'current_file_count': int(folder['File Count']),
                            'direct_file_count': int(folder['direct_file_count'])
                        })
                
                # If we found candidates at this level, add them to recommendations
                if candidates:
                    st.write(f"Found {len(candidates)} candidates at level {level}")
                    
                    # Sort candidates by file count (descending) to prioritize larger folders
                    candidates.sort(key=lambda x: x['current_file_count'], reverse=True)
                    
                    # Add candidates to folders_to_move until we're below threshold
                    for candidate in candidates:
                        folder_path = candidate['folder_path']
                        folder_file_count = candidate['current_file_count']
                        
                        # Mark this folder as processed
                        processed_folders.add(folder_path)
                        
                        # Add to folders to move
                        folders_to_move.append(candidate)
            
            # Calculate total files in selected folders
            total_files_to_move = sum([folder['current_file_count'] for folder in folders_to_move])
            
            # Sort folders by size (largest first)
            folders_to_move.sort(key=lambda x: x['current_file_count'], reverse=True)
            
            # Calculate remaining files after moving all selected folders
            remaining_files = total_file_count - total_files_to_move
            
            # If we've reduced below threshold, we might not need to use all folders
            # Otherwise, we'll need to consider a partial split
            if remaining_files <= self.file_threshold:
                # We've identified enough complete folders to move
                st.write(f"Successfully identified folders to move. Remaining files: {remaining_files:,}")
            else:
                st.write(f"Still above threshold ({remaining_files:,} > {self.file_threshold:,}), looking for partial splits...")
                
                # Find a folder for partial splitting
                all_folders = user_folders.sort_values(['Level', 'File Count'], ascending=[True, False])
                
                for _, folder in all_folders.iterrows():
                    folder_path = folder['Path']
                    
                    # Skip processed folders
                    if folder_path in processed_folders:
                        continue
                    
                    folder_file_count = folder['File Count']
                    
                    # Skip small folders
                    if folder_file_count < 10000:
                        continue
                    
                    # Calculate how many files to move from this folder
                    files_to_move = remaining_files - self.file_threshold
                    
                    if files_to_move > 0 and files_to_move < folder_file_count:
                        partial_folder = {
                            'folder_path': folder['Path'],
                            'folder_name': folder['Folder Name'],
                            'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                            'level': int(folder['Level']),
                            'current_file_count': int(folder['File Count']),
                            'direct_file_count': int(folder['direct_file_count']),
                            'recommended_files_to_move': int(files_to_move),
                            'is_partial_split': True
                        }
                        
                        folders_to_move.append(partial_folder)
                        processed_folders.add(folder_path)
                        
                        # Update remaining files
                        remaining_files = self.file_threshold
                        st.write(f"Added partial split of folder: {folder['Path']}, new total: {remaining_files:,}")
                        break
            
            # Now assign folders to service accounts
            for folder in folders_to_move:
                folder_file_count = folder.get('recommended_files_to_move', folder['current_file_count'])
                
                # Check if current service account can accommodate this folder
                if current_service_account_files + folder_file_count > self.file_threshold:
                    # Create a new service account
                    current_service_account += 1
                    current_service_account_files = 0
                    service_account_name = f"service_account_{current_service_account}"
                    recommendations['service_accounts'][service_account_name] = 0
                
                # Update service account files
                recommendations['service_accounts'][service_account_name] += folder_file_count
                current_service_account_files += folder_file_count
                
                # Add to recommended splits
                split_info = folder.copy()
                split_info['recommended_files_to_move'] = folder.get('recommended_files_to_move', folder['current_file_count'])
                split_info['new_owner'] = service_account_name
                split_info['new_total_after_split'] = total_file_count - sum([
                    s.get('recommended_files_to_move', s['current_file_count']) 
                    for s in recommendations['recommended_splits'] + [split_info]
                ])
                
                recommendations['recommended_splits'].append(split_info)
                
                st.write(f"Assigned {folder['folder_path']} to {service_account_name}, files: {folder_file_count:,}")
            
            # Calculate final file counts
            total_files_moved = sum([rec.get('recommended_files_to_move', rec['current_file_count']) 
                                   for rec in recommendations['recommended_splits']])
            
            # Correctly set final file count for original user
            recommendations['final_file_count'] = int(total_file_count - total_files_moved)
            recommendations['total_recommended_moves'] = int(total_files_moved)
            recommendations['remaining_excess_files'] = max(0, recommendations['final_file_count'] - self.file_threshold)
            
            # Save recommendations for this user
            self.recommendations[user_email] = recommendations
            self.service_accounts[user_email] = recommendations['service_accounts']
            
            st.write(f"Final recommendations for {user_email}:")
            st.write(f"Total files before splitting: {total_file_count:,}")
            st.write(f"Total files to move: {total_files_moved:,}")
            st.write(f"Final file count after all splits: {recommendations['final_file_count']:,}")
            st.write(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
            st.write(f"Service accounts created: {len(recommendations['service_accounts'])}")
            
            # Show service account distribution
            for sa_name, sa_files in recommendations['service_accounts'].items():
                st.write(f"  {sa_name}: {sa_files:,} files")
        
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
                lambda x: x[:30] + '...' if len(str(x)) > 30 else x
            )
            
            user_visualizations = {}
            
            # Plot recommended files to move by folder
            fig, ax = plt.subplots(figsize=(15, 10))  # Increased from (12, 8)
            
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
            
            # Plot current vs. recommended file count
            fig, ax = plt.subplots(figsize=(15, 10))  # Increased from (12, 8)
            
            # Sort by level and then by file count
            splits_df_sorted = splits_df.sort_values(['level', 'current_file_count'], ascending=[True, False])
            
            x = range(len(splits_df_sorted))
            width = 0.35
            
            ax.bar(x, splits_df_sorted['current_file_count'], width, label='Current File Count')
            ax.bar(x, splits_df_sorted['recommended_files_to_move'], width,
                 label='Files to Move', alpha=0.7, color='red')
            ax.axhline(y=self.file_threshold, color='green', linestyle='--',
                     label=f'Threshold ({self.file_threshold:,} files)')
            
            ax.set_xlabel('Folder Name', fontsize=12)
            ax.set_ylabel('File Count', fontsize=12)
            ax.set_title(f'Current vs. Recommended File Count for {user_email}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(splits_df_sorted['short_name'], rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            user_visualizations['current_vs_recommended'] = fig
            
            # Plot before and after total file count
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased from (10, 6)
            
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
                fig, ax = plt.subplots(figsize=(15, 10))  # Increased from (12, 8)
                
                # Create labels for original user and service accounts
                labels = [f"{user_email} (After)"]
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
                'excess_files': user_recs['excess_files'],
                'total_recommended_moves': user_recs.get('total_recommended_moves', 0),
                'remaining_excess_files': user_recs.get('remaining_excess_files', 0),
                'service_accounts_count': len(user_recs['service_accounts'])
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Plot total file count vs. threshold for each user
            fig, ax = plt.subplots(figsize=(12, 8))  # Increased from (10, 6)
            
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
                    'name': f"{user_email} (After)",
                    'files': user_recs['final_file_count'],
                    'type': 'Original User'
                })
                
                # Add service accounts for this user
                for sa_name, sa_files in user_recs['service_accounts'].items():
                    all_entities.append({
                        'name': f"{sa_name} (from {user_email})",
                        'files': sa_files,
                        'type': 'Service Account'
                    })
            
            if all_entities:
                entities_df = pd.DataFrame(all_entities)
                entities_df = entities_df.sort_values('files', ascending=False)
                
                fig, ax = plt.subplots(figsize=(16, 10))  # Increased from (14, 8)
                
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
                         'Partial Success' if user_recs['final_file_count'] < user_recs['total_file_count'] else
                         'No Solution'
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
