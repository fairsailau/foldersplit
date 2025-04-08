import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import io
import base64
from zipfile import ZipFile
import numpy as np

class FolderSplitRecommender:
    def __init__(self, df, file_threshold=500000):
        """Initialize the recommender with the dataframe and threshold."""
        self.df = df
        self.file_threshold = file_threshold
        self.users_exceeding = None
        self.recommendations = {}
        
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
    
    def assign_to_service_accounts(self, user_email, folders_to_split):
        """
        Assign folders to service accounts, keeping related folders together when possible
        and ensuring each service account stays under the threshold.
        """
        st.write(f"Assigning folders to service accounts for user: {user_email}")
        
        # Sort folders by path to keep related folders together
        folders_to_split.sort(key=lambda x: x['folder_path'])
        
        service_accounts = []
        current_account = {
            'account_name': f'service_account_1',
            'folders': [],
            'total_files': 0
        }
        service_accounts.append(current_account)
        
        # Group folders by common parent paths (first level)
        folder_groups = {}
        for folder in folders_to_split:
            # Extract first level path component
            path_parts = folder['folder_path'].strip('/').split('/')
            if len(path_parts) > 0:
                group_key = path_parts[0]
                if group_key not in folder_groups:
                    folder_groups[group_key] = []
                folder_groups[group_key].append(folder)
            else:
                # Handle root folders
                if 'root' not in folder_groups:
                    folder_groups['root'] = []
                folder_groups['root'].append(folder)
        
        # Process each group of related folders
        for group_key, group_folders in folder_groups.items():
            # Sort folders within group by file count (descending)
            group_folders.sort(key=lambda x: x['current_file_count'], reverse=True)
            
            for folder in group_folders:
                # Check if adding this folder would exceed the threshold for current account
                if current_account['total_files'] + folder['recommended_files_to_move'] > self.file_threshold:
                    # If current account would exceed threshold, create a new one
                    current_account = {
                        'account_name': f'service_account_{len(service_accounts) + 1}',
                        'folders': [],
                        'total_files': 0
                    }
                    service_accounts.append(current_account)
                
                # Add folder to current service account
                folder['assigned_to'] = current_account['account_name']
                current_account['folders'].append(folder)
                current_account['total_files'] += folder['recommended_files_to_move']
        
        return service_accounts
    
    def prioritize_folders(self):
        """Prioritize folders for splitting based on the level-based approach and assign to service accounts."""
        st.write("Prioritizing folders for splitting...")
        
        # First, identify nested folder relationships and calculate direct file counts
        self.identify_nested_folders()
        
        self.recommendations = {}
        
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
                'recommended_splits': []
            }
            
            # Get the maximum folder level for this user
            max_level = user_folders['Level'].max()
            
            # Track remaining files to be split
            remaining_files = total_file_count
            
            # Process all levels sequentially
            for level in range(1, int(max_level) + 1):
                st.write(f"Processing level {level} folders...")
                
                # If we've already reduced below threshold, break
                if remaining_files <= self.file_threshold:
                    st.write(f"Already below threshold after level {level-1}, stopping")
                    break
                
                # Get folders at this level
                level_folders = user_folders[user_folders['Level'] == level].copy()
                
                # Sort by file count in descending order to prioritize larger folders
                level_folders = level_folders.sort_values('File Count', ascending=False)
                
                # Find suitable candidates at this level
                candidates = []
                
                for _, folder in level_folders.iterrows():
                    folder_file_count = folder['File Count']
                    
                    # Skip folders that are too small to be worth splitting
                    if folder_file_count < 10000:
                        continue
                    
                    # Check if this folder is a good candidate (≤ threshold files)
                    if folder_file_count <= self.file_threshold:
                        # Calculate how much this split would reduce the total
                        new_total = remaining_files - folder_file_count
                        
                        # Add to candidates if it helps reduce the total
                        candidates.append({
                            'folder_path': folder['Path'],
                            'folder_name': folder['Folder Name'],
                            'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                            'level': int(folder['Level']),
                            'current_file_count': int(folder['File Count']),
                            'direct_file_count': int(folder['direct_file_count']),
                            'new_total': new_total
                        })
                
                # If we found candidates at this level, add them to recommendations
                if candidates:
                    st.write(f"Found {len(candidates)} candidates at level {level}")
                    
                    # Sort candidates by file count (descending) to prioritize larger folders
                    candidates.sort(key=lambda x: x['current_file_count'], reverse=True)
                    
                    # Add candidates to recommendations until we're below threshold or no more candidates
                    for candidate in candidates:
                        # Skip if we're already below threshold
                        if remaining_files <= self.file_threshold:
                            break
                        
                        # Add this candidate to recommendations
                        recommendations['recommended_splits'].append({
                            'folder_path': candidate['folder_path'],
                            'folder_name': candidate['folder_name'],
                            'folder_id': candidate['folder_id'],
                            'level': candidate['level'],
                            'current_file_count': candidate['current_file_count'],
                            'direct_file_count': candidate['direct_file_count'],
                            'recommended_files_to_move': candidate['current_file_count'],
                            'new_total_after_split': candidate['new_total']
                        })
                        
                        # Update remaining files
                        remaining_files = candidate['new_total']
                        st.write(f"  Added folder: {candidate['folder_path']}, new total: {remaining_files:,}")
                
                st.write(f"After level {level}, remaining files: {remaining_files:,}")
            
            # If we still haven't reached the threshold, look for partial splits
            if remaining_files > self.file_threshold:
                st.write("Still above threshold after all levels, looking for partial splits...")
                
                # Get all folders sorted by level and then by file count
                all_folders = user_folders.sort_values(['Level', 'File Count'], ascending=[True, False])
                
                for _, folder in all_folders.iterrows():
                    folder_file_count = folder['File Count']
                    
                    # Skip folders that are too small
                    if folder_file_count < 10000:
                        continue
                    
                    # For larger folders, calculate how many files to move
                    files_to_move = remaining_files - self.file_threshold
                    
                    # Only recommend if this folder has enough files
                    if files_to_move > 0 and files_to_move < folder_file_count:
                        recommendations['recommended_splits'].append({
                            'folder_path': folder['Path'],
                            'folder_name': folder['Folder Name'],
                            'folder_id': int(folder['Folder ID']) if not pd.isna(folder['Folder ID']) else None,
                            'level': int(folder['Level']),
                            'current_file_count': int(folder['File Count']),
                            'direct_file_count': int(folder['direct_file_count']),
                            'recommended_files_to_move': int(files_to_move),
                            'new_total_after_split': self.file_threshold,
                            'is_partial_split': True
                        })
                        
                        # Update remaining files
                        remaining_files = self.file_threshold
                        st.write(f"Added partial split of folder: {folder['Path']}, new total: {remaining_files:,}")
                        break
            
            # Calculate total recommended moves and remaining excess
            total_recommended_moves = sum([rec.get('recommended_files_to_move', 0) for rec in recommendations['recommended_splits']])
            recommendations['total_recommended_moves'] = total_recommended_moves
            recommendations['remaining_excess_files'] = int(remaining_files - self.file_threshold) if remaining_files > self.file_threshold else 0
            recommendations['final_file_count'] = int(remaining_files)
            
            # Assign folders to service accounts
            service_accounts = self.assign_to_service_accounts(user_email, recommendations['recommended_splits'])
            recommendations['service_accounts'] = service_accounts
            
            # Save recommendations for this user
            self.recommendations[user_email] = recommendations
            
            st.write(f"Final recommendations for {user_email}:")
            st.write(f"Total recommended moves: {total_recommended_moves:,}")
            st.write(f"Final file count after all splits: {remaining_files:,}")
            st.write(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
            st.write(f"Number of service accounts needed: {len(service_accounts)}")
        
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
            
            user_visualizations = {}
            
            # Plot recommended files to move by folder
            fig, ax = plt.subplots(figsize=(12, 8))
            bars = ax.barh(splits_df['folder_name'], splits_df['recommended_files_to_move'], 
                    color=splits_df['split_type'].map({'Complete Split': 'green', 'Partial Split': 'orange'}))
            
            # Add data labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width + width*0.02, bar.get_y() + bar.get_height()/2, 
                        f'{int(width):,}',
                        ha='left', va='center')
            
            ax.set_xlabel('Recommended Files to Move')
            ax.set_ylabel('Folder Name')
            ax.set_title(f'Recommended Folder Splits for {user_email}')
            ax.legend(handles=[
                plt.Rectangle((0,0),1,1, color='green', label='Complete Split'),
                plt.Rectangle((0,0),1,1, color='orange', label='Partial Split')
            ])
            plt.tight_layout()
            user_visualizations['recommendations'] = fig
            
            # Plot current vs. recommended file count
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Sort by level and then by file count
            splits_df_sorted = splits_df.sort_values(['level', 'current_file_count'], ascending=[True, False])
            
            x = range(len(splits_df_sorted))
            width = 0.35
            
            ax.bar(x, splits_df_sorted['current_file_count'], width, label='Current File Count')
            ax.bar(x, splits_df_sorted['recommended_files_to_move'], width, 
                   label='Files to Move', alpha=0.7, color='red')
            
            ax.axhline(y=self.file_threshold, color='green', linestyle='--', 
                       label=f'Threshold ({self.file_threshold:,} files)')
            
            ax.set_xlabel('Folder Name')
            ax.set_ylabel('File Count')
            ax.set_title(f'Current vs. Recommended File Count for {user_email}')
            ax.set_xticks(x)
            ax.set_xticklabels(splits_df_sorted['folder_name'], rotation=90)
            ax.legend()
            plt.tight_layout()
            user_visualizations['current_vs_recommended'] = fig
            
            # Plot before and after total file count
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = ['Before Split', 'After All Splits']
            values = [user_recs['total_file_count'], user_recs['final_file_count']]
            colors = ['#ff9999', '#66b3ff']
            
            ax.bar(labels, values, color=colors)
            ax.axhline(y=self.file_threshold, color='green', linestyle='--', 
                       label=f'Threshold ({self.file_threshold:,} files)')
            
            # Add data labels
            for i, v in enumerate(values):
                ax.text(i, v + v*0.02, f'{int(v):,}', ha='center')
            
            ax.set_ylabel('Total File Count')
            ax.set_title(f'Before vs. After All Splits for {user_email}')
            ax.legend()
            plt.tight_layout()
            user_visualizations['before_after'] = fig
            
            # NEW: Plot distribution across service accounts
            if 'service_accounts' in user_recs and user_recs['service_accounts']:
                # Create data for service account distribution
                account_names = [account['account_name'] for account in user_recs['service_accounts']]
                account_names.insert(0, f"{user_email} (After Split)")
                
                file_counts = [account['total_files'] for account in user_recs['service_accounts']]
                file_counts.insert(0, user_recs['final_file_count'])
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(12, 8))
                bars = ax.bar(account_names, file_counts, color=['#66b3ff'] + ['#ff9999'] * len(user_recs['service_accounts']))
                
                # Add threshold line
                ax.axhline(y=self.file_threshold, color='green', linestyle='--', 
                           label=f'Threshold ({self.file_threshold:,} files)')
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f'{int(height):,}', ha='center', va='bottom')
                
                ax.set_xlabel('Account')
                ax.set_ylabel('File Count')
                ax.set_title(f'File Distribution Across Accounts for {user_email}')
                plt.xticks(rotation=45, ha='right')
                ax.legend()
                plt.tight_layout()
                user_visualizations['service_account_distribution'] = fig
                
                # NEW: Create a visualization showing which folders go to which service account
                # Group folders by service account
                account_folders = {}
                for folder in user_recs['recommended_splits']:
                    account = folder.get('assigned_to', 'Unknown')
                    if account not in account_folders:
                        account_folders[account] = []
                    account_folders[account].append(folder)
                
                # Create a stacked bar chart showing folder distribution
                account_labels = list(account_folders.keys())
                folder_counts = []
                folder_labels = []
                
                # Prepare data for stacked bar chart
                for account in account_labels:
                    folders = account_folders[account]
                    for folder in folders:
                        folder_counts.append(folder['recommended_files_to_move'])
                        folder_labels.append(f"{folder['folder_name']} ({account})")
                
                # Create horizontal stacked bar chart
                fig, ax = plt.subplots(figsize=(14, 10))
                
                # Use a colormap to assign different colors to different folders
                colors = plt.cm.viridis(np.linspace(0, 1, len(folder_counts)))
                
                # Create the stacked bar chart
                y_pos = 0
                for i, (count, label, color) in enumerate(zip(folder_counts, folder_labels, colors)):
                    ax.barh(y_pos, count, color=color, label=label)
                    # Add text label inside the bar if there's enough space
                    if count > max(folder_counts) * 0.05:
                        ax.text(count/2, y_pos, label, ha='center', va='center', color='white')
                    y_pos += 1
                
                ax.set_yticks([])  # Hide y-axis ticks
                ax.set_xlabel('File Count')
                ax.set_title(f'Folder Distribution Across Service Accounts for {user_email}')
                
                # Create a custom legend
                handles = [plt.Rectangle((0,0),1,1, color=color) for color in colors]
                ax.legend(handles, folder_labels, loc='upper right', bbox_to_anchor=(1.1, 1), 
                          ncol=1, fontsize='small')
                
                plt.tight_layout()
                user_visualizations['folder_distribution'] = fig
            
            visualizations[user_email] = user_visualizations
        
        return visualizations

def main():
    st.title("Folder Splitting Recommendation Tool")
    
    st.write("""
    This tool analyzes folder ownership data and provides recommendations for splitting content 
    based on file count thresholds. It identifies users who own more than 500,000 files and 
    recommends which folders to split to bring users below this threshold.
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    # Threshold setting
    threshold = st.number_input("File Count Threshold", min_value=1, value=500000, step=1000)
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.write("Data loaded successfully!")
            st.write(f"Total rows: {len(df)}")
            
            # Display sample of the data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Check required columns
            required_columns = ['Path', 'Folder Name', 'Folder ID', 'Owner', 'Size (MB)', 'File Count', 'Level']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.stop()
            
            # Rename columns if needed
            column_mapping = {
                'folder_name': 'Folder Name',
                'folder_id': 'Folder ID',
                'owner_email': 'Owner',
                'size_mb': 'Size (MB)',
                'file_count': 'File Count',
                'level': 'Level'
            }
            
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Process data
            if st.button("Generate Recommendations"):
                try:
                    st.subheader("Analysis Results")
                    
                    # Create recommender
                    recommender = FolderSplitRecommender(df, threshold)
                    
                    # Calculate user statistics
                    users_exceeding = recommender.calculate_user_stats()
                    
                    # Display users exceeding threshold
                    st.write(f"Found {len(users_exceeding)} users exceeding the threshold of {threshold:,} files:")
                    st.dataframe(users_exceeding)
                    
                    # Generate recommendations
                    recommendations = recommender.prioritize_folders()
                    
                    # Create visualizations
                    visualizations = recommender.visualize_recommendations()
                    
                    # Display recommendations for each user
                    st.subheader("Recommendations by User")
                    
                    # Create a summary table
                    summary_data = []
                    
                    # First add original users with their updated file counts
                    for user_email, user_recs in recommendations.items():
                        # The original user should have exactly the threshold number of files after splitting
                        # or their original count if it was already below threshold
                        final_count = min(threshold, user_recs['total_file_count'])
                        
                        summary_data.append({
                            'User': user_email,
                            'Before Split': user_recs['total_file_count'],
                            'After All Splits': final_count,
                            'Files to Move': user_recs['total_recommended_moves'],
                            'Service Accounts': len(user_recs.get('service_accounts', [])),
                            'Status': 'Success' if final_count <= threshold else 'Partial Success'
                        })
                        
                        # Then add service accounts
                        if 'service_accounts' in user_recs:
                            for account in user_recs['service_accounts']:
                                summary_data.append({
                                    'User': account['account_name'],
                                    'Before Split': 0,  # Service accounts start with 0 files
                                    'After All Splits': account['total_files'],
                                    'Files to Move': account['total_files'],  # All files are moved to this account
                                    'Service Accounts': '',  # Service accounts don't have their own service accounts
                                    'Status': 'Success' if account['total_files'] <= threshold else 'Partial Success'
                                })
                    
                    summary_table = pd.DataFrame(summary_data)
                    st.dataframe(summary_table)
                    
                    # Download CSV button
                    csv = summary_table.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="summary.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Overall visualizations
                    st.subheader("Overall Visualizations")
                    
                    # Create a bar chart comparing before and after for all users
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Filter to only original users (not service accounts)
                    original_users = summary_table[summary_table['Before Split'] > 0]
                    
                    users = original_users['User']
                    before_values = original_users['Before Split']
                    after_values = original_users['After All Splits']
                    
                    x = np.arange(len(users))
                    width = 0.35
                    
                    ax.bar(x - width/2, before_values, width, label='Before Split')
                    ax.bar(x + width/2, after_values, width, label='After All Splits')
                    
                    # Add threshold line
                    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:,} files)')
                    
                    # Add data labels
                    for i, v in enumerate(before_values):
                        ax.text(i - width/2, v + v*0.02, f'{int(v):,}', ha='center', rotation=90)
                    
                    for i, v in enumerate(after_values):
                        ax.text(i + width/2, v + v*0.02, f'{int(v):,}', ha='center', rotation=90)
                    
                    ax.set_ylabel('File Count')
                    ax.set_title('Before vs. After All Splits by User')
                    ax.set_xticks(x)
                    ax.set_xticklabels(users)
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display detailed recommendations for each user
                    for user_email, user_recs in recommendations.items():
                        st.write("---")
                        st.subheader(f"Detailed Recommendations for {user_email}")
                        
                        st.write(f"Total file count before split: {user_recs['total_file_count']:,}")
                        # The final count should be the threshold or less
                        final_count = min(threshold, user_recs['total_file_count'])
                        st.write(f"Total file count after all splits: {final_count:,}")
                        st.write(f"Total files to move: {user_recs['total_recommended_moves']:,}")
                        
                        # Display service account information
                        if 'service_accounts' in user_recs and user_recs['service_accounts']:
                            st.subheader(f"Service Account Distribution for {user_email}")
                            st.write(f"Number of service accounts needed: {len(user_recs['service_accounts'])}")
                            
                            # Create a table showing service account distribution
                            account_data = []
                            for account in user_recs['service_accounts']:
                                account_data.append({
                                    'Service Account': account['account_name'],
                                    'Total Files': account['total_files'],
                                    'Number of Folders': len(account['folders'])
                                })
                            
                            account_df = pd.DataFrame(account_data)
                            st.dataframe(account_df)
                            
                            # Display folder assignments for each service account
                            st.subheader("Folder Assignments to Service Accounts")
                            for account in user_recs['service_accounts']:
                                st.write(f"**{account['account_name']}** - {account['total_files']:,} files")
                                
                                # Create a table of folders for this account
                                folder_data = []
                                for folder in account['folders']:
                                    folder_data.append({
                                        'Folder Name': folder['folder_name'],
                                        'Folder Path': folder['folder_path'],
                                        'Files to Move': folder['recommended_files_to_move'],
                                        'Split Type': 'Partial Split' if folder.get('is_partial_split', False) else 'Complete Split'
                                    })
                                
                                folder_df = pd.DataFrame(folder_data)
                                st.dataframe(folder_df)
                        
                        # Display recommended splits
                        if len(user_recs['recommended_splits']) > 0:
                            st.subheader("Recommended Folder Splits")
                            
                            # Create a DataFrame from the recommended splits
                            splits_df = pd.DataFrame(user_recs['recommended_splits'])
                            
                            # Create a display DataFrame with renamed columns
                            display_df = splits_df.rename(columns={
                                'folder_name': 'Folder Name',
                                'folder_path': 'Folder Path',
                                'level': 'Level',
                                'current_file_count': 'Total Files',
                                'direct_file_count': 'Direct Files',
                                'recommended_files_to_move': 'Files to Move',
                                'assigned_to': 'Assigned To'
                            })
                            
                            # Add split type column
                            display_df['Split Type'] = splits_df.apply(
                                lambda x: 'Partial Split' if x.get('is_partial_split', False) else 'Complete Split',
                                axis=1
                            )
                            
                            st.dataframe(display_df)
                            
                            # Display visualizations
                            user_viz = visualizations.get(user_email, {})
                            
                            if user_viz:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.pyplot(user_viz.get('recommendations', None))
                                with col2:
                                    st.pyplot(user_viz.get('current_vs_recommended', None))
                                
                                # Display service account distribution visualization
                                if 'service_account_distribution' in user_viz:
                                    st.pyplot(user_viz['service_account_distribution'])
                                
                                # Display folder distribution visualization
                                if 'folder_distribution' in user_viz:
                                    st.pyplot(user_viz['folder_distribution'])
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
                            
                            # Add service account assignments
                            service_account_data = []
                            for user_email, user_recs in recommendations.items():
                                if 'service_accounts' in user_recs and user_recs['service_accounts']:
                                    for account in user_recs['service_accounts']:
                                        for folder in account['folders']:
                                            service_account_data.append({
                                                'User': user_email,
                                                'Service Account': account['account_name'],
                                                'Folder Name': folder['folder_name'],
                                                'Folder Path': folder['folder_path'],
                                                'Files to Move': folder['recommended_files_to_move'],
                                                'Split Type': 'Partial Split' if folder.get('is_partial_split', False) else 'Complete Split'
                                            })
                            
                            if service_account_data:
                                service_account_df = pd.DataFrame(service_account_data)
                                zip_file.writestr('service_account_assignments.csv', service_account_df.to_csv(index=False))
                        
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
