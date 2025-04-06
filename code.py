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
            
            # Save recommendations for this user
            self.recommendations[user_email] = recommendations
            
            st.write(f"Final recommendations for {user_email}:")
            st.write(f"Total recommended moves: {total_recommended_moves:,}")
            st.write(f"Final file count after all splits: {remaining_files:,}")
            st.write(f"Remaining excess: {recommendations['remaining_excess_files']:,}")
        
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
                'remaining_excess_files': user_recs.get('remaining_excess_files', 0)
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # Plot total file count vs. threshold for each user
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = range(len(summary_df))
            width = 0.35
            
            ax.bar([i - width/2 for i in x], summary_df['total_file_count'], width, label='Before Split')
            ax.bar([i + width/2 for i in x], summary_df['final_file_count'], width, label='After All Splits')
            
            ax.axhline(y=self.file_threshold, color='r', linestyle='--', label=f'Threshold ({self.file_threshold:,} files)')
            
            # Add data labels
            for i, v in enumerate(summary_df['total_file_count']):
                ax.text(i - width/2, v + v*0.02, f'{int(v):,}', ha='center', va='bottom', rotation=0, fontsize=8)
            
            for i, v in enumerate(summary_df['final_file_count']):
                ax.text(i + width/2, v + v*0.02, f'{int(v):,}', ha='center', va='bottom', rotation=0, fontsize=8)
            
            ax.set_xlabel('User Email')
            ax.set_ylabel('File Count')
            ax.set_title('Before vs. After All Splits by User')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['user_email'], rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()
            visualizations['all_users_before_after'] = fig
        
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
                'Status': 'Success' if user_recs['final_file_count'] <= self.file_threshold else 
                         'Partial Success' if user_recs['final_file_count'] < user_recs['total_file_count'] else 
                         'No Solution'
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
                        if 'all_users_before_after' in visualizations:
                            st.subheader("Overall Visualizations")
                            st.pyplot(visualizations['all_users_before_after'])
                        
                        # Display user-specific visualizations
                        for user_email, user_viz in visualizations.items():
                            if user_email == 'all_users_before_after':
                                continue
                            
                            st.subheader(f"Recommendations for {user_email}")
                            
                            # Display before/after visualization
                            st.pyplot(user_viz['before_after'])
                            
                            # Display recommended splits
                            user_recs = recommender.recommendations[user_email]
                            if user_recs['recommended_splits']:
                                st.write("Recommended Folder Splits:")
                                
                                # Create DataFrame for display
                                splits_df = pd.DataFrame(user_recs['recommended_splits'])
                                display_df = splits_df[['folder_path', 'level', 'current_file_count', 
                                                      'direct_file_count', 'recommended_files_to_move']]
                                display_df = display_df.rename(columns={
                                    'folder_path': 'Folder Path',
                                    'level': 'Level',
                                    'current_file_count': 'Total Files',
                                    'direct_file_count': 'Direct Files',
                                    'recommended_files_to_move': 'Files to Move'
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
