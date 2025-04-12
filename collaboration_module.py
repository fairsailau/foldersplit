import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import base64
from typing import Dict, List, Any, Tuple, Optional, Set
import logging

# Set up logging
logger = logging.getLogger(__name__)

class CollaborationAnalyzer:
    """
    A class that analyzes collaboration data and determines which collaborators will be affected by folder splits.
    
    This analyzer cross-references folder IDs from the folder split recommendations with item_ids in the 
    collaboration data to determine which collaborators need to be added to which folders and which 
    collaborators will lose access.
    
    Attributes:
        collaboration_df (pd.DataFrame): DataFrame containing collaboration data
        folder_df (pd.DataFrame): DataFrame containing folder data
        recommendations (dict): Dictionary containing folder split recommendations
        collaboration_analysis (dict): Dictionary containing collaboration analysis results
    """
    
    def __init__(self, collaboration_df: pd.DataFrame, folder_df: pd.DataFrame, recommendations: Dict[str, Any]):
        """
        Initialize the analyzer with collaboration data, folder data, and recommendations.
        
        Args:
            collaboration_df: DataFrame containing collaboration data with required columns:
                - item_id: ID to match with Folder ID in the folder data
                - item_type: Type of item
                - collaborator_id: ID of the collaborator
                - collaborator_name: Name of the collaborator
                - collaborator_login: Login of the collaborator
                - collaborator_type: Type of collaborator
                - collaborator_permission: Permission level of the collaborator
                - collaboration_id: ID of the collaboration
            folder_df: DataFrame containing folder data with required columns:
                - Path: Folder path separated by "/"
                - Folder Name: Name of the folder
                - Folder ID: Integer ID of the folder
                - Owner: Email of the user that owns the folder
                - Level: Folder level in the folder tree hierarchy (1 is root level)
            recommendations: Dictionary containing folder split recommendations
        """
        self.collaboration_df = collaboration_df
        self.folder_df = folder_df
        self.recommendations = recommendations
        self.collaboration_analysis = {}
        
        # Validate input data
        self._validate_collaboration_data()
    
    def _validate_collaboration_data(self) -> None:
        """
        Validate that the collaboration DataFrame has all required columns and proper data types.
        
        Raises:
            ValueError: If required columns are missing or have incorrect data types
        """
        required_columns = [
            'item_id', 'item_type', 'collaborator_id', 'collaborator_name', 
            'collaborator_login', 'collaborator_type', 'collaborator_permission', 'collaboration_id'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in self.collaboration_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns in collaboration data: {', '.join(missing_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check data types and convert if necessary
        try:
            # Ensure item_id is numeric for matching with Folder ID
            self.collaboration_df['item_id'] = pd.to_numeric(self.collaboration_df['item_id'])
            
            # Ensure string columns are strings
            string_columns = ['item_type', 'collaborator_name', 'collaborator_login', 
                             'collaborator_type', 'collaborator_permission']
            for col in string_columns:
                self.collaboration_df[col] = self.collaboration_df[col].astype(str)
        except Exception as e:
            error_msg = f"Error converting collaboration data types: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def analyze_collaborations(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze collaboration data and determine which collaborators will be affected by folder splits.
        
        This method implements the collaboration logic:
        - For root-level folder splits: collaborations remain unchanged
        - For subfolder splits:
          - Collaborators added directly to the subfolder will see it in their root folders
          - Collaborators added at parent-folder level or above will lose access
        
        Returns:
            Dictionary containing collaboration analysis results
        """
        logger.info("Analyzing collaboration data...")
        st.write("Analyzing collaboration data...")
        
        try:
            # Create a mapping of folder IDs to folder paths
            folder_id_to_path = {}
            folder_id_to_level = {}
            for _, row in self.folder_df.iterrows():
                if not pd.isna(row['Folder ID']):
                    folder_id = int(row['Folder ID'])
                    folder_id_to_path[folder_id] = row['Path']
                    folder_id_to_level[folder_id] = int(row['Level'])
            
            # Create a mapping of item_id to collaborators
            item_to_collaborators = {}
            for _, row in self.collaboration_df.iterrows():
                item_id = int(row['item_id'])
                if item_id not in item_to_collaborators:
                    item_to_collaborators[item_id] = []
                
                item_to_collaborators[item_id].append({
                    'collaborator_id': row['collaborator_id'],
                    'collaborator_name': row['collaborator_name'],
                    'collaborator_login': row['collaborator_login'],
                    'collaborator_type': row['collaborator_type'],
                    'collaborator_permission': row['collaborator_permission'],
                    'collaboration_id': row['collaboration_id']
                })
            
            # For each user with recommendations
            for user_email, user_recs in self.recommendations.items():
                # Initialize collaboration analysis for this user
                self.collaboration_analysis[user_email] = {
                    'user_email': user_email,
                    'collaborators_to_add': [],
                    'collaborators_losing_access': []
                }
                
                # Process each recommended split
                for folder in user_recs['recommended_splits']:
                    folder_id = folder['folder_id']
                    folder_level = folder['level']
                    
                    # Skip if folder ID is not available
                    if folder_id is None:
                        continue
                    
                    # Get collaborators for this folder
                    folder_collaborators = item_to_collaborators.get(folder_id, [])
                    
                    # For root-level folder splits (level 1), collaborations remain unchanged
                    if folder_level == 1:
                        logger.info(f"Root-level folder split for {folder['folder_path']}, collaborations remain unchanged")
                        continue
                    
                    # For subfolder splits (level > 1)
                    # Find parent folder IDs
                    parent_folder_ids = []
                    folder_path = folder['folder_path']
                    path_parts = folder_path.strip('/').split('/')
                    
                    # Build potential parent paths
                    for i in range(1, len(path_parts)):
                        parent_path = '/' + '/'.join(path_parts[:i]) + '/'
                        # Find folder ID for this parent path
                        for fid, path in folder_id_to_path.items():
                            if path == parent_path:
                                parent_folder_ids.append(fid)
                    
                    # Get collaborators for parent folders
                    parent_collaborators = []
                    for parent_id in parent_folder_ids:
                        parent_collaborators.extend(item_to_collaborators.get(parent_id, []))
                    
                    # Collaborators added directly to the subfolder will see it in their root folders
                    # Add them to the list of collaborators to add
                    for collaborator in folder_collaborators:
                        self.collaboration_analysis[user_email]['collaborators_to_add'].append({
                            'folder_id': folder_id,
                            'folder_name': folder['folder_name'],
                            'folder_path': folder['folder_path'],
                            'collaborator_id': collaborator['collaborator_id'],
                            'collaborator_name': collaborator['collaborator_name'],
                            'collaborator_login': collaborator['collaborator_login'],
                            'collaborator_type': collaborator['collaborator_type'],
                            'collaborator_permission': collaborator['collaborator_permission'],
                            'collaboration_id': collaborator['collaboration_id'],
                            'service_account': folder.get('assigned_to', 'Unknown')
                        })
                    
                    # Collaborators added at parent-folder level or above will lose access
                    # Add them to the list of collaborators losing access
                    for collaborator in parent_collaborators:
                        # Check if this collaborator is not already directly added to the subfolder
                        if not any(c['collaborator_id'] == collaborator['collaborator_id'] for c in folder_collaborators):
                            self.collaboration_analysis[user_email]['collaborators_losing_access'].append({
                                'folder_id': folder_id,
                                'folder_name': folder['folder_name'],
                                'folder_path': folder['folder_path'],
                                'collaborator_id': collaborator['collaborator_id'],
                                'collaborator_name': collaborator['collaborator_name'],
                                'collaborator_login': collaborator['collaborator_login'],
                                'collaborator_type': collaborator['collaborator_type'],
                                'collaborator_permission': collaborator['collaborator_permission'],
                                'collaboration_id': collaborator['collaboration_id']
                            })
                
                # Log summary
                logger.info(f"Collaboration analysis for {user_email}:")
                logger.info(f"Collaborators to add: {len(self.collaboration_analysis[user_email]['collaborators_to_add'])}")
                logger.info(f"Collaborators losing access: {len(self.collaboration_analysis[user_email]['collaborators_losing_access'])}")
            
            return self.collaboration_analysis
            
        except Exception as e:
            error_msg = f"Error analyzing collaborations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise
    
    def get_collaboration_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate tables for collaborators to add and collaborators losing access.
        
        Returns:
            Tuple containing:
            - DataFrame of collaborators to add
            - DataFrame of collaborators losing access
        """
        try:
            # Check if we have collaboration analysis
            if not self.collaboration_analysis:
                return pd.DataFrame(), pd.DataFrame()
            
            # Prepare data for collaborators to add
            collaborators_to_add_data = []
            for user_email, analysis in self.collaboration_analysis.items():
                for collaborator in analysis['collaborators_to_add']:
                    collaborators_to_add_data.append({
                        'User': user_email,
                        'Folder ID': collaborator['folder_id'],
                        'Folder Name': collaborator['folder_name'],
                        'Folder Path': collaborator['folder_path'],
                        'Collaborator ID': collaborator['collaborator_id'],
                        'Collaborator Name': collaborator['collaborator_name'],
                        'Collaborator Email': collaborator['collaborator_login'],
                        'Collaborator Type': collaborator['collaborator_type'],
                        'Permission': collaborator['collaborator_permission'],
                        'Service Account': collaborator['service_account']
                    })
            
            # Prepare data for collaborators losing access
            collaborators_losing_access_data = []
            for user_email, analysis in self.collaboration_analysis.items():
                for collaborator in analysis['collaborators_losing_access']:
                    collaborators_losing_access_data.append({
                        'User': user_email,
                        'Folder ID': collaborator['folder_id'],
                        'Folder Name': collaborator['folder_name'],
                        'Folder Path': collaborator['folder_path'],
                        'Collaborator ID': collaborator['collaborator_id'],
                        'Collaborator Name': collaborator['collaborator_name'],
                        'Collaborator Email': collaborator['collaborator_login'],
                        'Collaborator Type': collaborator['collaborator_type'],
                        'Permission': collaborator['collaborator_permission']
                    })
            
            # Create DataFrames
            collaborators_to_add_df = pd.DataFrame(collaborators_to_add_data)
            collaborators_losing_access_df = pd.DataFrame(collaborators_losing_access_data)
            
            return collaborators_to_add_df, collaborators_losing_access_df
            
        except Exception as e:
            error_msg = f"Error generating collaboration tables: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return pd.DataFrame(), pd.DataFrame()
    
    def visualize_collaboration_impact(self) -> Dict[str, Dict[str, plt.Figure]]:
        """
        Create visualizations of the collaboration impact.
        
        Returns:
            Dictionary of visualizations for each user
        """
        try:
            visualizations = {}
            
            # For each user with collaboration analysis
            for user_email, analysis in self.collaboration_analysis.items():
                # Count collaborators to add and losing access
                collab_to_add = len(analysis['collaborators_to_add'])
                collab_losing = len(analysis['collaborators_losing_access'])
                
                if collab_to_add > 0 or collab_losing > 0:
                    user_visualizations = {}
                    
                    # Create bar chart showing collaboration impact
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    labels = ['Collaborators to Add', 'Collaborators Losing Access']
                    values = [collab_to_add, collab_losing]
                    colors = ['#66b3ff', '#ff9999']
                    
                    ax.bar(labels, values, color=colors)
                    
                    # Add data labels
                    for i, v in enumerate(values):
                        ax.text(i, v + v*0.02, str(v), ha='center')
                    
                    ax.set_ylabel('Number of Collaborators')
                    ax.set_title(f'Collaboration Impact for {user_email}')
                    plt.tight_layout()
                    
                    user_visualizations['collaboration_impact'] = fig
                    
                    # Store visualizations for this user
                    visualizations[user_email] = user_visualizations
            
            return visualizations
            
        except Exception as e:
            error_msg = f"Error creating collaboration visualizations: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return {}

def display_collaboration_results(collaboration_analysis, collaborators_to_add_df, collaborators_losing_access_df, collaboration_visualizations):
    """
    Display the results of the collaboration analysis.
    
    Args:
        collaboration_analysis: Dictionary containing collaboration analysis results
        collaborators_to_add_df: DataFrame of collaborators to add
        collaborators_losing_access_df: DataFrame of collaborators losing access
        collaboration_visualizations: Dictionary of collaboration visualizations
    """
    st.header("Collaboration Analysis")
    
    if not collaboration_analysis:
        st.info("No collaboration analysis results to display.")
        return
    
    # Create tabs for collaboration analysis
    collab_tabs = st.tabs(["Collaborators to Add", "Collaborators Losing Access", "Visualizations"])
    
    with collab_tabs[0]:
        if not collaborators_to_add_df.empty:
            st.write("These collaborators need to be added to the new folders:")
            st.dataframe(collaborators_to_add_df)
            
            # Download CSV button for collaborators to add
            csv_to_add = collaborators_to_add_df.to_csv(index=False)
            b64_to_add = base64.b64encode(csv_to_add.encode()).decode()
            href_to_add = f'<a href="data:file/csv;base64,{b64_to_add}" download="collaborators_to_add.csv">Download Collaborators to Add CSV</a>'
            st.markdown(href_to_add, unsafe_allow_html=True)
        else:
            st.info("No collaborators need to be added.")
    
    with collab_tabs[1]:
        if not collaborators_losing_access_df.empty:
            st.write("These collaborators will lose access to the split folders:")
            st.dataframe(collaborators_losing_access_df)
            
            # Download CSV button for collaborators losing access
            csv_losing = collaborators_losing_access_df.to_csv(index=False)
            b64_losing = base64.b64encode(csv_losing.encode()).decode()
            href_losing = f'<a href="data:file/csv;base64,{b64_losing}" download="collaborators_losing_access.csv">Download Collaborators Losing Access CSV</a>'
            st.markdown(href_losing, unsafe_allow_html=True)
        else:
            st.info("No collaborators will lose access.")
    
    with collab_tabs[2]:
        if collaboration_visualizations:
            for user_email, user_viz in collaboration_visualizations.items():
                st.subheader(f"Collaboration Impact for {user_email}")
                if 'collaboration_impact' in user_viz:
                    st.pyplot(user_viz['collaboration_impact'])
        else:
            st.info("No collaboration visualizations available.")
    
    # Display detailed collaboration analysis for each user
    for user_email, analysis in collaboration_analysis.items():
        st.write("---")
        st.subheader(f"Detailed Collaboration Analysis for {user_email}")
        
        # Create tabs for user collaboration analysis
        user_collab_tabs = st.tabs(["Collaborators to Add", "Collaborators Losing Access"])
        
        with user_collab_tabs[0]:
            if analysis['collaborators_to_add']:
                # Create a DataFrame for collaborators to add
                collab_add_data = []
                for collab in analysis['collaborators_to_add']:
                    collab_add_data.append({
                        'Folder Name': collab['folder_name'],
                        'Folder Path': collab['folder_path'],
                        'Collaborator Name': collab['collaborator_name'],
                        'Collaborator Email': collab['collaborator_login'],
                        'Permission': collab['collaborator_permission'],
                        'Service Account': collab['service_account']
                    })
                
                collab_add_df = pd.DataFrame(collab_add_data)
                st.write(f"Collaborators to add ({len(collab_add_df)} total):")
                st.dataframe(collab_add_df, use_container_width=True)
            else:
                st.info("No collaborators need to be added for this user.")
        
        with user_collab_tabs[1]:
            if analysis['collaborators_losing_access']:
                # Create a DataFrame for collaborators losing access
                collab_lose_data = []
                for collab in analysis['collaborators_losing_access']:
                    collab_lose_data.append({
                        'Folder Name': collab['folder_name'],
                        'Folder Path': collab['folder_path'],
                        'Collaborator Name': collab['collaborator_name'],
                        'Collaborator Email': collab['collaborator_login'],
                        'Permission': collab['collaborator_permission']
                    })
                
                collab_lose_df = pd.DataFrame(collab_lose_data)
                st.write(f"Collaborators losing access ({len(collab_lose_df)} total):")
                st.dataframe(collab_lose_df, use_container_width=True)
            else:
                st.info("No collaborators will lose access for this user.")
