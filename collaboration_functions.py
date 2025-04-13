import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import io
import base64
import json
import logging
from typing import Dict, List, Any, Tuple, Optional

# Function to analyze collaborations based on folder splits
def analyze_collaborations(collaboration_df, folder_splits_df):
    """
    Analyze the impact of folder splits on collaborations.
    
    Args:
        collaboration_df: DataFrame containing collaboration data with columns:
            - item_id: ID of the folder (matches Folder ID in folder splits)
            - item_type, collaborator_id, collaborator_name, collaborator_login
            - collaborator_type, collaborator_permission, collaboration_id
        folder_splits_df: DataFrame containing folder split recommendations
        
    Returns:
        Dictionary containing collaboration analysis results:
        - maintained_collaborations: Collaborations that will be maintained
        - lost_collaborations: Collaborations that will be lost
        - remediation_actions: Actions needed to maintain access
    """
    # Validate required columns in collaboration data
    required_columns = ['item_id', 'item_type', 'collaborator_id', 'collaborator_name', 
                       'collaborator_login', 'collaborator_type', 'collaborator_permission', 
                       'collaboration_id']
    
    missing_columns = [col for col in required_columns if col not in collaboration_df.columns]
    if missing_columns:
        st.error(f"Missing required columns in collaboration data: {', '.join(missing_columns)}")
        return None
    
    # Create a mapping of folder IDs to their levels and service accounts
    folder_mapping = {}
    if 'Folder ID' in folder_splits_df.columns:
        for _, row in folder_splits_df.iterrows():
            folder_id = row['Folder ID']
            folder_mapping[folder_id] = {
                'level': row.get('Level', None),
                'service_account': row.get('Service Account', None),
                'folder_path': row.get('Folder Path', None),
                'folder_name': row.get('Folder Name', None)
            }
    
    # Initialize result containers
    maintained_collaborations = []
    lost_collaborations = []
    remediation_actions = []
    
    # Process each collaboration
    for _, collab in collaboration_df.iterrows():
        item_id = collab['item_id']
        
        # Check if this item is affected by folder splits
        if item_id in folder_mapping:
            folder_info = folder_mapping[item_id]
            
            # Apply collaboration rules
            if folder_info['level'] == 1:  # Root-level folder
                # For root-level folder splits: collaborations remain unchanged
                maintained_collaborations.append({
                    'item_id': item_id,
                    'folder_name': folder_info['folder_name'],
                    'folder_path': folder_info['folder_path'],
                    'collaborator_id': collab['collaborator_id'],
                    'collaborator_name': collab['collaborator_name'],
                    'collaborator_login': collab['collaborator_login'],
                    'collaborator_type': collab['collaborator_type'],
                    'collaborator_permission': collab['collaborator_permission'],
                    'collaboration_id': collab['collaboration_id'],
                    'reason': "Root-level folder split - collaboration maintained"
                })
            else:  # Subfolder
                # For subfolder splits:
                # - Collaborators added directly to the subfolder will see it in their root folders
                # - Collaborators added at parent-folder level or above will lose access
                
                # Check if this is a direct collaboration on the subfolder
                # (This is a simplified check - in a real implementation, you would need to 
                # determine if the collaboration was added directly to this folder or inherited)
                is_direct_collaboration = True  # Simplified assumption
                
                if is_direct_collaboration:
                    maintained_collaborations.append({
                        'item_id': item_id,
                        'folder_name': folder_info['folder_name'],
                        'folder_path': folder_info['folder_path'],
                        'collaborator_id': collab['collaborator_id'],
                        'collaborator_name': collab['collaborator_name'],
                        'collaborator_login': collab['collaborator_login'],
                        'collaborator_type': collab['collaborator_type'],
                        'collaborator_permission': collab['collaborator_permission'],
                        'collaboration_id': collab['collaboration_id'],
                        'reason': "Direct collaboration on subfolder - will see in root folders"
                    })
                else:
                    lost_collaborations.append({
                        'item_id': item_id,
                        'folder_name': folder_info['folder_name'],
                        'folder_path': folder_info['folder_path'],
                        'collaborator_id': collab['collaborator_id'],
                        'collaborator_name': collab['collaborator_name'],
                        'collaborator_login': collab['collaborator_login'],
                        'collaborator_type': collab['collaborator_type'],
                        'collaborator_permission': collab['collaborator_permission'],
                        'collaboration_id': collab['collaboration_id'],
                        'reason': "Parent-level collaboration - will lose access"
                    })
                    
                    # Add remediation action
                    remediation_actions.append({
                        'item_id': item_id,
                        'folder_name': folder_info['folder_name'],
                        'folder_path': folder_info['folder_path'],
                        'collaborator_id': collab['collaborator_id'],
                        'collaborator_name': collab['collaborator_name'],
                        'collaborator_login': collab['collaborator_login'],
                        'action': "Add direct collaboration to maintain access",
                        'service_account': folder_info['service_account']
                    })
        else:
            # Item not affected by folder splits
            maintained_collaborations.append({
                'item_id': item_id,
                'folder_name': "Unknown",  # Not in folder splits data
                'folder_path': "Unknown",
                'collaborator_id': collab['collaborator_id'],
                'collaborator_name': collab['collaborator_name'],
                'collaborator_login': collab['collaborator_login'],
                'collaborator_type': collab['collaborator_type'],
                'collaborator_permission': collab['collaborator_permission'],
                'collaboration_id': collab['collaboration_id'],
                'reason': "Folder not affected by splits - collaboration maintained"
            })
    
    # Create DataFrames from the results
    maintained_df = pd.DataFrame(maintained_collaborations) if maintained_collaborations else pd.DataFrame()
    lost_df = pd.DataFrame(lost_collaborations) if lost_collaborations else pd.DataFrame()
    remediation_df = pd.DataFrame(remediation_actions) if remediation_actions else pd.DataFrame()
    
    return {
        'maintained_collaborations': maintained_df,
        'lost_collaborations': lost_df,
        'remediation_actions': remediation_df
    }

def display_collaboration_results(collaboration_analysis):
    """
    Display the results of the collaboration analysis.
    
    Args:
        collaboration_analysis: Dictionary containing collaboration analysis results
    """
    if not collaboration_analysis:
        st.error("No collaboration analysis results to display.")
        return
    
    maintained_df = collaboration_analysis.get('maintained_collaborations')
    lost_df = collaboration_analysis.get('lost_collaborations')
    remediation_df = collaboration_analysis.get('remediation_actions')
    
    # Display summary statistics
    st.subheader("Collaboration Impact Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maintained Collaborations", len(maintained_df) if maintained_df is not None else 0)
    with col2:
        st.metric("Lost Collaborations", len(lost_df) if lost_df is not None else 0)
    with col3:
        st.metric("Remediation Actions", len(remediation_df) if remediation_df is not None else 0)
    
    # Display detailed results in tabs
    tabs = st.tabs(["Maintained Collaborations", "Lost Collaborations", "Remediation Actions"])
    
    with tabs[0]:
        if maintained_df is not None and not maintained_df.empty:
            st.dataframe(maintained_df)
            
            # Download CSV button
            csv = maintained_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="maintained_collaborations.csv">Download Maintained Collaborations CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No maintained collaborations to display.")
    
    with tabs[1]:
        if lost_df is not None and not lost_df.empty:
            st.dataframe(lost_df)
            
            # Download CSV button
            csv = lost_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="lost_collaborations.csv">Download Lost Collaborations CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No lost collaborations to display.")
    
    with tabs[2]:
        if remediation_df is not None and not remediation_df.empty:
            st.dataframe(remediation_df)
            
            # Download CSV button
            csv = remediation_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="remediation_actions.csv">Download Remediation Actions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No remediation actions to display.")
    
    # Create visualizations
    st.subheader("Collaboration Visualizations")
    
    # Create a pie chart showing maintained vs. lost collaborations
    if (maintained_df is not None and not maintained_df.empty) or (lost_df is not None and not lost_df.empty):
        maintained_count = len(maintained_df) if maintained_df is not None else 0
        lost_count = len(lost_df) if lost_df is not None else 0
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie([maintained_count, lost_count], 
               labels=['Maintained', 'Lost'], 
               autopct='%1.1f%%',
               colors=['#66b3ff', '#ff9999'],
               startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        ax.set_title('Collaboration Impact')
        st.pyplot(fig)
    
    # Create a bar chart showing collaborations by permission type
    if maintained_df is not None and not maintained_df.empty:
        if 'collaborator_permission' in maintained_df.columns:
            permission_counts = maintained_df['collaborator_permission'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(permission_counts.index, permission_counts.values, color='#66b3ff')
            ax.set_xlabel('Permission Type')
            ax.set_ylabel('Number of Collaborations')
            ax.set_title('Maintained Collaborations by Permission Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
    
    # Create a bar chart showing lost collaborations by permission type
    if lost_df is not None and not lost_df.empty:
        if 'collaborator_permission' in lost_df.columns:
            permission_counts = lost_df['collaborator_permission'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(permission_counts.index, permission_counts.values, color='#ff9999')
            ax.set_xlabel('Permission Type')
            ax.set_ylabel('Number of Collaborations')
            ax.set_title('Lost Collaborations by Permission Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
