Folder Splitting Recommendation Tool Prompt

Task Overview

Create an interactive web-based tool that analyzes folder ownership data from CSV files and provides recommendations for splitting content based on file count thresholds. The tool should identify users who own more than 500,000 files and recommend which folders to split to bring users below this threshold.

CSV Data Format

The input CSV contains the following columns:

Path: Folder path separated by "/"

folder_name: Name of the folder

folder_id: Integer ID of the folder

owner_email: Email of the user that owns the folder

updated_date: Date that the folder was updated

size_mb: Size of the folder in MB

file_count: Number of active files within the folder and all subfolders

level: Folder level in the folder tree hierarchy (1 is root level)

Key Requirements

1. Total File Count Calculation

Calculate total file count per user by summing only level 1 folders

File counts of level 2, 3, etc. are already included in level 1 counts

Example: If a user has two level 1 folders with 100,000 and 200,000 files, their total is 300,000

2. Level-Based Folder Prioritization

Process all folder levels sequentially (level 1, then level 2, then level 3, etc.)

Continue adding folders from each level until the user's total file count is reduced to 500,000 or less

Do not stop after finding candidates at a single level

Prioritize folders with file counts of 500,000 or less (ideal split candidates)

3. After Split Calculation

Show the total count after all splits for each specific user

Clearly label this as "After All Splits" in the interface

This should represent the final file count after implementing all recommended splits

4. Nested Folder Structure Handling

Recognize that subfolders' file counts are already included in parent folders' counts

Calculate "Direct File Count" (files directly in a folder, excluding subfolders)

Use this information to provide better context in recommendations

Algorithm Logic

Load and preprocess the CSV data

Calculate user statistics by summing only level 1 folders for each user

Identify users exceeding the 500,000 file threshold

For each user exceeding the threshold: a. Process all levels sequentially (level 1, then level 2, then level 3, etc.) b. At each level, identify folders with file count ≤ 500,000 (ideal candidates) c. Add these folders to recommendations, tracking the remaining file count d. Continue until user's total file count is reduced to 500,000 or less e. If no suitable folders are found, consider partial splits of larger folders

Generate visualizations showing before vs. after split file counts

Create an HTML report with all recommendations and visualizations

Implementation Details

Core Components

Data Analysis Module:

Load CSV data

Calculate user statistics

Identify nested folder relationships

Prioritize folders for splitting

Web Interface:

File upload form

Threshold setting option

Results display with visualizations

Downloadable report

Visualization Components:

Before vs. After All Splits comparison

Recommended folders to move

Total vs. Direct file counts

User summary statistics

Technical Requirements

Use Python for backend processing (pandas, matplotlib)

Create a Flask web application for the interface

Implement responsive design for the web interface

Provide clear documentation of the algorithm logic

Include detailed comments in the code

Output Format

The tool should generate:

Interactive web interface for uploading CSV files

Detailed HTML report with recommendations

Visualizations showing before and after scenarios

Downloadable results package with all data and visualizations

Evaluation Criteria

The solution will be evaluated based on:

Correct calculation of total file count (summing only level 1 folders)

Proper implementation of level-based folder prioritization

Accurate display of "After All Splits" totals

Handling of nested folder structures

Quality and clarity of visualizations and reports

Overall usability of the web interface

This prompt captures all the amendments and clarifications we've made throughout our development process, providing a clear guide for implementing the folder splitting recommendation tool.



