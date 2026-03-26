import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import argparse

def analyze_closeness_single_sheet(df):
    """
    Analyze which prediction (confounded or unconfounded) is closer to the outcomes
    for a single dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with confounded, unconfounded, noised, and noiseless columns
    
    Returns:
    --------
    tuple : (processed_df, summary_dict)
    """
    # Calculate absolute differences for all combinations
    df['noised_confounded_diff'] = np.abs(df['noised'] - df['confounded'])
    df['noised_unconfounded_diff'] = np.abs(df['noised'] - df['unconfounded'])
    df['noiseless_confounded_diff'] = np.abs(df['noiseless'] - df['confounded'])
    df['noiseless_unconfounded_diff'] = np.abs(df['noiseless'] - df['unconfounded'])
    
    # Determine which is closer for noised outcome
    df['noised_closer'] = df.apply(
        lambda row: 'confounded' if row['noised_confounded_diff'] < row['noised_unconfounded_diff'] 
        else 'unconfounded', 
        axis=1
    )
    
    # Determine which is closer for noiseless outcome
    df['noiseless_closer'] = df.apply(
        lambda row: 'confounded' if row['noiseless_confounded_diff'] < row['noiseless_unconfounded_diff'] 
        else 'unconfounded', 
        axis=1
    )
    
    # Count occurrences
    noised_counts = df['noised_closer'].value_counts()
    noiseless_counts = df['noiseless_closer'].value_counts()
    
    # Create summary statistics
    summary = {
        'total_observations': len(df),
        'noised_outcome': {
            'confounded_closer': int(noised_counts.get('confounded', 0)),
            'unconfounded_closer': int(noised_counts.get('unconfounded', 0))
        },
        'noiseless_outcome': {
            'confounded_closer': int(noiseless_counts.get('confounded', 0)),
            'unconfounded_closer': int(noiseless_counts.get('unconfounded', 0))
        }
    }
    
    # Keep all columns including the new difference columns
    output_columns = [
        'treatment', 'yf','ycf','mu1','mu0','confounded', 'unconfounded', 'confidence', 'RPCE_CATE', 
        'noised', 'noiseless', 
        'noised_confounded_diff', 'noised_unconfounded_diff',
        'noiseless_confounded_diff', 'noiseless_unconfounded_diff',
        'noised_closer', 'noiseless_closer'
    ]
    df_output = df[output_columns].copy()
    
    return df_output, summary


def analyze_all_sheets(input_file, output_file):
    """
    Analyze all sheets in the Excel file.
    
    Parameters:
    -----------
    input_file : str
        Path to the input Excel file
    output_file : str
        Path to save the output Excel file with new columns
    
    Returns:
    --------
    dict : Summary statistics for all sheets
    """
    # Read all sheets
    excel_file = pd.ExcelFile(input_file)
    all_summaries = {}
    
    # Create a writer object
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name in excel_file.sheet_names:
            print(f"Processing {sheet_name}...")
            
            # Read the sheet
            df = pd.read_excel(input_file, sheet_name=sheet_name)
            
            # Analyze the sheet
            df_output, summary = analyze_closeness_single_sheet(df)
            
            # Store summary
            all_summaries[sheet_name] = summary
            
            # Write to output file
            df_output.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return all_summaries


def create_summary_report(all_summaries):
    """
    Create a summary dataframe from all sheet summaries.
    
    Parameters:
    -----------
    all_summaries : dict
        Dictionary containing summaries for all sheets
    
    Returns:
    --------
    pd.DataFrame : Summary report
    """
    summary_data = []
    
    for sheet_name, summary in all_summaries.items():
        summary_data.append({
            'Sheet': sheet_name,
            'Total_Observations': summary['total_observations'],
            'Noised_Confounded_Closer': summary['noised_outcome']['confounded_closer'],
            'Noised_Unconfounded_Closer': summary['noised_outcome']['unconfounded_closer'],
            'Noiseless_Confounded_Closer': summary['noiseless_outcome']['confounded_closer'],
            'Noiseless_Unconfounded_Closer': summary['noiseless_outcome']['unconfounded_closer']
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add totals row
    totals = {
        'Sheet': 'TOTAL',
        'Total_Observations': summary_df['Total_Observations'].sum(),
        'Noised_Confounded_Closer': summary_df['Noised_Confounded_Closer'].sum(),
        'Noised_Unconfounded_Closer': summary_df['Noised_Unconfounded_Closer'].sum(),
        'Noiseless_Confounded_Closer': summary_df['Noiseless_Confounded_Closer'].sum(),
        'Noiseless_Unconfounded_Closer': summary_df['Noiseless_Unconfounded_Closer'].sum()
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([totals])], ignore_index=True)
    
    return summary_df


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze IHDP output Excel file and generate summary reports'
    )
    
    parser.add_argument(
        '-i', '--input-file',
        default='/mnt/user-data/uploads/IHDP_output.xlsx',
        help='Path to input Excel file (default: /mnt/user-data/uploads/IHDP_output.xlsx)'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        default='/mnt/user-data/outputs/IHDP_output_analyzed.xlsx',
        help='Path to output Excel file (default: /mnt/user-data/outputs/IHDP_output_analyzed.xlsx)'
    )
    
    parser.add_argument(
        '-s', '--summary-file',
        default='/mnt/user-data/outputs/IHDP_summary_report.xlsx',
        help='Path to summary report file (default: /mnt/user-data/outputs/IHDP_summary_report.xlsx)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use the arguments
    input_file = args.input_file
    output_file = args.output_file
    summary_file = args.summary_file
    
    
    print("=" * 80)
    print("IHDP OUTPUT ANALYSIS - PROCESSING ALL SHEETS")
    print("=" * 80)
    print()
    
    # Run analysis on all sheets
    all_summaries = analyze_all_sheets(input_file, output_file)
    
    # Create summary report
    summary_df = create_summary_report(all_summaries)
    
    # Save summary report
    summary_df.to_excel(summary_file, index=False)
    
    print()
    print("=" * 80)
    print("SUMMARY REPORT - ALL SHEETS")
    print("=" * 80)
    print()
    print(summary_df.to_string(index=False))
    
    # Calculate overall statistics
    totals = summary_df[summary_df['Sheet'] == 'TOTAL'].iloc[0]
    
    print()
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"\nTotal Observations Across All Sheets: {totals['Total_Observations']}")
    print()
    print("NOISED OUTCOME:")
    print(f"  Confounded closer:   {totals['Noised_Confounded_Closer']} times " +
          f"({totals['Noised_Confounded_Closer']/totals['Total_Observations']*100:.1f}%)")
    print(f"  Unconfounded closer: {totals['Noised_Unconfounded_Closer']} times " +
          f"({totals['Noised_Unconfounded_Closer']/totals['Total_Observations']*100:.1f}%)")
    print()
    print("NOISELESS OUTCOME:")
    print(f"  Confounded closer:   {totals['Noiseless_Confounded_Closer']} times " +
          f"({totals['Noiseless_Confounded_Closer']/totals['Total_Observations']*100:.1f}%)")
    print(f"  Unconfounded closer: {totals['Noiseless_Unconfounded_Closer']} times " +
          f"({totals['Noiseless_Unconfounded_Closer']/totals['Total_Observations']*100:.1f}%)")
    
    print()
    print("=" * 80)
    print("COLUMN STRUCTURE")
    print("=" * 80)
    print("\nThe output file contains the following columns:")
    print("  1. confounded")
    print("  2. unconfounded")
    print("  3. confidence")
    print("  4. RPCE_CATE")
    print("  5. noised")
    print("  6. noiseless")
    print("  7. noised_confounded_diff (|noised - confounded|)")
    print("  8. noised_unconfounded_diff (|noised - unconfounded|)")
    print("  9. noiseless_confounded_diff (|noiseless - confounded|)")
    print(" 10. noiseless_unconfounded_diff (|noiseless - unconfounded|)")
    print(" 11. noised_closer (which is closer to noised)")
    print(" 12. noiseless_closer (which is closer to noiseless)")
    
    print()
    print("=" * 80)
    print(f"Analyzed data saved to: {output_file}")
    print(f"Summary report saved to: {summary_file}")
    print("=" * 80)
    
    # Show sample of first sheet
    print()
    print("=" * 80)
    print("SAMPLE OUTPUT (First 5 rows of Tensor_1)")
    print("=" * 80)
    sample_df = pd.read_excel(output_file, sheet_name='Tensor_1', nrows=5)
    print(sample_df.to_string(index=False))
