import pandas as pd

try:
    df = pd.read_excel(r'C:/Users/Varad/Downloads/TDC_D2C.xlsx')
    print("File loaded successfully.")

    # Define the groups of columns to merge
    visual_hook_cols = ['Visual Hook (0-3s)', 'visual_hook_0_3s', 'visual_hook', 'visual_hook_0_to_3s']
    copy_hook_cols = ['Copy Hook (0-3s)', 'copy_hook_0_3s', 'copy_hook', 'copy_hook_0_to_3s']
    problem_setup_cols = ['Problem Setup', 'problem_setup']
    product_mot_cols = ['Product MOT', 'product_mot']
    hero_claim_cols = ['Hero Claim', 'hero_claim']
    rtbs_cols = ['RTBs', 'rtbs']
    product_format_cols = ['Product Format', 'product_format']
    application_demo_cols = ['Application Demo', 'application_demo']
    visual_style_cols = ['Visual Style', 'visual_style']
    cta_cols = ['CTA', 'cta']
    ad_structure_cols = ['Ad Structure', 'ad_structure']
    other_notables_cols = ['Other Notables', 'other_notables']

    # Merge the columns
    df['Visual Hook Combined'] = df[visual_hook_cols].bfill(axis=1).iloc[:, 0]
    df['Copy Hook Combined'] = df[copy_hook_cols].bfill(axis=1).iloc[:, 0]
    df['Problem Setup Combined'] = df[problem_setup_cols].bfill(axis=1).iloc[:, 0]
    df['Product MOT Combined'] = df[product_mot_cols].bfill(axis=1).iloc[:, 0]
    df['Hero Claim Combined'] = df[hero_claim_cols].bfill(axis=1).iloc[:, 0]
    df['RTBs Combined'] = df[rtbs_cols].bfill(axis=1).iloc[:, 0]
    df['Product Format Combined'] = df[product_format_cols].bfill(axis=1).iloc[:, 0]
    df['Application Demo Combined'] = df[application_demo_cols].bfill(axis=1).iloc[:, 0]
    df['Visual Style Combined'] = df[visual_style_cols].bfill(axis=1).iloc[:, 0]
    df['CTA Combined'] = df[cta_cols].bfill(axis=1).iloc[:, 0]
    df['Ad Structure Combined'] = df[ad_structure_cols].bfill(axis=1).iloc[:, 0]
    df['Other Notables Combined'] = df[other_notables_cols].bfill(axis=1).iloc[:, 0]


    # Drop the original columns
    df.drop(columns=visual_hook_cols + copy_hook_cols + problem_setup_cols + product_mot_cols + hero_claim_cols +
            rtbs_cols + product_format_cols + application_demo_cols + visual_style_cols + cta_cols +
            ad_structure_cols + other_notables_cols, inplace=True)

    # Save the modified dataframe to a new Excel file
    output_path = r'C:/Users/Varad/Downloads/TDC_D2C_combined.xlsx'
    df.to_excel(output_path, index=False)
    print(f"Successfully combined the columns and saved the new file to {output_path}")

except FileNotFoundError:
    print("Error: The file was not found at the specified path.")
except Exception as e:
    print(f"An error occurred: {e}")