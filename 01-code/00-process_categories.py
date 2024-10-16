from aux_download_pdfs import download_pdfs_from_page, check_pdfs_in_folder
import pandas as pd

# Function to process a group of URLs under a given label
def process_category(label, urls, df_statistics):
    save_folder = f'/media/pablo/windows_files/00 - Master/05 - Research&Thesis/02-data/pdfs/{label}'
    print(f"Processing: {label}")
    for url in urls:
        df_statistics = download_pdfs_from_page(label, url, save_folder, df_statistics)

    return df_statistics

# Function to check through folders and delete corrupted files
def process_corrupted(label, urls, df_statistics):
    save_folder = f'/media/pablo/windows_files/00 - Master/05 - Research&Thesis/02-data/pdfs/{label}'
    print(f"Processing corrupted pdfs: {label}")
    df_statistics = check_pdfs_in_folder(save_folder,df_statistics)
    
    return df_statistics

def main(categories):
    # Create an empty DataFrame with the required columns
    df_statistics = pd.DataFrame(columns=[
        'Label', 'Total_found', 'Downloaded_Valid', 'Downloaded_Invalid', 'Downloaded_Exceptions', 
        'Cleanup_Valid', 'Cleanup_Invalid', 'Cleanup_Exceptions'
    ])

    # Process each category
    for label, urls in categories.items():
        df_statistics = process_category(label, urls, df_statistics)


    # Delete corrupted pdfs
    for label, urls in categories.items():
        # df_statistics = process_corrupted(label, urls,df_statistics)
        pass

    # Return or display the resulting DataFrame
    return df_statistics

if __name__ == "__main__":
    # Dictionary to store categories and their URLs
    categories = {
        "Controller-accessories": [
            'https://www.digikey.com/en/products/filter/controllers/controller-accessories/816?s=N4IgrCBcoA5QjAGhDOl4AYMF9tA',
            'https://www.digikey.com/en/products/filter/controllers/controller-accessories/816?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ'
        ],
        "Lighting-equipment": [
            'https://www.digikey.com/en/products/filter/industrial-lighting/task-lighting/1061?s=N4IgrCBcoA5QjAGhDOl4AYMF9tA',
            'https://www.digikey.com/en/products/filter/industrial-lighting/task-lighting/1061?s=N4IgrCBcoA5WAaEM6QIwAYMF9tA',
            'https://www.digikey.com/en/products/filter/industrial-lighting/task-lighting/1061?s=N4IgrCBcoA5QzAJgDQhnSBGADNgvnkA',
            'https://www.digikey.com/en/products/filter/industrial-lighting/task-lighting/1061?s=N4IgrCBcoA5QzABgDQhnSBGRiC%2Bug'
        ],
        "Non-Rechargeable-batteries": [
            "https://www.digikey.com/en/products/filter/batteries-non-rechargeable-primary/90"
        ],
        "Rechargeable-batteries": [
            "https://www.digikey.com/en/products/filter/batteries-rechargeable-secondary/91"
        ],
        "Battery-chargers": [
            "https://www.digikey.com/en/products/filter/battery-chargers/85"
        ],
        "Alarms-and-sirens": [
            "https://www.digikey.com/en/products/filter/alarms-buzzers-and-sirens/157",
            "https://www.digikey.com/en/products/filter/alarms-buzzers-and-sirens/157?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ]
    }

    # Call the main function and get the processed DataFrame
    df_statistics = main(categories)

    output_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/02-data/pdfs/df_statistics.xlsx'
    df_statistics.to_excel(output_path, index=False)