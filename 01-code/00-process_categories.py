from aux_download_pdfs import download_pdfs_from_page, check_pdfs_in_folder, check_wifi_connection, create_urls
import pandas as pd

# Define the variable for the first part of the path
base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'

def main(categories,clean=False):

    if check_wifi_connection():
        print("Wi-Fi is connected.")
    else:
        print("Wi-Fi is not connected.")

   # Download pdfs from each link in the category
    for label, urls in categories.items():
        save_folder = base_path + label
        print(f"Processing: {label}")

        download_pdfs_from_page(urls, save_folder)

    if clean:
        # Delete corrupted pdfs
        for label, urls in categories.items():
            save_folder = base_path + label
            print(f"Processing corrupted pdfs: {label}")

            check_pdfs_in_folder(save_folder)

if __name__ == "__main__":
    urls = [
        "https://www.digikey.com/en/products/filter/controllers/controller-accessories/816",
            "https://www.digikey.com/en/products/filter/industrial-lighting/task-lighting/1061",
            "https://www.digikey.com/en/products/filter/batteries-non-rechargeable-primary/90",
            "https://www.digikey.com/en/products/filter/batteries-rechargeable-secondary/91",
            "https://www.digikey.com/en/products/filter/battery-chargers/85",
            "https://www.digikey.be/en/products/filter/battery-packs/89",
            "https://www.digikey.be/en/products/filter/microphones/158",
            "https://www.digikey.be/en/products/filter/speakers/156",
            "https://www.digikey.com/en/products/filter/alarms-buzzers-and-sirens/157",
            "https://www.digikey.be/en/products/filter/anti-static-esd-bags-materials/605",
            "https://www.digikey.be/en/products/filter/boxes/594",
            "https://www.digikey.be/en/products/filter/card-racks/588",
            "https://www.digikey.be/en/products/filter/rack-accessories/598",
            "https://www.digikey.be/en/products/filter/circular-cable-assemblies/448",
            "https://www.digikey.be/en/products/filter/rack-accessories/598",
            "https://www.digikey.be/en/products/filter/circular-cable-assemblies/448",
            "https://www.digikey.be/en/products/filter/fiber-optic-cables/449",
            "https://www.digikey.be/en/products/filter/jumper-wires-pre-crimped-leads/453",
            "https://www.digikey.be/en/products/filter/coaxial-cables-rf/475",
            "https://www.digikey.be/en/products/filter/multiple-conductor-cables/473",
            "https://www.digikey.be/en/products/filter/single-conductor-cables-hook-up-wire/474",
            "https://www.digikey.be/en/products/filter/accessories/479",
            "https://www.digikey.be/en/products/filter/cable-ties-and-zip-ties/482",
            "https://www.digikey.be/en/products/filter/protective-hoses-solid-tubing-sleeving/480",
            "https://www.digikey.be/en/products/filter/aluminum-polymer-capacitors/69",
            "https://www.digikey.be/en/products/filter/aluminum-electrolytic-capacitors/58",
            "https://www.digikey.be/en/products/filter/tantalum-capacitors/59",
            "https://www.digikey.be/en/products/filter/circuit-breakers/143",
            "https://www.digikey.be/en/products/filter/electrical-specialty-fuses/155",
            "https://www.digikey.be/en/products/category/transient-voltage-suppressors-tvs/2040",
            "https://www.digikey.be/en/products/filter/printers-label-makers/887",
            "https://www.digikey.be/en/products/category/ac-power-connectors/2026",
            "https://www.digikey.be/en/products/filter/backplane-connectors/backplane-connector-housings/372",
            "https://www.digikey.be/en/products/category/banana-and-tip-connectors/2001",
            "https://www.digikey.be/en/products/category/barrel-connectors/2002",
            "https://www.digikey.be/en/products/category/blade-type-power-connectors/2003"
    ]

    categories = create_urls(urls)
    # Call the main function and get the processed DataFrame
    main(categories)