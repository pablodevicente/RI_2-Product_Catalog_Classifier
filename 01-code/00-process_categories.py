from aux_download_pdfs import download_pdfs_from_page, check_pdfs_in_folder, check_wifi_connection
import pandas as pd

# Define the variable for the first part of the path
base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'

def main(categories):

    if check_wifi_connection():
        print("Wi-Fi is connected.")
    else:
        print("Wi-Fi is not connected.")

   # Download pdfs from each link in the category
    for label, urls in categories.items():
        save_folder = base_path + label
        print(f"Processing: {label}")

        download_pdfs_from_page(urls, save_folder)

    # Delete corrupted pdfs
    for label, urls in categories.items():
        save_folder = base_path + label
        print(f"Processing corrupted pdfs: {label}")

        check_pdfs_in_folder(save_folder)

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
            "https://www.digikey.be/en/products/filter/batteries-non-rechargeable-primary/90?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/batteries-non-rechargeable-primary/90?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
        ],
        "Rechargeable-batteries": [
            "https://www.digikey.com/en/products/filter/batteries-rechargeable-secondary/91"
            "https://www.digikey.be/en/products/filter/batteries-rechargeable-secondary/91?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/batteries-rechargeable-secondary/91?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
        ],
        "Battery-chargers": [
            "https://www.digikey.com/en/products/filter/battery-chargers/85"
            "https://www.digikey.be/en/products/filter/battery-chargers/85?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/battery-chargers/85?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
        ],
        "Battery-packs" : [
            "https://www.digikey.be/en/products/filter/battery-packs/89"
            "https://www.digikey.be/en/products/filter/battery-packs/89?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/battery-packs/89?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
        ],
        "Microphones": [
            #"https://www.digikey.be/en/products/filter/microphones/158"    #for some reason it gets stuck at 1 with this two links
            #"https://www.digikey.be/en/products/filter/microphones/158?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/microphones/158?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/microphones/158?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ],
        "Speakers": [
            "https://www.digikey.be/en/products/filter/speakers/156"
            "https://www.digikey.be/en/products/filter/speakers/156?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/speakers/156?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/speakers/156?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ],
        "Alarms-and-sirens": [
            "https://www.digikey.com/en/products/filter/alarms-buzzers-and-sirens/157",
            "https://www.digikey.com/en/products/filter/alarms-buzzers-and-sirens/157?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ],
        "Capacitors": [
            "https://www.digikey.be/en/products/filter/aluminum-polymer-capacitors/69"
            "https://www.digikey.be/en/products/filter/aluminum-polymer-capacitors/69?s=N4IgrCBcoA5QjAGhDOl4AYMF9tA",
            "https://www.digikey.be/en/products/filter/aluminum-polymer-capacitors/69?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
        ],
        "Anti-static, ESD Bags, Materials": [
            "https://www.digikey.be/en/products/filter/anti-static-esd-bags-materials/605"    
            "https://www.digikey.be/en/products/filter/anti-static-esd-bags-materials/605?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/anti-static-esd-bags-materials/605?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/anti-static-esd-bags-materials/605?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ],
        "Boxes" : [
            "https://www.digikey.be/en/products/filter/boxes/594"
            "https://www.digikey.be/en/products/filter/boxes/594?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/boxes/594?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/boxes/594?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/boxes/594?s=N4IgrCBcoA5WAaEM6QIwAYMF9tA"
        ],
        "Card-Racks" : [
            "https://www.digikey.be/en/products/filter/card-racks/588"
            "https://www.digikey.be/en/products/filter/card-racks/588?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
        ],
        "Rack-Accessories" : [
            "https://www.digikey.be/en/products/filter/rack-accessories/598"
            "https://www.digikey.be/en/products/filter/rack-accessories/598?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/rack-accessories/598?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/rack-accessories/598?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
            "https://www.digikey.be/en/products/filter/rack-accessories/598?s=N4IgrCBcoA5WAaEM6QIwAYMF9tA"
        ],
        "Circular-Cable-Assemblies" : [
            "https://www.digikey.be/en/products/filter/circular-cable-assemblies/448"
            "https://www.digikey.be/en/products/filter/circular-cable-assemblies/448?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ"
        ]
    }

    # Call the main function and get the processed DataFrame
    main(categories)


