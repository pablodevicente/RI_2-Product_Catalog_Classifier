def create_urls (urls):
    def create_urls_and_append_to_catalog(base_url, catalog):
        # Extract the label from the base URL (assumed to be between "filter/" and the next "/")
        label_start = base_url.find("/filter/") + len("/filter/")
        label_end = base_url.find("/", label_start)
        label = base_url[label_start:label_end]

        # Create the additional URLs by appending different query parameters
        additional_queries = [
            "?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ]
        
        # Create the list of full URLs
        new_urls = [base_url] + [base_url + query for query in additional_queries]

        # Check if the label exists in the catalog
        if label in catalog:
            # Append the new URLs to the existing list
            catalog[label].extend(new_urls)
        else:
            # Create a new entry for the label if it doesn't exist
            catalog[label] = new_urls


    catalog = { }

    # Iterate over the list and call the function for each URL
    for url in urls:
        create_urls_and_append_to_catalog(url, catalog)

    return catalog