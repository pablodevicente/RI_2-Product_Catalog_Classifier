import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import json


def load_results(file_path):
    """
    Load the CSV file and perform type conversion for the columns
    that contain dictionary or list data stored as strings.
    """
    df = pd.read_csv(file_path)

    # If your CSV columns that store dictionaries (or lists) were written as strings,
    # you can convert them back using ast.literal_eval. Adjust the column names as needed.
    dict_columns = ['grid_search_best_params', 'train_class_distribution',
                    'test_class_distribution', 'confusion_matrix']

    for col in dict_columns:
        if col in df.columns:
            # Sometimes the conversion might fail for rows not having eval-parsable strings.
            # Use a function to safely convert these.
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else x)

    return df


def plot_cross_val_scores(df):
    """
    Plot cross-validation scores with error bars (standard deviation).
    Assumes columns 'name', 'cross_val_accuracy', and 'cross_val_std' exist.
    """
    plt.figure(figsize=(10, 6))
    # Again, assuming one record per model or that these have been aggregated appropriately.
    ax = plt.gca()
    ax.errorbar(x=df['file_name'], y=df['cross_val_accuracy'], yerr=df['cross_val_std'],
                fmt='o', capsize=5, linestyle='None', marker='s', markersize=8)
    plt.title("Cross-Validation Accuracy with Standard Deviation")
    plt.xlabel("Model Name")
    plt.ylabel("Cross-Validation Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_class_distributions(df, model_name):
    """
    For a given model_name, plot side-by-side bar charts comparing the
    training and testing class distributions.
    It assumes that the columns 'train_class_distribution' and
    'test_class_distribution' contain dictionaries.
    """
    # Filter the dataframe for the given model name
    model_df = df[df['file_name'] == model_name]
    if model_df.empty:
        print(f"No data found for model '{model_name}'")
        return


    # Step 3: Access dictionary entries using keys
    train_dist = df['train_class_distribution']
    test_dist = df['test_class_distribution']

    # Get sorted keys for consistent ordering.
    classes = sorted(set(train_dist.keys()).union(set(test_dist.keys())))

    train_counts = [train_dist.get(c, 0) for c in classes]
    test_counts = [test_dist.get(c, 0) for c in classes]

    x = range(len(classes))

    plt.figure(figsize=(10, 6))
    plt.bar([p - 0.15 for p in x], train_counts, width=0.3, label='Training Set')
    plt.bar([p + 0.15 for p in x], test_counts, width=0.3, label='Test Set')

    plt.xticks(x, classes)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title(f"Class Distribution Comparison for {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_for_model(df, model_name):
    """
    Extract the confusion matrix for a particular model and plot it as a heatmap.
    Assumes that the confusion matrix is stored under the column 'confusion_matrix'
    and is a list-of-lists or 2D array.
    """
    # Filter the dataframe for the given model name
    model_df = df[df['file_name'] == model_name]
    if model_df.empty:
        print(f"No data found for model '{model_name}'")
        return

    conf_matrix = df['confusion_matrix']

    # Plot the confusion matrix using seaborn heatmap.
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

def plot_accuracy(df):
    # Set the visual style
    sns.set(style="whitegrid", font_scale=1.1)

    # To ensure bars appear in a consistent order, sort the DataFrame.
    df_sorted = df.sort_values(by=["classifier", "file_name"]).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    # Create a grouped bar plot:
    # - x-axis: classifier
    # - y-axis: accuracy
    # - hue: file_name so that each file gets a unique color.
    ax = sns.barplot(data=df_sorted, x='classifier', y='accuracy', hue='file_name', palette='Set2', dodge=True)

    # Annotate each bar with its accuracy value.
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(f'{height:.2f}',
                    (patch.get_x() + patch.get_width() / 2, height),
                    ha='center', va='bottom',
                    fontsize=10,
                    color='black',
                    xytext=(0, 3),
                    textcoords='offset points')

    # The order of bars in ax.patches should match the order of the rows in df_sorted.
    patches = ax.patches

    # For each row in the sorted DataFrame, add an error bar showing cross_val_std.
    # (Assumes each bar corresponds to one row, in the same order as df_sorted.)
    for i, row in df_sorted.iterrows():
        patch = patches[i]
        # Calculate the center x-coordinate of the bar.
        x_center = patch.get_x() + patch.get_width() / 2
        y_val = patch.get_height()
        std_value = row['cross_val_std']
        # Add an error bar (using fmt='none' so that only the error bar is drawn).
        ax.errorbar(x_center, y_val, yerr=std_value, fmt='none', color='black', capsize=4)

    # Configure plot title, labels, and legend.
    plt.title("Accuracy Comparison by File and Classifier (with Cross Val STD)", fontsize=16)
    plt.xlabel(" ", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1

    # Place the legend at the bottom of the plot.
    n_files = df["file_name"].nunique()
    plt.legend(title=" ", loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=n_files)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == '__main__':
    # Replace 'results.csv' with the path to your CSV file.
    df = pd.read_csv("../02-data/04-Classifier/classifiers-4-50-4-150.csv")

    plot_accuracy(df)

    # Plot cross-validation accuracies with standard deviation.
    # (Ensure your CSV includes these columns)
    #if 'cross_val_accuracy' in df.columns and 'cross_val_std' in df.columns:
    #    plot_cross_val_scores(df)

    # For a specified model name, plot the class distributions and confusion matrix.
    #example_model = 'fasttext-4-150.pkl'  # Replace with an actual model name from your CSV.
    #plot_class_distributions(df, example_model)
    #plot_confusion_matrix_for_model(df, example_model)
