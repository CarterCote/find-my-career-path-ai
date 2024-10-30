from data_loader import CareerDataLoader

def main():
    # Create an instance of the data loader
    loader = CareerDataLoader()

    # Download the dataset
    dataset_path = loader.download_dataset()
    print(f"Dataset downloaded to: {dataset_path}")

    # Load the job postings
    df = loader.load_job_postings()
    return df

if __name__ == "__main__":
    main()