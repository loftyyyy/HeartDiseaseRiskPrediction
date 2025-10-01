import pandas as pd;
def main():
    df = pd.read_csv('../data/heart_disease_risk_dataset_earlymed.csv')
    print(df.isnull().sum())

 

if __name__ == "__main__":
    main()
