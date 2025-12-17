from datasets import load_dataset
import pandas as pd

DATASET = "philschmid/germeval18"
SHOW_TEXT_SAMPLES = False  # bewusst aus, weil Texte potenziell beleidigend sein können

def main():
    ds = load_dataset(DATASET)  # lädt vom Hugging Face Hub :contentReference[oaicite:3]{index=3}
    print("Splits:", list(ds.keys()))
    print("Train size:", len(ds["train"]))
    print("Test size :", len(ds["test"]))

    print("\nColumns:", ds["train"].column_names)
    # Erwartet: text, binary, multi :contentReference[oaicite:4]{index=4}

    # Kleine Stats/Checks
    train_df = ds["train"].to_pandas()

    print("\nBinary label values:")
    print(train_df["binary"].value_counts())

    print("\nMulti label values (4-class):")
    print(train_df["multi"].value_counts())

    lbr_count = train_df["text"].str.contains(r"\|LBR\|", regex=True).sum()
    print(f"\nTexts containing '|LBR|': {lbr_count} / {len(train_df)}")

    if SHOW_TEXT_SAMPLES:
        print("\nSample rows (truncated):")
        for i in range(3):
            t = train_df.loc[i, "text"].replace("\n", " ")
            print(f"- multi={train_df.loc[i,'multi']}, binary={train_df.loc[i,'binary']}, text[:120]={t[:120]!r}")

if __name__ == "__main__":
    main()
