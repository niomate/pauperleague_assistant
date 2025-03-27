from pauper_assistant.extract import (
    preprocess,
    extract_table,
    fix_names,
    REGISTERED_PLAYERS,
)


if __name__ == "__main__":
    image = preprocess("table.jpeg")
    df = extract_table(image)
    df = fix_names(df, REGISTERED_PLAYERS)
    print(df)
