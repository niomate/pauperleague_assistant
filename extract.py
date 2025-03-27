from img2table.document import Image
from img2table.ocr import PaddleOCR
import cv2
import io
import json
import os
import pandas as pd

REGISTERED_PLAYERS = [
    "Tigerente Till",
    "Max Backes",
    "Anna Larisch",
    "Jana Gusenburger",
    "Mira Kubiczak",
    "Mathias Behre",
    "Alexis Darras",
    "Alexander Ponticello",
    "Marc Schmitz",
]


def normalize_name(s):
    return s.replace(".", "").replace(" ", "").lower()


def preprocess(img_file):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 140, 255, cv2.THRESH_TOZERO)
    img = cv2.bitwise_not(img)
    return img


def extract_table(image):
    ret, encoded = cv2.imencode(".jpg", image)
    buf = io.BytesIO(encoded)

    ocr = PaddleOCR()
    doc = Image(buf, detect_rotation=False)

    df = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        implicit_columns=True,
        borderless_tables=True,
        min_confidence=80,
    )[0].df

    df.columns = ["Rank", "Name", "Points", "W-L-D", "OMW%"]
    df = df.reset_index()
    df = df.drop(df.index[0])

    wld = df["W-L-D"].str.split(pat="-", expand=True).astype(int)
    wld.columns = ["W", "L", "D"]

    del df["Rank"]
    del df["index"]
    del df["W-L-D"]
    del df["OMW%"]

    df = df.join(wld)
    df["Points"] = 3 * df["W"] + df["D"]

    return df


def fix_names(df, registered_players):
    matched_names = []
    for index, row in df.iterrows():
        n = df.at[index, "Name"]
        for s in registered_players:
            if n is not None and normalize_name(n) in normalize_name(s):
                matched_names.append(s)
                break
        else:
            new_name = input(
                f"Could not find registered player {n} in row {row}. Please enter the correct name: "
            )
            matched_names.append(new_name)

    df["Name"] = matched_names
    return df


def update_standings(latest_rankings):
    if os.path.exists("rankings.json"):
        with open("rankings.json", "r") as f:
            standings = json.load(f)
    else:
        standings = []

    standings.append(json.loads(latest_rankings.to_json(orient="records")))

    with open("rankings.json", "w") as f:
        json.dump(standings, f)

    return standings


def calculate_standings(standings):
    combined_standings = pd.DataFrame()
    for fnm in standings:
        df = pd.DataFrame(fnm)
        combined_standings = pd.concat([combined_standings, df])

    combined_standings = combined_standings.groupby("Name").sum()

    combined_standings["Rounds Played"] = (
        combined_standings["W"] + combined_standings["L"] + combined_standings["D"]
    )

    combined_standings["Average Points"] = (
        combined_standings["Points"] / combined_standings["Rounds Played"]
    )

    combined_standings.sort_values(by="Average Points", ascending=False)
    return combined_standings


if __name__ == "__main__":
    image = preprocess("table.jpeg")
    df = extract_table(image)
    df = fix_names(df, REGISTERED_PLAYERS)
    print(df)
