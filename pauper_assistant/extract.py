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


