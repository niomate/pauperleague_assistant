import io
import cv2
from img2table.document import Image
from img2table.ocr import PaddleOCR


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
