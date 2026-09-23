"""

PYTHONIOENCODING=utf-8 python3 generate_wordstr_box.py
  -i "/<dir>/urn+nbn+de+gbv+3+5-1192015415-166770450197611-12_p211_line_1696937461725_511.tif"
  -t "/<dir>/urn+nbn+de+gbv+3+5-1192015415-166770450197611-12_p211_line_1696937461725_511.gt.txt" >
     "/<dir>/urn+nbn+de+gbv+3+5-1192015415-166770450197611-12_p211_line_1696937461725_511.box"
tesseract "/<dir>/urn+nbn+de+gbv+3+5-1192015415-166770450197611-12_p211_line_1696937461725_511.tif"
          "/<dir>/urn+nbn+de+gbv+3+5-1192015415-166770450197611-12_p211_line_1696937461725_511" --psm 13 lstm.train

a) box files
        cf. https://github.com/tesseract-ocr/tesstrain/generate_wordstr_box.py if RTL using bidi.algorithm
        +
        cf. https://github.com/tesseract-ocr/tesstrain/generate_line_box.py

b) lsmt files:
        cf. https://github.com/tesseract-ocr/tesstrain/Makefile
        %.lstmf: %.tif %.box
                tesseract "$<" $* --psm 13 lstm.train

"""

import io
import sys
import unicodedata

from pathlib import Path

import bidi.algorithm
import PIL.Image


def _write_box(txt_path: Path, image_path: Path, out_path: Path):
    width, height = PIL.Image.open(image_path).size
    with io.open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        if len(lines) != 1:
            raise ValueError(f"ERROR: GT file {txt_path} must not have 1+ lines: {len(lines)}")
    line = unicodedata.normalize("NFC", lines[0].strip())
    if line:
        for i in range(1, len(line)):
            char = line[i]
            prev_char = line[i - 1]
            if unicodedata.combining(char):
                print("%s 0 0 %d %d 0" % ((prev_char + char), width, height))
            elif not unicodedata.combining(prev_char):
                print("%s 0 0 %d %d 0" % (prev_char, width, height))
        if not unicodedata.combining(line[-1]):
            print("%s 0 0 %d %d 0" % (line[-1], width, height))
        print("\t 0 0 %d %d 0" % (width, height))


def _write_box_rtl(txt_path: Path, image_path: Path, out_path: Path):
    """"""
    width, height = PIL.Image.open(image_path).size
    with io.open(txt_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        if len(lines) != 1:
            raise ValueError(f"ERROR: GT file {txt_path} must not have 1+ lines: {len(lines)}")
    line = unicodedata.normalize("NFC", lines[0].strip())
    # create WordStr line boxes for Indic & RTL
    if line:
        line = bidi.algorithm.get_display(line)
        print("WordStr 0 0 %d %d 0 #%s" % (width, height, line))
        print("\t 0 0 %d %d 0" % (width, height))


def _main(path_data_dir: Path):
    pass


if __name__ == "__main__":
    path_data_pairs = Path(sys.argv[1])
    if not path_data_pairs.is_dir():
        raise RuntimeError(f"Invalid data_path {path_data_pairs}!")
    _main(path_data_pairs)
