# OCR Util

![python-app](https://github.com/ulb-sachsen-anhalt/ocr-util/actions/workflows/python-app.yml/badge.svg) [![Coverage](https://raw.githubusercontent.com/ulb-sachsen-anhalt/ocr-util/main/coverage.svg)](https://github.com/ulb-sachsen-anhalt/ocr-util/actions/workflows/python-app.yml) [![PyPi version](https://badgen.net/pypi/v/ocr-util/)](https://pypi.org/project/ocr-util) ![PyPI - Downloads](https://img.shields.io/pypi/dm/ocr-util) ![PyPI - License](https://img.shields.io/pypi/l/ocr-util) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ocr-util)


Collection of utils for 
* evaluation of OCR data for the masses
* generation of extended OCR-Evaluation Corpora
* generation of pair-wise Trainingdata for OCR-Backends

## Requirements

* recent *nix-OS
* Python3.10+ Environment

## Usage

Each section contains detailed usage help instructions:

```bash
# evaluation
ocr-util eval --help

# corpus management
ocr-util corpus --help

# slice image by image + input OCR
ocr-util slice --help

# render image + input OCR
ocr-util show --help
```

### Data problems

Inconsistent OCR Groundtruth with empty texts (ALTO String elements missing CONTENT or PAGE without TextEquiv) or invalid geometrical coordinates (less than 3 points or even empty) will lead to evaluation errors if geometry must be respected.

_Please note_:  
Invalid data files are excluded and reported where possible from evaluation.  
The term 'invalid' refers to errors in schemas in structured XML-data, i.e. syntax errors and further if included geometrical information includes inconsistencies like missing points or missmatching shapes.

### Evaluation Filter-Then-Aggregate

The evaluation CLI supports a single pre-aggregation filter using metadata extractors.

Example: keep only entries where MODS language is exactly German, then aggregate by publication century:

```bash
ocr-util eval <candidates> \
	--reference <groundtruth> \
	--mets-file <mets.xml> \
	--filter-by "mods:language=ger" \
	--aggregate-by "mods:dateIssued:century"
```

Multi-language filter values are interpreted as sets:

```bash
ocr-util eval <candidates> \
	--reference <groundtruth> \
	--mets-file <mets.xml> \
	--filter-by "mods:language=ger+eng" \
	--aggregate-by "mods:dateIssued:century"
```

Behavior:
* single filter value -> exact match (e.g. `ger` does not match `ger+eng`)
* multi-value filter -> all filter values must be present in any order
* entries missing the filter criterion are reported as WARNING and discarded

## Development

Platform: Intel(R) Core(TM) i5-6500 CPU@3.20GHz, 16GB RAM, Ubuntu 22.04 LTS, Python 3.10+

```bash
# clone local
git clone <repository-url> <local-dir>
cd <local-dir>

# enable virtual python 3 environment (linux)
# and update pip itself
python3.10 -m venv venv
. venv/bin/activate
python -m pip install -U pip

# install with dev dependencies
python -m pip install -e ".[dev,test]"

# run tests with coverage
python -m pytest --cov=src

# run tests faster (parallel, auto worker count)
python -m pytest -q -n auto
```

## Contribution

Contributions, suggestions and proposals welcome!

## License

Under terms of the [MIT license](https://opensource.org/licenses/MIT).

**NOTE**: This software depends on packages that _might_ be licensed under different terms.
