# OCR Util

![python-app](https://github.com/ulb-sachsen-anhalt/ocr-util/actions/workflows/python-app.yml/badge.svg) [![Coverage](./coverage.svg)](https://github.com/ulb-sachsen-anhalt/ocr-util/actions/workflows/pytest.yml) [![PyPi version](https://badgen.net/pypi/v/digital-eval/)](https://pypi.org/project/digital-eval) ![PyPI - Downloads](https://img.shields.io/pypi/dm/digital-eval) ![PyPI - License](https://img.shields.io/pypi/l/digital-eval) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/digital-eval)


Collection of utils to 
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
ocr eval --help

# corpus management
ocr corpus --help

# slice image by image + input OCR
ocr slice --help

# render image + input OCR
ocr show --help
```

### Data problems

Inconsistent OCR Groundtruth with empty texts (ALTO String elements missing CONTENT or PAGE without TextEquiv) or invalid geometrical coordinates (less than 3 points or even empty) will lead to evaluation errors if geometry must be respected.

_Please note_:  
Invalid data files are tried(!) to be excluded from evaluation.

### Evaluation Filter-Then-Aggregate

The evaluation CLI supports a single pre-aggregation filter using metadata extractors.

Example: keep only entries where MODS language is exactly German, then aggregate by publication century:

```bash
ocr eval <candidates> \
	--reference <groundtruth> \
	--mets-file <mets.xml> \
	--filter-by "mods:language=ger" \
	--aggregate-by "mods:dateIssued:century"
```

Multi-language filter values are interpreted as sets:

```bash
ocr eval <candidates> \
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

Plattform: Intel(R) Core(TM) i5-6500 CPU@3.20GHz, 16GB RAM, Ubuntu 22.04 LTS, Python 3.10+

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
