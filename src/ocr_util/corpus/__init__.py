"""Public API for the OCR ground truth corpus generation package.

Import :func:`~ocr_util.corpus.generate_corpus.generate` and
:class:`~ocr_util.corpus.common.CorpusArgs` to create a METS-based corpus
from a directory of PAGE-XML ground truth files::

    from ocr_util.corpus.generate_corpus import generate
    from ocr_util.corpus.common import CorpusArgs
    from pathlib import Path

    result = generate(CorpusArgs(
        input_dir=Path("gt/"),
        output_dir=Path("corpus/"),
        local_cache_dir=Path("/tmp/mets_cache"),
    ))
    print(result.file_path, result.n_pages)
"""
