"""Preprocessings for metric input"""

import unicodedata
import string
import typing

from pathlib import Path

import nltk
import nltk.corpus as nltk_corp

from ocr_util.eval import constants as eval_constants

import ocr_util.eval.model.common as mcommon
import ocr_util.eval.model.main as mmain
import ocr_util.eval.model.digital_object_model as mdom

# Python3 standard Unicode Normalization
#
UC_NORMALIZATION_DEFAULT = "NFC"
UC_NORMALIZATION_NFKD = "NFKD"

# whitespaces
#
# usual spatium and special control sequences
WHITESPACES = string.whitespace

WHITESPACES_EXCL_SPACE_CHARS = WHITESPACES[1:]

# punctuations
#
#   * regular ASCII-punctuations
#   * Dashes        \u2012-2017
#   * Quotations    \u2018-201F
PUNCTUATIONS = (
    string.punctuation
    + "\u2012"
    + "\u2013"
    + "\u2014"
    + "\u2015"
    + "\u2016"
    + "\u2017"
    + "\u2018"
    + "\u2019"
    + "\u201a"
    + "\u201b"
    + "\u201c"
    + "\u201d"
    + "\u201e"
    + "\u201f"
)
# no special line break delimiter
PUNCTUATIONS = PUNCTUATIONS + "\u2e17"  # DOUBLE OBLIQUE HYPHEN
# no spaces
PUNCTUATIONS = (
    PUNCTUATIONS
    + "\u0020"
    + "\u00a0"
    + "\u2000"
    + "\u2001"
    + "\u2002"
    + "\u2003"
    + "\u2004"
    + "\u2005"
    + "\u2006"
    + "\u2007"
    + "\u2008"
    + "\u2009"
    + "\u200a"
    + "\u2028"
    + "\u205f"
    + "\u3000"
)

# digits
#
#   * ASCII digits
#   * arabic digits
#   * persian / indic digits
DIGITS = (
    string.digits
    + "\u0660"
    + "\u0661"
    + "\u0662"
    + "\u0663"
    + "\u0664"
    + "\u0665"
    + "\u0666"
    + "\u0667"
    + "\u0668"
    + "\u0669"
)
# persian indic digits
DIGITS = (
    DIGITS
    + "\u06f0"
    + "\u06f1"
    + "\u06f2"
    + "\u06f3"
    + "\u06f4"
    + "\u06f5"
    + "\u06f6"
    + "\u06f7"
    + "\u06f8"
    + "\u06f9"
)

WHITESPACE_TRNSL = str.maketrans("", "", WHITESPACES)
PUNCT_TRNSL = str.maketrans("", "", PUNCTUATIONS)
DIGIT_TRNSL = str.maketrans("", "", DIGITS)

#
# information retrieval (nltk)
#
NLTK_STOPWORDS = [
    "german",
    "russian",
    "english",
    "french",
    "greek",
    "arabic",
    "turkish",
    "italian",
]
STOPWORDS_DEFAULT = ["german", "english", "arabic", "russian"]


class PreprocessingException(Exception):
    """Mark problems concerning preprocessing"""


class Preprocessor:
    """Wrapper for different preprocessing
    implementations concerning textual
    OCR metric input data"""

    def __init__(self, input_data):
        self._input = input_data
        self._input_count = None
        self._input_unit = None
        self._steps = []
        self.spatial_report = None

    @staticmethod
    def _measure(value):
        if isinstance(value, str):
            return len(value), "characters"
        if isinstance(value, list):
            return len(value), "tokens"
        if isinstance(value, set):
            return len(value), "unique tokens"
        return 0, "items"

    def _set_input_basis(self):
        self._input_count, self._input_unit = self._measure(self._input)

    def _record_step(self, name, before, after, details=None):
        before_count, before_unit = self._measure(before)
        after_count, after_unit = self._measure(after)
        self._steps.append(
            {
                eval_constants.STEP_NAME: name,
                eval_constants.STEP_BEFORE_COUNT: before_count,
                eval_constants.STEP_BEFORE_UNIT: before_unit,
                eval_constants.STEP_AFTER_COUNT: after_count,
                eval_constants.STEP_AFTER_UNIT: after_unit,
                eval_constants.STEP_DETAILS: details or {},
            }
        )

    def run(self):
        """Wrap required preprocessings"""

    @property
    def result(self):
        """get processed data"""
        return self._input

    @property
    def preprocessing_report(self):
        """Describe the concrete metric basis and transformations applied."""
        output_count, output_unit = self._measure(self._input)
        return {
            eval_constants.REPORT_PREPROCESSOR: type(self).__name__,
            eval_constants.REPORT_INPUT_COUNT: self._input_count,
            eval_constants.REPORT_INPUT_UNIT: self._input_unit,
            eval_constants.REPORT_OUTPUT_COUNT: output_count,
            eval_constants.REPORT_OUTPUT_UNIT: output_unit,
            eval_constants.REPORT_STEPS: self._steps,
            eval_constants.REPORT_SPATIAL: self.spatial_report,
        }


class TextPreprocessor(Preprocessor):
    """Common flavor to handle plain text strings"""

    def __init__(self, input_data):
        super().__init__(input_data)
        self.code_norm = UC_NORMALIZATION_DEFAULT
        self.frame = None
        self.one_liner = True

    def normalize_encoding(self):
        """Convert code points to
        required unicode form"""

        before = self._input
        self._input = unicodedata.normalize(self.code_norm, self._input)
        self._record_step(f"Unicode {self.code_norm}", before, self._input)

    def normalize_whitespace(self):
        """normalize line breaks and control characters to spaces.
        
        replaces newlines, carriage returns, and other line break
        sequences with single spaces. This ensures consistent handling
        of multi-line text data from different sources (ALTO, PAGE, etc).
        """
        if isinstance(self._input, str):
            before = self._input
            # Replace various line break sequences with spaces
            self._input = self._input.replace('\r\n', ' ')  # Windows line breaks
            self._input = self._input.replace('\r', ' ')    # Old Mac line breaks
            self._input = self._input.replace('\n', ' ')    # Unix line breaks
            self._input = self._input.replace('\v', ' ')    # Vertical tab
            self._input = self._input.replace('\f', ' ')    # Form feed
            self._record_step("normalize line breaks", before, self._input)

    def run(self):
        if isinstance(self._input, Path):
            self.spatial_report = {}
            self._input, _ = file_to_text(
                self._input, self.frame, self.one_liner, self.spatial_report
            )
        self._set_input_basis()
        self.normalize_whitespace()
        self.normalize_encoding()


class LetterPreprocessor(TextPreprocessor):
    """Strip any non-letter characters
    from input data"""

    def strip_chars(self):
        """remove non-letter characters"""

        before = self._input
        removed = {
            "whitespace": sum(char in WHITESPACES for char in before),
            "punctuation": sum(
                char in PUNCTUATIONS and char not in WHITESPACES for char in before
            ),
            "digits": sum(char in DIGITS for char in before),
        }
        self._input = self._input.translate(WHITESPACE_TRNSL)
        self._input = self._input.translate(PUNCT_TRNSL)
        self._input = self._input.translate(DIGIT_TRNSL)
        self._record_step("remove non-letter characters", before, self._input, removed)

    def run(self):
        super().run()
        self.strip_chars()


class SimpleTokenizer(TextPreprocessor):
    """Tokenize input string"""

    def tokenize(self):
        """make string list"""
        before = self._input
        self._input = (
            self._input.split() if isinstance(self._input, str) else self._input
        )
        self._record_step("tokenize", before, self._input)

    def run(self):
        super().run()
        self.tokenize()


class LanguageAwareTokenizer(SimpleTokenizer):
    """Tokenize to set"""

    def __init__(self, input_data, languages=None):
        super().__init__(input_data)
        if languages is None:
            languages = STOPWORDS_DEFAULT
        self.languages = languages

    def tokenize_to_sorted_set(self):
        """enhanced tokenizing"""
        self.tokenize()
        before = self._input
        self._input = set(sorted(self._input))
        self._record_step("deduplicate tokens", before, self._input)

    def strip_stopwords(self):
        """remove some tokens"""
        before = self._input
        self._input = self._input - LanguageAwareTokenizer._get_stopwords(
            languages=self.languages
        )
        self._record_step(
            "remove stopwords", before, self._input, {"languages": self.languages}
        )

    def run(self):
        if isinstance(self._input, Path):
            self.spatial_report = {}
            self._input, _ = file_to_text(
                self._input, self.frame, self.one_liner, self.spatial_report
            )
        self._set_input_basis()
        self.normalize_encoding()
        self.tokenize_to_sorted_set()
        self.strip_stopwords()

    @classmethod
    def _get_stopwords(cls, nltk_mappings=None, languages=None) -> typing.Set[str]:
        """Helper Function to gather NLTK stopword data
        * ensure stopwords files are locally available
        * extract them as set
        """
        if nltk_mappings is None:
            nltk_mappings = NLTK_STOPWORDS
        try:
            for mapping in nltk_mappings:
                nltk_corp.stopwords.words(mapping)
        except LookupError:
            nltk.download("stopwords")
        if languages is None:
            languages = STOPWORDS_DEFAULT
        _stopwords = {
            _all_words
            for _lang in languages
            for _all_words in nltk_corp.stopwords.words(_lang)
        }
        return _stopwords


class DictionaryTextPreprocessor(TextPreprocessor):
    """Prepare input data to be comparable
    with dictionary entries or spellcheckers
    """

    # diacritica to take care of
    _COMBINING_SMALL_E = "\u0364"

    def __init__(self, input_data):
        if isinstance(input_data, Path):
            input_data, _ = file_to_dict_text(input_data, oneliner=True)
        super().__init__(input_data)

    def normalize_vocal_ligatures(self) -> str:
        """Replace vocal ligatures, which otherwise
        may confuse the index component workflow,
        especially COMBINING SMALL LETTER E : \u0364

        a^e, o^e, u^e => (u0364) => ä, ö, ü
        """

        _out = []
        for i, _c in enumerate(self._input):
            if _c == DictionaryTextPreprocessor._COMBINING_SMALL_E:
                _preceeding_vocal = _out[i - 1]
                _vocal_name = unicodedata.name(_preceeding_vocal)
                _replacement = ""
                if "LETTER A" in _vocal_name:
                    _replacement = "ä"
                elif "LETTER O" in _vocal_name:
                    _replacement = "ö"
                elif "LETTER U" in _vocal_name:
                    _replacement = "ü"
                else:
                    _msg = f"No conversion for {_preceeding_vocal} ('{self._input}')!"
                    raise PreprocessingException(f"normalize vocal ligatures: {_msg}")
                _out[i - 1] = _replacement
            _out.append(_c)

        # strip all combining e's anyway
        self._input = "".join(_out).replace(
            DictionaryTextPreprocessor._COMBINING_SMALL_E, ""
        )

    def run(self):
        self.normalize_encoding()
        self.normalize_vocal_ligatures()


def file_to_dict_text(file_path: str, frame=None, oneliner=False) -> typing.Tuple:
    """Convert file data into sanitized text
    usable for spellcheckings or dictionairies"""

    line_texts: typing.List[str]
    len_lines: int
    line_texts, len_lines = file_to_text(
        file_path=file_path, frame=frame, oneliner=False
    )
    non_empty_lines: typing.List[str] = [
        line_text for line_text in line_texts if len(line_text) > 0
    ]
    lines_sanitized_wraps: typing.List[str] = _sanitize_wraps(non_empty_lines)
    lines_sanitized_chars: typing.List[str] = _sanitize_chars(lines_sanitized_wraps)
    text = " ".join(lines_sanitized_chars) if oneliner else lines_sanitized_chars
    return text, len_lines


def file_to_text(file_path, frame=None, oneliner=True, spatial_report=None) -> typing.Tuple:
    """Convert file data into plain text"""

    try:
        top_digo: mdom.DigitalObjectTree = mmain.to_digital_object(str(file_path))
        candidate_page_frame = top_digo.dimensions
        requested_frame = frame
        # explicit filter frame?
        if not frame:
            frame = top_digo.dimensions
        elif len(frame) == 2:
            frame = [
                [frame[0][0], frame[0][1]],
                [frame[1][0], frame[0][1]],
                [frame[1][0], frame[1][1]],
                [frame[0][0], frame[1][1]],
            ]
        frame_digo = mdom.DigitalObjectTree()
        frame_digo.dimensions = frame
        filtered_words, total_words = filter_pieces(frame_digo, top_digo)
        if spatial_report is not None:
            spatial_report.update(
                {
                    eval_constants.SPATIAL_MODE: "full page"
                    if requested_frame is None
                    else "frame filtering",
                    eval_constants.SPATIAL_REQUESTED_FRAME: requested_frame,
                    eval_constants.SPATIAL_CANDIDATE_PAGE_FRAME: candidate_page_frame,
                    eval_constants.SPATIAL_TOTAL_TOKEN: total_words,
                    eval_constants.SPATIAL_EXCLUDED_TOKEN: f"{len(filtered_words)}: ({', '.join(filtered_words)})",
                    eval_constants.SPATIAL_RETAINED_TOKEN: total_words - len(filtered_words),
                }
            )
        the_lines = _get_digos_from_digo(top_digo)
        if oneliner:
            return top_digo.transcription, len(the_lines)
        return [line.transcription for line in the_lines], len(the_lines)
    except mcommon.DigitalObjectException as _:
        with open(file_path, mode="r", encoding="utf-8") as fhandle:
            text_lines = fhandle.readlines()
            if oneliner:
                text_lines = " ".join([l.strip() for l in text_lines])
            return text_lines, len(text_lines)
    except Exception as exc:
        raise RuntimeError(f"Unexpected error for {file_path}: {exc}") from exc


def filter_pieces(frame, current) -> typing.Tuple[typing.List, int]:
    """respect frame for current digital object
    return information about filtered elements
    """
    _filtered = []
    _tmp_stack = []
    _total_stack = []
    # stack all items
    _total_stack.append(current)
    _tmp_stack.append(current)
    while _tmp_stack:
        _current: mdom.DigitalObjectTree = _tmp_stack.pop()
        if _current.children:
            _tmp_stack += _current.children
            _total_stack += _current.children
    # now pick words
    _words = [_p for _p in _total_stack if _p.level == mdom.DigitalObjectLevel.WORD]

    # check for each word piece
    for _word in _words:
        if _word not in frame:
            _filtered.append(_word.transcription)
            _uplete(_word)
    return _filtered, len(_words)


def _uplete(curr: mdom.DigitalObjectTree):
    if len(curr.children) == 0 and curr.level < mdom.DigitalObjectLevel.PAGE \
        and curr.parent is not None:
        _pa: mdom.DigitalObjectTree = curr.parent
        _pa.remove_children(curr)
        _uplete(_pa)


def _get_digos_from_digo(digo: mdom.DigitalObjectTree, lines=None) -> typing.List:
    if lines is None:
        lines = []
    if digo.level == mdom.DigitalObjectLevel.LINE and digo.transcription:
        lines.append(digo)
        return lines
    for child in digo.children:
        _get_digos_from_digo(child, lines)
    return lines


_HYPHENS: typing.List[str] = [
    "⸗",
    "-",
    "—",
]


def _sanitize_wraps(lines: typing.List[str]) -> typing.List[str]:
    """Sanitize word wraps if
    * last word token ends with '-', "⸗" or "—"
    * another line following
    * following line not empty
    """

    normalized_lines: typing.List[str] = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            for hyphen in _HYPHENS:
                if line.endswith(hyphen):
                    next_line = lines[i + 1]
                    if len(next_line.strip()) == 0:
                        # encountered empty next line, no merge possible
                        continue
                    next_line_tokens = next_line.split()
                    nextline_first_token = next_line_tokens.pop(0)
                    # join the rest of valid next line
                    lines[i + 1] = " ".join(next_line_tokens)
                    line = line[:-1] + nextline_first_token
                    break
        normalized_lines.append(line)
    return normalized_lines


def _sanitize_chars(lines: typing.List[str]) -> typing.List[str]:
    """Replace or remove nonrelevant chars for current german word error rate"""

    sanitized: typing.List[str] = []
    for line in lines:
        text = line.strip()
        bad_chars = "0123456789“„\"'?!*.;:-=[]()|"
        text = "".join([c for c in text if c not in bad_chars])
        if ".." in text:
            text = text.replace("..", "")
        if "  " in text:
            text = text.replace("  ", " ")
        text = " ".join([t for t in text.split() if len(t) > 1])
        sanitized.append(text)

    return sanitized
