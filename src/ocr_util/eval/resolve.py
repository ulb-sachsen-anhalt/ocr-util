"""Search and find data pairs
candidates and corresponding reference/groundtruth
"""

import os
import typing

from pathlib import Path


from .evaluation import EvalEntry


def gather_candidates(start_path: Path, file_ext=".xml") -> typing.List[EvalEntry]:
    """gather all files from start_path, by default
    XML-like (ALTO, PAGE)"""
    candidates = []
    if os.path.isdir(start_path):
        for curr_dir, _, files in os.walk(start_path):
            xml_files = [f for f in files if str(f).endswith(file_ext)]
            if xml_files:
                for xml_file in xml_files:
                    rel_path: Path = Path(curr_dir) / xml_file
                    entry = EvalEntry(rel_path.absolute(), start_path)
                    candidates.append(entry)
    else:
        candidates.append(EvalEntry(start_path, start_path.parent))
    candidates.sort(key=lambda e: e.path_candidate)
    return candidates


def find_groundtruth(eval_entry: EvalEntry, gt_domain_root):
    """Find correspondig groundtruth file for
    given candidate by directory layout and check
    start and end of probably match
    """
    candidate_name = eval_entry.path_candidate.stem

    gt_files = [
        Path(c, f)
        for c, _, fs in os.walk(gt_domain_root, followlinks=True)
        for f in fs
        if _name_approved(f, candidate_name)
    ]
    if len(gt_files) > 0:
        eval_entry.path_groundtruth = gt_files[0]
        return gt_files[0]
    return None


def _name_approved(fname: str, estm_name: str) -> bool:
    suffix_ok = fname.endswith(".gt.xml") or fname.endswith("gt.txt") or fname.endswith(".xml")
    return fname.startswith(estm_name) and suffix_ok
