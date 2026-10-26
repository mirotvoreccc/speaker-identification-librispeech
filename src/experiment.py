"""Closed-set speaker identification on LibriSpeech train-clean-100."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def collect_files(root: Path, *, speakers: int, per_speaker: int, seed: int) -> list[tuple[Path, int, str]]:
    """Select four disjoint chapters per speaker for train, validation and test."""
    if speakers < 2 or per_speaker < 40 or per_speaker % 4:
        raise ValueError("Use at least 2 speakers and a per-speaker count divisible by 4 and >= 40")
    quota = per_speaker // 4
    by_speaker: dict[int, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.rglob("*.flac")):
        try:
            speaker_id = int(path.parent.parent.name)
        except ValueError:
            continue
        by_speaker[speaker_id][path.parent.name].append(path)
    eligible = sorted(
        speaker_id for speaker_id, chapters in by_speaker.items()
        if sum(len(paths) >= quota for paths in chapters.values()) >= 4
    )
    if len(eligible) < speakers:
        raise ValueError(
            f"Found {len(eligible)} speakers with four chapters of {quota} recordings; "
            f"requested {speakers}. Check --data-dir and subset size."
        )
    rng = np.random.default_rng(seed)
    selected: list[tuple[Path, int, str]] = []
    for speaker_id in eligible[:speakers]:
        chapters = sorted(chapter for chapter, paths in by_speaker[speaker_id].items() if len(paths) >= quota)
        picked = rng.choice(chapters, size=4, replace=False)
        for chapter, partition in zip(picked, ("train", "train", "validation", "test")):
            paths = by_speaker[speaker_id][chapter]
            indices = rng.choice(len(paths), size=quota, replace=False)
            selected.extend((paths[int(i)], speaker_id, partition) for i in indices)
    return selected


def extract_features(path: Path) -> np.ndarray:
    """Mean and standard deviation of MFCC and delta coefficients."""
    import librosa
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    combined = np.vstack([mfcc, delta])
    return np.r_[combined.mean(axis=1), combined.std(axis=1)]


def metrics(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predicted = model.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y, predicted)), 4),
        "macro_f1": round(float(f1_score(y, predicted, average="macro")), 4),
    }


def run(root: Path, *, speakers: int = 10, per_speaker: int = 40, seed: int = 42) -> dict:
    chosen = collect_files(root, speakers=speakers, per_speaker=per_speaker, seed=seed)
    X = np.vstack([extract_features(path) for path, _, _ in chosen])
    y = np.array([speaker_id for _, speaker_id, _ in chosen])
    split = {
        part: np.array([i for i, (_, _, assigned) in enumerate(chosen) if assigned == part])
        for part in ("train", "validation", "test")
    }
    chapter_allocation: dict[str, dict[str, list[str]]] = {}
    for speaker_id in sorted(int(x) for x in np.unique(y)):
        chapter_allocation[str(speaker_id)] = {
            part: sorted({path.parent.name for path, speaker, assigned in chosen
                          if speaker == speaker_id and assigned == part})
            for part in split
        }
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "linear_svm": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=seed)),
    }
    validation = {}
    for name, model in candidates.items():
        model.fit(X[split["train"]], y[split["train"]])
        validation[name] = metrics(model, X[split["validation"]], y[split["validation"]])
    selected = max(validation, key=lambda name: validation[name]["macro_f1"])
    test = metrics(candidates[selected], X[split["test"]], y[split["test"]])
    return {
        "task": "closed-set speaker identification",
        "dataset": "LibriSpeech train-clean-100",
        "seed": seed,
        "speakers": speakers,
        "selected_speaker_ids": sorted(int(speaker_id) for speaker_id in np.unique(y)),
        "utterances_per_speaker": per_speaker,
        "samples": {part: len(index) for part, index in split.items()},
        "chapter_allocation": chapter_allocation,
        "validation": validation,
        "selected_model": selected,
        "test": test,
        "limitations": "The same speakers occur in train and test, but chapters are disjoint. This is not unseen-speaker verification.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    parser.add_argument("--speakers", type=int, default=10)
    parser.add_argument("--per-speaker", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(
        args.data_dir,
        speakers=args.speakers,
        per_speaker=args.per_speaker,
        seed=args.seed,
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
