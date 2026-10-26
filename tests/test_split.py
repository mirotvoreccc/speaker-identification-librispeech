import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.experiment import collect_files


class ChapterSplitTests(unittest.TestCase):
    def test_chapters_are_disjoint_and_selection_is_balanced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for speaker in (10, 20, 30):
                for chapter in (100, 200, 300, 400, 500):
                    folder = root / str(speaker) / str(chapter)
                    folder.mkdir(parents=True)
                    for i in range(12):
                        (folder / f"{speaker}-{chapter}-{i:04}.flac").touch()

            chosen = collect_files(root, speakers=2, per_speaker=40, seed=42)
            self.assertEqual(chosen, collect_files(root, speakers=2, per_speaker=40, seed=42))
            counts = Counter((speaker, part) for _, speaker, part in chosen)
            for speaker in (10, 20):
                self.assertEqual(counts[(speaker, "train")], 20)
                self.assertEqual(counts[(speaker, "validation")], 10)
                self.assertEqual(counts[(speaker, "test")], 10)
                chapters = {
                    part: {path.parent.name for path, selected, assigned in chosen
                           if selected == speaker and assigned == part}
                    for part in ("train", "validation", "test")
                }
                self.assertEqual([len(chapters[part]) for part in chapters], [2, 1, 1])
                self.assertFalse(chapters["train"] & chapters["validation"])
                self.assertFalse(chapters["train"] & chapters["test"])
                self.assertFalse(chapters["validation"] & chapters["test"])

    def test_rejects_insufficient_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "four chapters"):
                collect_files(Path(tmp), speakers=2, per_speaker=40, seed=42)


if __name__ == "__main__":
    unittest.main()
