"""
test_multimodal_search.py

Verifies multimodal search capability:
1. Indexing an image with visual description and detected OCR text.
2. Searching by visual description (e.g., 'man in orange scarf and traditional outfit').
3. Searching by OCR text (e.g., 'Festival Celebrations 2026').
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from file_intel.semantic_index import SemanticFileIndex
from file_intel.retriever_ranker import HybridFileRetriever
from file_intel.multimodal_extractor import MultimodalExtractor


class TestMultimodalSearch(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_multimodal.db")
        self.index = SemanticFileIndex(db_path=self.db_path)
        self.retriever = HybridFileRetriever(self.index)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multimodal_extractor_metadata(self):
        extractor = MultimodalExtractor()
        test_img = Path("facerecog/test.png")
        self.assertTrue(test_img.exists())
        meta = extractor.extract_image_metadata(test_img)
        self.assertGreater(meta["width"], 0)
        self.assertGreater(meta["height"], 0)
        self.assertIn(meta["format"].lower(), ["png"])

    def test_multimodal_visual_and_ocr_search(self):
        img_path = Path("facerecog/test.png").resolve()
        content = (
            f"File: {img_path.name}\n"
            f"Path: {str(img_path)}\n"
            "Category: image\n"
            "Image Dimensions: 216x260 (PNG)\n\n"
            "[Visual Content & Scene Description]:\n"
            "A photo of a smiling man wearing a traditional Indian outfit with an orange and white patterned scarf "
            "and a dark blue vest standing against a dark backdrop.\n\n"
            "[Detected Text / OCR]:\n"
            "Namaste Traditional Festival Celebrations 2026\n"
        )

        fid = self.index.index_file(str(img_path), content=content, force=True)
        self.assertIsNotNone(fid)

        # 1. Visual Query: Search by what the image looks like
        results_visual = self.retriever.search("photo of a person wearing an orange scarf", top_k=3)
        self.assertGreater(len(results_visual), 0)
        top_hit = results_visual[0]
        self.assertEqual(top_hit.file_name, "test.png")
        self.assertEqual(top_hit.category, "image")
        self.assertIn("orange", top_hit.text_snippet.lower())

        # 2. OCR Query: Search by the text written on the image
        results_ocr = self.retriever.search("Festival Celebrations 2026", top_k=3)
        self.assertGreater(len(results_ocr), 0)
        self.assertTrue(any(r.file_name == "test.png" for r in results_ocr))


if __name__ == "__main__":
    unittest.main()
