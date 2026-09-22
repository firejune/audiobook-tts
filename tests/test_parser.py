# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0

import unittest
from audiobook_tts.text import (
    is_audiobook_markdown,
    parse_frontmatter,
    parse_audiobook_script,
    split_into_sentences,
)


class TestAudiobookParser(unittest.TestCase):

    def test_plain_text_auto_detection(self):
        text = """어두운 밤이었다. 찬 바람이 불었다.
누구냐! 그가 소리쳤다.
"""
        self.assertFalse(is_audiobook_markdown(text))
        chunks = parse_audiobook_script(
            text, default_speaker="Vivian", default_instruct="Storyteller"
        )
        self.assertEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["character"], "Narrator")
        self.assertEqual(chunks[0]["speaker"], "Vivian")
        self.assertEqual(chunks[0]["instruct"], "Storyteller")
        self.assertEqual(chunks[0]["text"], "어두운 밤이었다.")
        self.assertEqual(chunks[2]["text"], "누구냐!")
        self.assertEqual(chunks[3]["text"], "그가 소리쳤다.")

    def test_markdown_frontmatter_and_character_mapping(self):
        script = """---
speaker: Serena
instruct: Calm storytelling
characters:
  카엘:
    speaker: Ryan
    instruct: Brave hero
  엘리스:
    speaker: Vivian
    instruct: Gentle mage
---

어두운 동굴 깊은 곳에서 차가운 바람이 불어왔다.

**카엘**: 드디어 찾아냈군.

**엘리스** (속삭이며): 조심해요. 무언가 있어요.
"""
        self.assertTrue(is_audiobook_markdown(script))
        chunks = parse_audiobook_script(script)
        
        # Chunk 0: Exposition
        self.assertEqual(chunks[0]["character"], "Narrator")
        self.assertEqual(chunks[0]["speaker"], "Serena")
        self.assertEqual(chunks[0]["instruct"], "Calm storytelling")

        # Chunk 1: Kael
        self.assertEqual(chunks[1]["character"], "카엘")
        self.assertEqual(chunks[1]["speaker"], "Ryan")
        self.assertEqual(chunks[1]["instruct"], "Brave hero")
        self.assertEqual(chunks[1]["text"], "드디어 찾아냈군.")

        # Chunk 2 & 3: Alice with overridden emotion
        self.assertEqual(chunks[2]["character"], "엘리스")
        self.assertEqual(chunks[2]["speaker"], "Vivian")
        self.assertEqual(chunks[2]["instruct"], "속삭이며")
        self.assertEqual(chunks[2]["text"], "조심해요.")

        self.assertEqual(chunks[3]["character"], "엘리스")
        self.assertEqual(chunks[3]["speaker"], "Vivian")
        self.assertEqual(chunks[3]["instruct"], "속삭이며")
        self.assertEqual(chunks[3]["text"], "무언가 있어요.")

    def test_inline_speaker_and_blockquote(self):
        script = """> **아서 [Uncle]** (비장하게): "성문을 닫아라!"
"""
        self.assertTrue(is_audiobook_markdown(script))
        chunks = parse_audiobook_script(script)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["character"], "아서")
        self.assertEqual(chunks[0]["speaker"], "Uncle")
        self.assertEqual(chunks[0]["instruct"], "비장하게")
        self.assertEqual(chunks[0]["text"], "성문을 닫아라!")  # Quotes stripped

    def test_multiline_dialogue(self):
        script = """**카엘**:
어둠 속에서 무언가 움직인다.
모두 전투 준비!
"""
        self.assertTrue(is_audiobook_markdown(script))
        chunks = parse_audiobook_script(script, default_speaker="Ryan")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["character"], "카엘")
        self.assertEqual(chunks[0]["text"], "어둠 속에서 무언가 움직인다.")
        self.assertEqual(chunks[1]["character"], "카엘")
        self.assertEqual(chunks[1]["text"], "모두 전투 준비!")

    def test_acoustic_safeguards_and_bundling(self):
        script = """---
speaker: Vivian
---
퍽! 퍽! 퍽! 퍽!
---
으악! 살려줘!
"""
        chunks = parse_audiobook_script(script)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["text"], "퍽! 퍽! 퍽! 퍽!")
        self.assertEqual(chunks[1]["text"], "으악!")
        self.assertEqual(chunks[2]["text"], "살려줘!")

    def test_narrative_leading_instruction(self):
        script = """---
speaker: Vivian
instruct: Default tone
---
(불길한 어조로) 하늘이 붉게 물들기 시작했다.
그리고 아무도 없었다.
"""
        chunks = parse_audiobook_script(script)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["character"], "Narrator")
        self.assertEqual(chunks[0]["speaker"], "Vivian")
        self.assertEqual(chunks[0]["instruct"], "불길한 어조로")
        self.assertEqual(chunks[0]["text"], "하늘이 붉게 물들기 시작했다.")

        self.assertEqual(chunks[1]["character"], "Narrator")
        self.assertEqual(chunks[1]["speaker"], "Vivian")
        self.assertEqual(chunks[1]["instruct"], "Default tone")
        self.assertEqual(chunks[1]["text"], "그리고 아무도 없었다.")

    def test_markdown_header_strip(self):
        script = """# 제 1장: 시작의 여명
**카엘**: 드디어 때가 왔다.
"""
        chunks = parse_audiobook_script(script, default_speaker="Vivian")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["character"], "Narrator")
        self.assertEqual(chunks[0]["text"], "제 1장: 시작의 여명")
        self.assertEqual(chunks[1]["character"], "카엘")
        self.assertEqual(chunks[1]["text"], "드디어 때가 왔다.")


if __name__ == "__main__":
    unittest.main()
