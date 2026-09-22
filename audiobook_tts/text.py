# coding=utf-8
# Copyright 2026 The Audiobook-TTS Authors.
# SPDX-License-Identifier: Apache-2.0
"""
Text processing, audiobook markdown script parsing, and line-aware sentence tokenization.

Features:
1. Zero-config Auto-Detection: Distinguishes between plain literary text and multi-character
   Audiobook Markdown scripts.
2. Frontmatter & Script Syntax: Supports YAML frontmatter (characters, default voices, styles)
   and playwright script patterns (**Character** (Emotion): Speech).
3. Acoustic Safeguards: Strips disruptive quotation marks, drops divider lines, and bundles
   ultra-short bursts (e.g. '퍽! 퍽! 퍽! 퍽!') into natural narration units.
"""

import base64
import io
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict


class NarrationChunk(TypedDict):
    index: int
    text: str
    character: str
    speaker: str
    instruct: Optional[str]


# Pre-compiled regex patterns
_QUOTE_CLEAN_RE = re.compile(r'["\'“”‘’「」『』`]')
_PUNCT_SPLIT_RE = re.compile(r'([.!?。！？]+(?:\s+|$))')
_KOREAN_QUOTE_PARTICLES = ('라고', '이라고', '이라며', '라며', '하고', '하며', '면서', '다며', '라고는', '이라고는')
_KOREAN_CASE_PARTICLES = ('은', '는', '이', '가', '을', '를', '과', '와')

_STAGE_DIRECTION_KEYWORDS = (
    '하며', '면서', '하게', '듯이', '채로', '조로', '톤으로', '목소리로', '어조로',
    '속삭이며', '한숨', '웃음', '울먹이며', '소리치며', '절규하며', '조용히', '분노하며',
    '차갑게', '따뜻하게', '단호하게', '비장하게', '놀라며', '당황하며', '기뻐하며',
    '침통하게', '떨리는', '비웃으며', '비명'
)


def _is_hanging_particle(s: str) -> bool:
    """Check if s is a hanging particle or quotation verb resulting from split."""
    if not s:
        return False
    if any(s.startswith(p) for p in _KOREAN_QUOTE_PARTICLES):
        return True
    if s in _KOREAN_CASE_PARTICLES:
        return True
    if any(s.startswith(p + " ") for p in _KOREAN_CASE_PARTICLES):
        return True
    return False


_DIALOGUE_LINE_RE = re.compile(
    r'^\s*(?:>\s*)?\*\*([^\*]+?)\*\*\s*(?:\(([^)]+)\)|\[([^\]]+)\])?\s*:\s*(.*)$'
)
_INNER_SPEAKER_RE = re.compile(r'^(.+?)(?:\s*[\[\(]([a-zA-Z0-9_\-]+)[\]\)])?$')
_LEAD_INSTRUCT_RE = re.compile(r'^\s*\(([^)]+)\)\s*(.*)$')
_DIVIDER_LINE_RE = re.compile(r'^[-*=_~#\s]{2,}$')


def _simple_yaml_parse(yaml_str: str) -> Dict[str, Any]:
    """
    Lightweight, zero-dependency YAML parser for audiobook frontmatter metadata.
    Handles basic string key-values and 2-level nested character dictionaries.
    """
    data: Dict[str, Any] = {}
    current_key: Optional[str] = None
    sub_dict: Optional[Dict[str, Any]] = None

    for line in yaml_str.splitlines():
        raw = line
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        indent = len(raw) - len(raw.lstrip())

        if ":" in line:
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip().strip("'\"")

            if indent == 0:
                sub_dict = None
                if v == "":
                    data[k] = {}
                    current_key = k
                else:
                    data[k] = v
                    current_key = None
            elif indent > 0:
                if current_key:
                    if indent == 2:
                        if v == "":
                            data[current_key][k] = {}
                            sub_dict = data[current_key][k]
                        else:
                            data[current_key][k] = v
                    elif indent >= 4 and sub_dict is not None:
                        sub_dict[k] = v
    return data


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter block if present at document start.
    Returns (metadata_dict, remaining_body_text).
    """
    text = (text or "").strip()
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, text

    yaml_str = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:]).strip()

    meta = _simple_yaml_parse(yaml_str)
    return meta, body


def is_audiobook_markdown(text: str) -> bool:
    """
    Detect whether the input text is formatted as an audiobook markdown script
    containing character dialogue cues, frontmatter, or voice instructions.
    """
    text = (text or "").strip()
    if not text:
        return False

    if text.startswith("---") and "\n---" in text:
        return True

    dialogue_check_re = re.compile(
        r'^\s*(?:>\s*)?\*\*([^\*]+?)\*\*\s*(?:\([^\)]+\)|\[[^\]]+\])*\s*:',
        re.MULTILINE,
    )
    return bool(dialogue_check_re.search(text))


def sanitize_speech_text(text: str) -> str:
    """
    Sterilize raw manuscript or dialogue text into clean, spoken dialogue for Qwen3-TTS.

    1. Normalizes ellipsis combinations (……! / ...! -> !, …… / ... -> ,)
    2. Strips or extracts embedded stage directions in parentheses/brackets.
    3. Completely eliminates unbalanced, orphan, or residual bracket characters.
    4. Strips quotation marks, markdown symbols, and isolated Korean jamo.
    5. Normalizes consecutive exclamation/question marks and excessive spacing.
    """
    if not text:
        return ""

    # 1. Normalize ellipsis combined with punctuation:
    #    ……! / ...! / …! -> !
    #    ……? / ...? / …? -> ?
    #    standalone ellipsis …… / ... / … -> , (natural breathing pause)
    text = re.sub(r'[…\.]{2,}[!！]+', '!', text)
    text = re.sub(r'[…\.]{2,}[\?？]+', '?', text)
    text = re.sub(r'[…\.]{2,}', ', ', text)
    text = re.sub(r'[…]+', ', ', text)

    # 2. Extract or strip parenthesized directions inside dialogue
    def _replace_direction_bracket(m: re.Match) -> str:
        content = m.group(1).strip()
        if (
            any(kw in content for kw in _STAGE_DIRECTION_KEYWORDS)
            or (len(content) <= 6 and content.endswith(('히', '게', '며', '로', '채')))
        ):
            return ' '
        # Non-direction clarification (e.g. 사과(Apple) -> 사과 Apple)
        return f' {content} '

    text = re.sub(r'\(([^)]*)\)', _replace_direction_bracket, text)
    text = re.sub(r'\[([^\]]*)\]', _replace_direction_bracket, text)
    text = re.sub(r'\{([^\}]*)\}', _replace_direction_bracket, text)

    # 3. Strip all residual, orphaned, or enclosing bracket characters
    text = re.sub(r'[\(\)\[\]\{\}⟨⟩〈〉《》「」『』【】〔〕]', ' ', text)

    # 4. Strip quotes and markdown formatting
    text = re.sub(r'["\'“”‘’`*#_~>|]', ' ', text)

    # 5. Clean isolated Korean jamo (e.g. ㅋㅋ, ㅠㅠ, ㅎㅎ)
    text = re.sub(r'[ㄱ-ㅎㅏ-ㅣ]+', ' ', text)

    # 6. Normalize punctuation runs (!!! -> !, ??? -> ?)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'[,]{2,}', ',', text)
    text = re.sub(r'~+', ' ', text)

    # 7. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_and_split_sentences(line: str) -> List[str]:
    """
    Split text into distinct sentences optimized for audiobook pacing and narration flow.

    1. Fully sanitizes text via sanitize_speech_text to remove bracket artifacts.
    2. Splits safely on terminal punctuation [.!?。！？].
    3. Merges hanging Korean dialogue particles ('라고', '하고', '이라며').
    4. Bundles consecutive ultra-short fragments into natural narration units.
    """
    line = sanitize_speech_text(line)
    if not line:
        return []

    parts = _PUNCT_SPLIT_RE.split(line)
    it = iter(parts)
    buffer = ""
    line_chunks: List[str] = []
    for item in it:
        if not item:
            continue
        if _PUNCT_SPLIT_RE.fullmatch(item):
            buffer += item.strip()
            if buffer.strip():
                line_chunks.append(buffer.strip())
            buffer = ""
        else:
            buffer += item

    if buffer.strip():
        line_chunks.append(buffer.strip())

    # Merge hanging Korean particles
    merged_chunks: List[str] = []
    for s in line_chunks:
        s = s.strip()
        if not s:
            continue
        if merged_chunks and (
            not any(c.isalnum() for c in s) or
            _is_hanging_particle(s)
        ):
            merged_chunks[-1] = f"{merged_chunks[-1]} {s}".strip()
        else:
            merged_chunks.append(s)

    # Smart bundle short bursts:
    # Bundle when either chunk is a micro-burst (<= 1 alphanumeric char, e.g. '퍽!', '아!')
    # or consecutive chunks are identical repetitive onomatopoeia (e.g. '쾅!' '쾅!').
    bundled: List[str] = []
    accum = ""
    for chunk in merged_chunks:
        if not accum:
            accum = chunk
        else:
            accum_alnums = [c for c in accum if c.isalnum()]
            chunk_alnums = [c for c in chunk if c.isalnum()]
            is_repetition = (accum.strip().rstrip('!?.… ') == chunk.strip().rstrip('!?.… '))
            is_micro = (len(accum_alnums) <= 1 or len(chunk_alnums) <= 1)

            if is_micro or is_repetition:
                accum = f"{accum} {chunk}"
            else:
                bundled.append(accum)
                accum = chunk

    if accum:
        bundled.append(accum)

    results: List[str] = []
    for b in bundled:
        b_clean = re.sub(r'\s+', ' ', b).strip()
        if any(c.isalnum() for c in b_clean):
            results.append(b_clean)

    return results


def parse_audiobook_script(
    text: str,
    default_speaker: str = "Vivian",
    default_instruct: Optional[str] = None,
) -> List[NarrationChunk]:
    """
    Parse manuscript or markdown script into structured narration chunks,
    extracting assigned character, voice speaker preset, and emotional instruction.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Fallback to plain text pipeline if not markdown script
    if not is_audiobook_markdown(text):
        raw_lines = []
        for l in text.splitlines():
            l_str = l.strip()
            if not l_str:
                continue
            if _DIVIDER_LINE_RE.match(l_str) or not any(c.isalnum() for c in l_str):
                continue
            raw_lines.append(l_str)

        all_sents: List[str] = []
        for l in raw_lines:
            all_sents.extend(clean_and_split_sentences(l))

        return [
            {
                "index": i,
                "text": s,
                "character": "Narrator",
                "speaker": default_speaker,
                "instruct": default_instruct,
            }
            for i, s in enumerate(all_sents)
        ]

    # Markdown Script Pipeline
    meta, body = parse_frontmatter(text)
    book_speaker = meta.get("speaker", default_speaker)
    book_instruct = meta.get("instruct", default_instruct)
    characters_map: Dict[str, Any] = meta.get("characters", {})

    chunks: List[NarrationChunk] = []
    lines = body.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue

        # Skip horizontal dividers
        if _DIVIDER_LINE_RE.match(line) or not any(c.isalnum() for c in line):
            continue

        # Dialogue Cue Check
        m = _DIALOGUE_LINE_RE.match(line)
        if m:
            char_raw = m.group(1).strip()
            inline_instruct = m.group(2)
            inline_speaker_bracket = m.group(3)
            speech = m.group(4).strip()

            # Inner speaker format check inside bold: e.g. **카엘 [Ryan]**
            inner_m = _INNER_SPEAKER_RE.match(char_raw)
            if inner_m:
                character = inner_m.group(1).strip()
                if inner_m.group(2):
                    inline_speaker = inner_m.group(2).strip()
                else:
                    inline_speaker = inline_speaker_bracket
            else:
                character = char_raw
                inline_speaker = inline_speaker_bracket

            # Consume subsequent lines belonging to this dialogue block
            # until an empty line, divider, or a new dialogue cue appears
            speech_lines = [speech] if speech else []
            while i < len(lines):
                next_line = lines[i].strip()
                if not next_line:
                    i += 1
                    break
                if _DIVIDER_LINE_RE.match(next_line) or _DIALOGUE_LINE_RE.match(next_line) or next_line.startswith("**"):
                    break
                speech_lines.append(next_line)
                i += 1

            full_speech = " ".join(speech_lines).strip()

            # If stage direction was placed right at the beginning of dialogue:
            # e.g. **카엘**: (속삭이며) 도망쳐!
            lead_speech_m = _LEAD_INSTRUCT_RE.match(full_speech)
            if lead_speech_m:
                if not inline_instruct:
                    inline_instruct = lead_speech_m.group(1).strip()
                full_speech = lead_speech_m.group(2).strip()

            # Resolve character profile
            char_cfg = characters_map.get(character, {}) if isinstance(characters_map, dict) else {}
            speaker = inline_speaker or (char_cfg.get("speaker") if isinstance(char_cfg, dict) else None) or book_speaker
            instruct = inline_instruct or (char_cfg.get("instruct") if isinstance(char_cfg, dict) else None) or book_instruct

            # Clean and tokenize speech sentences
            sub_sents = clean_and_split_sentences(full_speech)
            for s in sub_sents:
                chunks.append({
                    "index": len(chunks),
                    "text": s,
                    "character": character,
                    "speaker": speaker,
                    "instruct": instruct,
                })
        else:
            # Narrative (Exposition) Line
            # Check for leading stage direction: (불길한 어조로) 하늘이 붉게 물들었다.
            narrative_instruct = book_instruct
            lead_m = _LEAD_INSTRUCT_RE.match(line)
            if lead_m:
                narrative_instruct = lead_m.group(1).strip()
                narrative_text = lead_m.group(2).strip()
            else:
                narrative_text = line

            # Strip markdown header syntax: # Chapter 1 -> Chapter 1
            narrative_text = re.sub(r'^#+\s*', '', narrative_text).strip()

            sub_sents = clean_and_split_sentences(narrative_text)
            for s in sub_sents:
                chunks.append({
                    "index": len(chunks),
                    "text": s,
                    "character": "Narrator",
                    "speaker": book_speaker,
                    "instruct": narrative_instruct,
                })

    return chunks


def split_into_sentences(text: str) -> List[str]:
    """
    Backward-compatible sentence splitter returning simple text strings.
    """
    chunks = parse_audiobook_script(text)
    return [c["text"] for c in chunks]


def wav_to_base64_data_url(wav: Any, sr: int) -> str:
    """
    Encode float numpy waveform to standard 16-bit PCM WAV base64 data URL.
    Ensures universal cross-browser HTML5 Audio playback compatibility.
    """
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    wav = np.asarray(wav, dtype=np.float32)
    max_val = np.max(np.abs(wav))
    if max_val > 1.0:
        wav = wav / max_val
    wav_int16 = (wav * 32767.0).astype(np.int16)
    sf.write(buf, wav_int16, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"
