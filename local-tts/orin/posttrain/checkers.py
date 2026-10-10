"""Automatic checkers for TTS output: intelligibility, code-switching, voice, naturalness, sanity.

Each checker takes a list of Utterance (audio + reference text) and fills `u.scores`.
Models load lazily and are freed with .close(), so they can run one after another in limited memory.

    Checker                 metric(s)                                   model
    ASRChecker              wer/cer/mer (S, D, I counts), en_recall      funasr/paraformer-zh (zh, mixed)
                            + second-opinion error rate                 openai/whisper-large-v3 (en)
    SpeakerSimChecker       sim_ref (vs reference clips), sim_self       microsoft/wavlm-base-plus-sv
    UTMOSChecker            utmos (predicted MOS, 1-5)                   SpeechMOS utmos22_strong
    SanityChecker           duration, rate (syll/s), silences, clipping  — (signal stats)
"""
from __future__ import annotations

import gc
import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import torch

SR_ASR = 16000
CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
TOKEN = re.compile(r"[㐀-䶿一-鿿豈-﫿]|[a-z0-9]+")


@dataclass
class Utterance:
    id: str
    kind: str                  # zh | en | mixed
    text: str                  # reference text (what the TTS was asked to say)
    audio: np.ndarray          # float32 mono
    sr: int
    scores: dict = field(default_factory=dict)

    def resampled(self, sr: int) -> np.ndarray:
        if self.sr == sr:
            return self.audio
        import librosa
        return librosa.resample(self.audio, orig_sr=self.sr, target_sr=sr)


# ---------------------------------------------------------------- text normalization + error rates
try:
    from opencc import OpenCC          # opencc-python-reimplemented
    _t2s = OpenCC("t2s").convert
except ImportError:                    # Whisper sometimes answers in traditional characters
    _t2s = lambda s: s


_en_norm = None


def set_english_normalizer(whisper_dir: str):
    """Whisper's EnglishTextNormalizer (numbers -> digits, British -> American spelling, contractions).
    Applied to English-only text, so "ten or fifteen" and "10 or 15" compare equal."""
    global _en_norm
    import json
    from pathlib import Path
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    _en_norm = EnglishTextNormalizer(json.loads((Path(whisper_dir) / "normalizer.json").read_text()))


def normalize(text: str) -> str:
    """Lowercase, traditional->simplified, punctuation -> space. Same function for ref and hyp."""
    text = unicodedata.normalize("NFKC", text)
    if _en_norm is not None and not CJK.search(text):
        text = _en_norm(text)
    text = _t2s(text).lower()
    text = text.replace("’", "").replace("'", "")                  # let's == lets
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    """Mixed tokenization: each Chinese character is a token, each English word is a token.
    On zh-only text this gives CER, on en-only WER, on mixed text MER (mixed error rate)."""
    return TOKEN.findall(normalize(text))


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment; returns (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    s = dl = ins = 0
    i, j = n, m
    # Backtrace to split the distance into S / D / I. Several alignments can tie; prefer exact
    # matches, then deletions/insertions over substitutions, so skipped or repeated words show up
    # as D/I instead of a chain of shifted substitutions.
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1; i -= 1
        elif j > 0 and d[i, j] == d[i, j - 1] + 1:
            ins += 1; j -= 1
        else:
            s += 1; i -= 1; j -= 1
    return s, dl, ins


def english_recall(ref_text: str, hyp_text: str) -> tuple[int, int]:
    """Code-switching check: how many English words of the reference came back as English words."""
    ref_en = [t for t in tokens(ref_text) if not CJK.match(t)]
    hyp_en = [t for t in tokens(hyp_text) if not CJK.match(t)]
    pool, hit = list(hyp_en), 0
    for w in ref_en:
        if w in pool:
            pool.remove(w); hit += 1
    return hit, len(ref_en)


# ---------------------------------------------------------------- checkers
class Checker:
    name = "checker"

    def __call__(self, utts: list[Utterance]) -> None:
        raise NotImplementedError

    def close(self):
        for attr in list(vars(self)):
            if attr.startswith("_m"):
                setattr(self, attr, None)
        gc.collect(); torch.cuda.empty_cache()


class WhisperASR:
    """openai/whisper-large-v3. Strong on English; on code-switched speech it sometimes *translates*
    the minority-language part ("一杯咖啡" -> "a cup of coffee") instead of transcribing it."""
    def __init__(self, model_dir, device="cuda", batch_size=8):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device, self.bs = device, batch_size
        self.proc = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_dir, dtype=torch.float16).to(device).eval()

    @torch.inference_mode()
    def __call__(self, audios, kind):
        lang = "en" if kind == "en" else "zh"
        out = []
        for i in range(0, len(audios), self.bs):
            feats = self.proc(audios[i:i + self.bs], sampling_rate=SR_ASR, return_tensors="pt")
            ids = self.model.generate(feats.input_features.to(self.device, torch.float16),
                                      language=lang, task="transcribe", num_beams=1)
            out += self.proc.batch_decode(ids, skip_special_tokens=True)
        return [t.strip() for t in out]


class ParaformerASR:
    """funasr/paraformer-zh: non-autoregressive Mandarin ASR (the Seed-TTS-eval zh judge). Keeps
    English words inside Chinese speech as English, and can't "translate" — it has no decoder LM."""
    def __init__(self, model_dir, device="cuda"):
        from funasr import AutoModel
        self.model = AutoModel(model=model_dir, device=device, disable_update=True, disable_pbar=True, log_level="ERROR")

    def __call__(self, audios, kind):
        return [r["text"].strip() for r in self.model.generate(input=list(audios), batch_size=8)]


class ASRChecker(Checker):
    """Transcribe with two ASR systems and score against the reference text.

    primary (err_rate, S/D/I):  Paraformer for zh and mixed, Whisper for en — each on its strong side.
    alt (err_rate_alt):         the other system, as a second opinion.
    A token both systems get wrong is likely a real TTS error; one-sided errors are likely ASR quirks.
    The ASRs are part of the measurement: calibrate them on real human recordings (see README)."""
    name = "asr"
    PRIMARY = {"zh": "paraformer", "mixed": "paraformer", "en": "whisper"}

    def __init__(self, whisper_dir: str, paraformer_dir: str, device="cuda"):
        self.dirs = {"whisper": whisper_dir, "paraformer": paraformer_dir}
        set_english_normalizer(whisper_dir)
        self.device = device

    @staticmethod
    def score(ref_text, hyp):
        r, h = tokens(ref_text), tokens(hyp)
        s, d, i = edit_ops(r, h)
        return dict(n_ref=len(r), sub=s, dele=d, ins=i, err_rate=(s + d + i) / max(len(r), 1))

    def __call__(self, utts):
        hyps = {}                                   # engine -> {utt id: text}; engines load one at a time
        for engine, cls in (("whisper", WhisperASR), ("paraformer", ParaformerASR)):
            self._masr = cls(self.dirs[engine], self.device)
            hyps[engine] = {}
            for kind in ("zh", "mixed", "en"):
                group = [u for u in utts if u.kind == kind]
                if group:
                    for u, t in zip(group, self._masr([u.resampled(SR_ASR) for u in group], kind)):
                        hyps[engine][u.id] = t
            self.close()
        for u in utts:
            prim = self.PRIMARY[u.kind]
            alt = "whisper" if prim == "paraformer" else "paraformer"
            u.scores.update(asr=prim, asr_text=hyps[prim][u.id], **self.score(u.text, hyps[prim][u.id]),
                            asr_alt=alt, asr_text_alt=hyps[alt][u.id],
                            err_rate_alt=self.score(u.text, hyps[alt][u.id])["err_rate"])
            if u.kind == "mixed":
                hit, n = english_recall(u.text, hyps[prim][u.id])
                u.scores.update(en_hit=hit, en_n=n)


class SpeakerSimChecker(Checker):
    """Cosine similarity of WavLM x-vectors.

    sim_ref:  vs the centroid of reference recordings of the target speaker (did we get the voice?)
    sim_self: vs the centroid of this run's own outputs (is the voice consistent across zh/en/mixed?)"""
    name = "speaker"

    def __init__(self, model_dir: str, ref_audios: list[np.ndarray] | None = None, device="cuda"):
        self.model_dir, self.refs, self.device = model_dir, ref_audios or [], device

    @torch.inference_mode()
    def embed(self, audios: list[np.ndarray]) -> torch.Tensor:
        embs = []
        for a in audios:                         # one at a time: no padding effects on the x-vector
            x = self._mfe(a, sampling_rate=SR_ASR, return_tensors="pt").to(self.device)
            embs.append(torch.nn.functional.normalize(self._mmodel(**x).embeddings, dim=-1)[0])
        return torch.stack(embs)

    def __call__(self, utts):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self._mfe = AutoFeatureExtractor.from_pretrained(self.model_dir)
        self._mmodel = WavLMForXVector.from_pretrained(self.model_dir).to(self.device).eval()
        e = self.embed([u.resampled(SR_ASR) for u in utts])
        self_c = torch.nn.functional.normalize(e.mean(0), dim=-1)
        ref_c = torch.nn.functional.normalize(self.embed(self.refs).mean(0), dim=-1) if self.refs else None
        for k, u in enumerate(utts):
            u.scores["sim_self"] = float(e[k] @ self_c)
            if ref_c is not None:
                u.scores["sim_ref"] = float(e[k] @ ref_c)
        self.close()


class UTMOSChecker(Checker):
    """Predicted naturalness MOS (1-5), UTMOS22 strong learner (VoiceMOS 2022 winner) via SpeechMOS."""
    name = "utmos"

    def __init__(self, device="cuda"):
        self.device = device

    @torch.inference_mode()
    def __call__(self, utts):
        self._mmodel = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(self.device)
        for u in utts:
            wav = torch.from_numpy(u.resampled(SR_ASR)).float().unsqueeze(0).to(self.device)
            u.scores["utmos"] = float(self._mmodel(wav, SR_ASR))
        self.close()


class SanityChecker(Checker):
    """Cheap signal checks that catch truncation, run-on generation, long silences and clipping."""
    name = "sanity"
    RATE_OK = (2.0, 8.0)       # syllables per second; conversational speech is ~3-6

    @staticmethod
    def syllables(text: str) -> float:
        t = tokens(text)
        zh = sum(1 for x in t if CJK.match(x))
        en = sum(max(1, len(re.findall(r"[aeiouy]+", x))) for x in t if not CJK.match(x))
        return zh + en

    def __call__(self, utts):
        for u in utts:
            a, sr = u.audio, u.sr
            dur = len(a) / sr
            frame = int(0.02 * sr)
            rms = np.sqrt(np.mean(a[: len(a) // frame * frame].reshape(-1, frame) ** 2, axis=1) + 1e-12)
            voiced = np.where(rms > max(0.01, 0.05 * rms.max()))[0]
            lead = voiced[0] * 0.02 if len(voiced) else dur
            trail = (len(rms) - 1 - voiced[-1]) * 0.02 if len(voiced) else dur
            speech = max(dur - lead - trail, 1e-3)
            rate = self.syllables(u.text) / speech
            clip = float(np.mean(np.abs(a) > 0.99))
            flags = []
            if not self.RATE_OK[0] <= rate <= self.RATE_OK[1]:
                flags.append("rate")
            if lead > 1.0 or trail > 1.5:
                flags.append("silence")
            if clip > 0.001:
                flags.append("clipping")
            u.scores.update(duration=dur, rate=rate, lead_sil=lead, trail_sil=trail, clip=clip,
                            sanity_flags=",".join(flags))


# ---------------------------------------------------------------- aggregation
def summarize(utts: list[Utterance]) -> dict:
    """Corpus-level metrics per kind. Error rates are pooled (sum errors / sum ref tokens), not averaged."""
    out = {}
    for kind in ("zh", "en", "mixed", "all"):
        g = [u for u in utts if kind == "all" or u.kind == kind]
        if not g:
            continue
        s = {"n": len(g)}
        if all("err_rate" in u.scores for u in g):
            n = sum(u.scores["n_ref"] for u in g)
            for k in ("sub", "dele", "ins"):
                s[k] = sum(u.scores[k] for u in g)
            s["err_rate"] = (s["sub"] + s["dele"] + s["ins"]) / max(n, 1)
            s["n_ref_tokens"] = n
            if all("err_rate_alt" in u.scores for u in g):
                s["err_rate_alt"] = sum(u.scores["err_rate_alt"] * u.scores["n_ref"] for u in g) / max(n, 1)
        hits = [u for u in g if "en_n" in u.scores]
        if hits:
            s["en_recall"] = sum(u.scores["en_hit"] for u in hits) / max(sum(u.scores["en_n"] for u in hits), 1)
        for k in ("sim_ref", "sim_self", "utmos", "rate", "duration"):
            vals = [u.scores[k] for u in g if k in u.scores]
            if vals:
                s[k] = float(np.mean(vals))
        s["sanity_fail"] = sum(1 for u in g if u.scores.get("sanity_flags"))
        out[kind] = s
    return out
