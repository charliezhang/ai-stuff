"""Alignment-free pronunciation scoring with a multilingual CTC phoneme recognizer (pluggable backends).

Pipeline (reference text is known, like Azure Pronunciation Assessment's "scripted" mode):

  text --eSpeak-NG--> canonical phonemes per word --backend.units()--> the recognizer's own phone tokens
  audio --backend.log_probs()--> per-frame phoneme posteriors (CTC)

  For every canonical phoneme i we ask: does the audio support the canonical sequence better than the
  same sequence with phoneme i *substituted* by another phoneme, or *deleted*? Each hypothesis is scored
  with the CTC likelihood, which sums over **all** possible time alignments — no forced alignment, which
  is fragile on children's speech (high pitch, irregular timing).

      GOP_i  = log P(canonical | audio) - max_alt log P(alt_i | audio)      (log-likelihood ratio)
      score  = 100 * sigmoid((GOP_i - bias[backend][lang]) / temp)

  GOP_i > 0: the canonical phoneme is the best explanation. The winning alternative tells you what it
  sounded like instead ("/θ/ sounded like /s/", "dropped"). Word = mean of its phonemes, sentence =
  mean of words. This is the family of CTC-based GOP methods (Cao et al. 2024; segmentation-free GOP,
  TASLP 2025; substitution-aware GOP, Interspeech 2025), simplified for a demo and calibrated per
  backend and language on native TTS speech (calibrate.py), not on human scores.

Backends
  w2v    facebook/wav2vec2-xlsr-53-espeak-cv-ft (2021, 316M, Apache-2.0). Trained on eSpeak labels, so
         eSpeak phonemes are used as-is. Mandarin finals carry a (weak) tone digit -> experimental tone score.
  powsm  espnet/powsm_ctc (2026, 426M, CC-BY-4.0). E-Branchformer encoder-CTC from the POWSM / PRiSM work,
         trained on IPAPack++ (17k h). Language-prompted (<eng>/<cmn>/<spa>). One token per IPA segment
         (+ diacritics); no tones in practice, so Mandarin is scored on segments only.
"""
from __future__ import annotations

import glob
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LANGS = {"en": "en-us", "es": "es-419", "zh": "cmn"}
TEMP, BIAS = 2.0, 0.0          # default LLR (nats) -> 0..100 mapping; per backend/language in calib.json
CALIB_FILE = Path(__file__).with_name("calib.json")     # written by calibrate.py
TOP_K_COMPETITORS = 48         # substitution candidates per utterance (most likely phonemes in the audio)
TONE = re.compile(r"[12345ɜ]$")  # w2v Mandarin finals carry a tone mark (first digit of the Chao contour)
# Accepted variants: allophones and accent variants that are not pronunciation errors. Groups are written
# in each backend's own unit names; groups with units a backend doesn't have are simply ignored.
EQUIV = {
    "zh": [{"χ", "x"}, {"a", "ɑ"}],                        # Mandarin h (χ vs x); /a/ allophones
    "en": [{"t", "ɾ"}, {"d", "ɾ"},                         # American flap: "cat is", "water"
           {"ɛɹ", "ɛ", "ɛə"}, {"ɑːɹ", "ɑː"}, {"ɔːɹ", "ɔː"}, {"ɪɹ", "ɪə"}, {"ʊɹ", "ʊə"}, {"ɚ", "ə"},   # non-rhotic (w2v)
           {"ɜ", "ə˞", "ɝ", "ɚ"}, {"ɑ", "ɔ"}],             # r-colored vowel spellings; cot-caught merger (powsm)
    "es": [{"ð", "d"}, {"β", "b"}, {"ɣ", "ɡ"}],            # Spanish soft/hard stop allophones
}


def base(p: str) -> str:
    return TONE.sub("", p)


def tone_of(p: str) -> str:
    m = TONE.search(p)
    return m.group(0) if m else ""


# ===================================================================== backends
class W2VBackend:
    name, label, tonal = "w2v", "wav2vec2-XLSR-53 eSpeak (2021 · 316M)", True

    def __init__(self, model_dir: str, device: str):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval()
        self.vocab = json.loads((Path(model_dir) / "vocab.json").read_text())
        self.blank = self.vocab["<pad>"]
        self.special = {self.vocab[t] for t in ("<s>", "</s>", "<unk>", "<pad>")}

    def prefix(self, lang):                     # tokens every target starts with (none for wav2vec2)
        return []

    def units(self, phone: str, lang: str) -> list[str]:
        return [phone] if phone in self.vocab else []            # trained on eSpeak labels: 1:1

    @torch.inference_mode()
    def log_probs(self, audio16k: np.ndarray, lang: str) -> torch.Tensor:
        x = self.fe(audio16k, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        return F.log_softmax(self.model(x).logits[0].float(), dim=-1)


class POWSMBackend:
    name, label, tonal = "powsm", "POWSM-CTC (2026 · 426M)", False
    LANG_SYM = {"en": "<eng>", "zh": "<cmn>", "es": "<spa>"}
    # eSpeak -> IPAPack++ conventions, applied before segmenting (learned by decoding native speech).
    ZH_REWRITE = [("ts.h", "ʈʂ"), ("ts.", "ʈʂ"), ("s.", "ʂ"), ("z.", "ʐ"), ("tɕh", "tɕʰ"), ("tsh", "tsʰ"),
                  ("ph", "pʰ"), ("th", "tʰ"), ("kh", "kʰ"), ("i.", "ɨ"), ("i̪", "ɨ"), ("χ", "x"),
                  ("onɡ", "ʊŋ"), ("uei", "wei"), ("ua", "wa"), ("uo", "wo"), ("uə", "wə"),   # medial u = glide w
                  ("ɑu", "ɑʊ"), ("ou", "oʊ"), ("ei", "eɪ"), ("ai", "aɪ")]
    EN_REWRITE = [("ɜː", "ə˞"), ("ɚ", "ə˞"), ("ᵻ", "ɪ"), ("əl", "əl")]
    MODIFIERS = set("ʰʲʷˠˤˀ˞ⁿᶣʼ")

    def __init__(self, snapshot: str, device: str):
        from espnet2.bin.s2t_inference_ctc import Speech2Text
        exp = "exp/s2t_train_conv2d_size768_e9_d9_mel80_raw_bpe40000"
        cwd = os.getcwd()
        os.chdir(snapshot)                                       # config uses relative paths
        try:
            self.s2t = Speech2Text(s2t_train_config=f"{exp}/config.yaml",
                                   s2t_model_file=glob.glob(f"{exp}/*.pth")[0],
                                   device=device, task_sym="<pr>", lang_sym="<unk>")
        finally:
            os.chdir(cwd)
        # The BPE model is loaded lazily on first encode via a *relative* path: make it absolute.
        if getattr(self.s2t, "tokenizer", None) is not None and hasattr(self.s2t.tokenizer, "model"):
            self.s2t.tokenizer.model = str(Path(snapshot) / self.s2t.tokenizer.model)
        self.model = self.s2t.s2t_model.eval()
        tl = self.model.token_list
        # Phone tokens are "/x/"; expose them by their bare IPA so EQUIV / UI use plain symbols.
        self.vocab = {}
        for i, t in enumerate(tl):
            key = t[1:-1] if len(t) > 2 and t.startswith("/") and t.endswith("/") else t
            self.vocab.setdefault(key, i)
        self.blank = tl.index("<blank>")
        self.special = {i for i, t in enumerate(tl) if not (len(t) > 2 and t.startswith("/") and t.endswith("/"))}
        self._prefix = {l: [tl.index(s), tl.index("<pr>"), tl.index("<notimestamps>"), tl.index("▁")]
                        for l, s in self.LANG_SYM.items()}

    def prefix(self, lang):                     # the model emits <lang> <pr> <notimestamps> ▁ before phones
        return self._prefix[lang]

    def _segments(self, s: str) -> list[str]:
        """Split an IPA string into segments: base letter + combining marks / modifier letters."""
        segs = []
        for ch in s:
            if segs and (unicodedata.combining(ch) or ch in self.MODIFIERS or segs[-1].endswith("͡")):
                segs[-1] += ch
            else:
                segs.append(ch)
        return segs

    def units(self, phone: str, lang: str) -> list[str]:
        p = phone
        if lang == "zh":
            p = TONE.sub("", p)
            for a, b in self.ZH_REWRITE:
                p = p.replace(a, b)
        elif lang == "en":
            for a, b in self.EN_REWRITE:                         # before dropping ː ("ɜː" -> "ə˞")
                p = p.replace(a, b)
        p = p.replace("ː", "")                                   # IPAPack++ en/es/zh: no length marks
        out = []
        for seg in self._segments(p):
            while seg and seg not in self.vocab:                 # drop diacritics the inventory lacks
                seg = seg[:-1]
            if seg:
                out.append(seg)
        return out

    PAD_SECONDS = float(os.environ.get("POWSM_PAD_SECONDS", "20"))   # trained on fixed 20 s inputs
    # The model emits its prompt (<lang> <pr> <notimestamps> ▁) in the first ~4 frames (160 ms). If speech
    # starts right away, the prompt crowds out the first phoneme and the first word scores low. Leading
    # silence gives the prompt room.
    LEAD_SECONDS = float(os.environ.get("POWSM_LEAD_SECONDS", "0.3"))

    @torch.inference_mode()
    def log_probs(self, audio16k: np.ndarray, lang: str) -> torch.Tensor:
        a = np.concatenate([np.zeros(int(self.LEAD_SECONDS * 16000), np.float32), audio16k])
        if self.PAD_SECONDS and len(a) < self.PAD_SECONDS * 16000:
            a = np.pad(a, (0, int(self.PAD_SECONDS * 16000) - len(a)))
        enc, _ = self.s2t._encode(torch.tensor(a, dtype=torch.float32).unsqueeze(0), None,
                                  self.LANG_SYM[lang], "<pr>")
        return self.model.ctc.log_softmax(enc)[0].float()


# ===================================================================== results
@dataclass
class PhoneScore:
    phone: str
    score: float
    gop: float
    heard: str                 # best alternative ("" if canonical won, "∅" if deletion won)
    tone_score: float | None = None   # w2v Mandarin only, experimental (see score())
    tone_heard: str = ""


@dataclass
class WordScore:
    word: str
    phones: list[PhoneScore]

    @property
    def score(self) -> float:
        return float(np.mean([p.score for p in self.phones])) if self.phones else 0.0


# ===================================================================== scorer
class PronunciationScorer:
    def __init__(self, backend):
        self.b = backend
        self.inv = {i: p for p, i in backend.vocab.items()}
        self._g2p = {}
        calib = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}
        self.calib = calib.get(backend.name, {})

    # ------------------------------------------------------------- text -> canonical units per word
    def g2p(self, text: str, lang: str) -> list[tuple[str, list[str]]]:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        if lang not in self._g2p:
            self._g2p[lang] = EspeakBackend(LANGS[lang], preserve_punctuation=False, with_stress=False,
                                            language_switch="remove-flags")
        if lang == "zh":                                         # Chinese has no spaces: segment words
            import jieba
            words = [w for w in jieba.lcut(re.sub(r"[^一-鿿]", "", text)) if w.strip()]
        else:
            words = [w for w in re.findall(r"[\w'’áéíóúüñÁÉÍÓÚÜÑ]+", text)]
        sep = Separator(phone=" ", word="|", syllable=None)   # eSpeak may split one jieba word (喜欢 -> 喜|欢)
        out = []
        for w, ph in zip(words, self._g2p[lang].phonemize(words, separator=sep, strip=True)):
            units = [u for p in ph.replace("|", " ").split() for u in self.b.units(p, lang)]
            out.append((w, units))
        return [(w, u) for w, u in out if u]

    def heard(self, lp: torch.Tensor) -> str:
        """Free (greedy) phoneme decoding: what the recognizer hears without the reference text."""
        ids = lp.argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i not in self.b.special:
                out.append(self.inv[i])
            prev = i
        return " ".join(out)

    @torch.inference_mode()
    def seq_logp(self, lp: torch.Tensor, seqs: list[list[int]], batch: int = 512) -> torch.Tensor:
        """CTC log-likelihood of each target sequence (sums over all alignments)."""
        T = lp.shape[0]
        out = []
        for s in range(0, len(seqs), batch):
            chunk = seqs[s:s + batch]
            L = max(len(q) for q in chunk) or 1
            tgt = torch.full((len(chunk), L), self.b.blank, dtype=torch.long)
            for k, q in enumerate(chunk):
                tgt[k, :len(q)] = torch.tensor(q)
            lens = torch.tensor([len(q) for q in chunk])
            lpb = lp.unsqueeze(1).expand(T, len(chunk), lp.shape[1])
            # zero_infinity=False: a target that can't fit in T frames (e.g. a long sentence vs a short
            # clip with 40 ms frames) has probability 0, i.e. log P = -inf. Zeroing it would claim P = 1.
            nll = F.ctc_loss(lpb, tgt.to(lp.device), torch.full((len(chunk),), T, dtype=torch.long), lens,
                             blank=self.b.blank, reduction="none", zero_infinity=False)
            out.append(torch.nan_to_num(-nll, nan=-np.inf, posinf=np.inf, neginf=-np.inf))   # keep ±inf
        return torch.cat(out)

    # ------------------------------------------------------------- streaming: how far into the sentence?
    def _equiv_map(self, lang: str) -> dict[int, set[int]]:
        equiv = {}
        for grp in EQUIV.get(lang, []):
            ids = {self.b.vocab[x] for x in grp if x in self.b.vocab}
            for i in ids:
                equiv.setdefault(i, set()).update(ids)
        return equiv

    def progress(self, lp: torch.Tensor, words: list[tuple[str, list[str]]], lang: str) -> int:
        """Estimate how many canonical phonemes the speaker has covered so far.

        Semi-global edit distance between the freely decoded phonemes (what was heard) and every prefix of
        the canonical sequence; the best-matching prefix length is the progress. Phonemes compare by base
        (tone stripped) and accepted variants, so a wrong tone or an American flap doesn't stall progress."""
        heard = self.heard(lp).split()
        canon = [p for _, ph in words for p in ph]
        if not heard or not canon:
            return 0
        eq = {self.inv[a]: {self.inv[b] for b in bs} for a, bs in self._equiv_map(lang).items()}
        same = lambda h, c: base(h) == base(c) or h in eq.get(c, ())
        n, m = len(heard), len(canon)
        d = np.zeros((n + 1, m + 1))
        d[0, :] = np.arange(m + 1)
        d[:, 0] = np.arange(n + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (not same(heard[i - 1], canon[j - 1])))
        row = d[n]
        return int(max(j for j in range(m + 1) if row[j] == row.min()))     # ties -> furthest prefix

    def score_stream(self, audio16k: np.ndarray, text: str, lang: str) -> dict:
        """Partial result while the learner is still speaking: score only the words already covered.
        The word in progress is excluded; the last completed word is provisional (its boundary with the
        next word isn't certain until more audio arrives)."""
        words = self.g2p(text, lang)
        lp = self.b.log_probs(audio16k, lang)
        covered = self.progress(lp, words, lang)
        ends = np.cumsum([len(ph) for _, ph in words])
        done = int(sum(e <= covered for e in ends))
        if done == 0:
            return {"partial": True, "done_words": 0, "n_words": len(words), "covered_phones": covered,
                    "words": [], "heard": self.heard(lp), "model": self.b.name}
        r = self.score(audio16k, text, lang, words=words[:done], lp=lp)
        r.update(partial=True, done_words=done, n_words=len(words), covered_phones=covered,
                 complete=done == len(words))
        return r

    # ------------------------------------------------------------- scoring
    def score(self, audio16k: np.ndarray, text: str, lang: str, words=None, lp=None) -> dict:
        words = words if words is not None else self.g2p(text, lang)
        canon = [self.b.vocab[p] for _, ph in words for p in ph]
        lp = lp if lp is not None else self.b.log_probs(audio16k, lang)
        pre = self.b.prefix(lang)

        # Substitution candidates: phonemes with the most posterior mass in this audio + canonical ones.
        mass = lp.exp().sum(0)
        mass[list(self.b.special)] = 0
        comp = set(mass.topk(TOP_K_COMPETITORS).indices.tolist()) | set(canon)

        equiv = self._equiv_map(lang)
        variants, owner = [canon], [(-1, "", "seg")]
        for i, c in enumerate(canon):
            variants.append(canon[:i] + canon[i + 1:]); owner.append((i, "∅", "seg"))     # deletion
            cp = self.inv[c]
            for q in comp:
                if q == c or q in equiv.get(c, ()):
                    continue
                qp = self.inv[q]
                # w2v Mandarin: a competitor that differs only in tone tests the *tone*, not the segment.
                kind = "tone" if self.b.tonal and lang == "zh" and tone_of(cp) and base(qp) == base(cp) else "seg"
                variants.append(canon[:i] + [q] + canon[i + 1:]); owner.append((i, qp, kind))
        ll = self.seq_logp(lp, [pre + v for v in variants]).cpu().numpy()
        base_ll = ll[0]

        best = {k: [(-np.inf, "")] * len(canon) for k in ("seg", "tone")}
        for (i, alt, kind), v in zip(owner[1:], ll[1:]):
            if v > best[kind][i][0]:
                best[kind][i] = (v, alt)

        cal = self.calib.get(lang, {})
        bias, temp = cal.get("bias", BIAS), cal.get("temp", TEMP)

        def to100(g):
            if not np.isfinite(g):                  # canonical impossible in this audio -> 0; alts impossible -> 100
                return 0.0 if g < 0 or np.isnan(g) else 100.0
            return round(100.0 / (1.0 + np.exp(-(g - bias) / temp)), 1)
        k, out_words = 0, []
        for w, ph in words:
            ps = []
            for p in ph:
                gop = float(base_ll - best["seg"][k][0]) if np.isfinite(base_ll) else -np.inf
                pscore = PhoneScore(p, to100(gop), round(gop, 2) if np.isfinite(gop) else (-99.0 if gop < 0 else 99.0),
                                    "" if gop > 0 else best["seg"][k][1])
                if best["tone"][k][1]:                      # tonal syllable with tone competitors
                    tg = float(base_ll - best["tone"][k][0])
                    pscore.tone_score = to100(tg)
                    pscore.tone_heard = "" if tg > 0 else tone_of(best["tone"][k][1])
                ps.append(pscore)
                k += 1
            out_words.append(WordScore(w, ps))

        word_scores = [ws.score for ws in out_words]
        accuracy = float(np.mean(word_scores)) if word_scores else 0.0
        completeness = 100.0 * np.mean([s >= 30 for s in word_scores]) if word_scores else 0.0
        return {
            "text": text, "lang": lang, "model": self.b.name,
            "overall": round(max(1.0, 0.8 * accuracy + 0.2 * completeness), 1),
            "accuracy": round(accuracy, 1), "completeness": round(completeness, 1),
            "words": [{"word": ws.word, "score": round(ws.score, 1),
                       "phones": [vars(p) for p in ws.phones]} for ws in out_words],
            "expected": " | ".join(" ".join(ph) for _, ph in words),
            "tone": (round(float(np.mean(t)), 1) if (t := [p.tone_score for ws in out_words for p in ws.phones
                                                         if p.tone_score is not None]) else None),
            "heard": self.heard(lp),
            "n_hypotheses": len(variants), "duration_s": round(len(audio16k) / 16000, 2),
        }


# ===================================================================== loading
HF = "/hf/hub"
BACKENDS = {
    "w2v": lambda dev: W2VBackend(glob.glob(f"{HF}/models--facebook--wav2vec2-xlsr-53-espeak-cv-ft/snapshots/*")[0], dev),
    "powsm": lambda dev: POWSMBackend(glob.glob(f"{HF}/models--espnet--powsm_ctc/snapshots/*")[0], dev),
}


def load(name: str, device: str = "cuda") -> PronunciationScorer:
    return PronunciationScorer(BACKENDS[name](device))
