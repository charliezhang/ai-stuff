"""Alignment-free pronunciation scoring with a multilingual CTC phoneme recognizer.

Pipeline (reference text is known, like Azure Pronunciation Assessment's "scripted" mode):

  text --eSpeak-NG--> canonical phonemes per word (IPA, the recognizer's own label format)
  audio --wav2vec2-xlsr-53-espeak-cv-ft--> per-frame phoneme posteriors (CTC)

  For every canonical phoneme i we ask: does the audio support the canonical sequence better than the
  same sequence with phoneme i *substituted* by another phoneme, or *deleted*? Each hypothesis is scored
  with the CTC likelihood, which sums over **all** possible time alignments — no forced alignment, which
  is fragile on children's speech (high pitch, irregular timing).

      GOP_i  = log P(canonical | audio) - max_alt log P(alt_i | audio)      (log-likelihood ratio)
      score  = 100 * sigmoid((GOP_i - BIAS) / TEMP)

  Mandarin: competitors that differ only in tone are scored separately (tone_score, experimental — the
  recognizer's tone labels are weak and merge tones 1 and 4), so tone errors don't swamp segment scores.

  GOP_i > 0: the canonical phoneme is the best explanation. The winning alternative tells you what it
  sounded like instead ("/θ/ sounded like /s/", "dropped"). Word = mean of its phonemes, sentence =
  mean of words. This is the family of CTC-based GOP methods (Cao et al. 2024; segmentation-free GOP,
  TASLP 2025; substitution-aware GOP, Interspeech 2025), simplified for a demo and *not* calibrated on
  human scores yet — TEMP/BIAS were set so native TTS speech scores ~90+ and wrong-sentence audio < 30.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LANGS = {"en": "en-us", "es": "es-419", "zh": "cmn"}
TEMP, BIAS = 2.0, 0.0          # default LLR (nats) -> 0..100 mapping; per-language overrides in calib.json
CALIB_FILE = Path(__file__).with_name("calib.json")     # written by calibrate.py
TOP_K_COMPETITORS = 48         # substitution candidates per utterance (most likely phonemes in the audio)
TONE = re.compile(r"[12345ɜ]$")  # Mandarin finals carry a tone mark (first digit of the Chao contour; ɜ = 3)
# Accepted variants per language: allophones and accent variants that are not pronunciation errors.
# Without these, natural native speech is penalized against eSpeak's dictionary form.
EQUIV = {
    "zh": [{"χ", "x"}],                                    # Mandarin h: eSpeak χ, recognizer x
    "en": [{"t", "ɾ"}, {"d", "ɾ"},                         # American flap: "cat is", "water"
           {"ɛɹ", "ɛ", "ɛə"}, {"ɑːɹ", "ɑː"}, {"ɔːɹ", "ɔː"}, {"ɪɹ", "ɪə"}, {"ʊɹ", "ʊə"}, {"ɚ", "ə"}],   # non-rhotic accents
    "es": [{"ð", "d"}, {"β", "b"}, {"ɣ", "ɡ"}],            # Spanish soft/hard stop allophones
}


def base(p: str) -> str:
    return TONE.sub("", p)


def tone_of(p: str) -> str:
    m = TONE.search(p)
    return m.group(0) if m else ""


@dataclass
class PhoneScore:
    phone: str
    score: float
    gop: float
    heard: str                 # best alternative ("" if canonical won, "∅" if deletion won)
    tone_score: float | None = None   # Mandarin only, experimental (see score())
    tone_heard: str = ""


@dataclass
class WordScore:
    word: str
    phones: list[PhoneScore]

    @property
    def score(self) -> float:
        return float(np.mean([p.score for p in self.phones])) if self.phones else 0.0


class PronunciationScorer:
    def __init__(self, model_dir: str, device: str = "cuda"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
        self.device = device
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval()
        self.vocab = json.loads((Path(model_dir) / "vocab.json").read_text())
        self.inv = {i: p for p, i in self.vocab.items()}
        self.blank = self.vocab["<pad>"]                         # CTC blank
        self.special = {self.vocab[t] for t in ("<s>", "</s>", "<unk>", "<pad>")}
        self._g2p = {}
        self.calib = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else {}

    # ------------------------------------------------------------- text -> canonical phonemes per word
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
            phones = [p for p in ph.replace("|", " ").split() if p in self.vocab]   # all in-vocab after the eSpeak fix
            out.append((w, phones))
        return [(w, p) for w, p in out if p]

    # ------------------------------------------------------------- audio -> CTC log posteriors
    @torch.inference_mode()
    def log_probs(self, audio16k: np.ndarray) -> torch.Tensor:
        x = self.fe(audio16k, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        return F.log_softmax(self.model(x).logits[0].float(), dim=-1)          # (T, V)

    def heard(self, lp: torch.Tensor) -> str:
        """Free (greedy) phoneme decoding: what the recognizer hears without the reference text."""
        ids = lp.argmax(-1).tolist()
        out, prev = [], None
        for i in ids:
            if i != prev and i not in self.special:
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
            tgt = torch.full((len(chunk), L), self.blank, dtype=torch.long)
            for k, q in enumerate(chunk):
                tgt[k, :len(q)] = torch.tensor(q)
            lens = torch.tensor([len(q) for q in chunk])
            lpb = lp.unsqueeze(1).expand(T, len(chunk), lp.shape[1])
            nll = F.ctc_loss(lpb, tgt.to(lp.device), torch.full((len(chunk),), T, dtype=torch.long), lens,
                             blank=self.blank, reduction="none", zero_infinity=True)
            out.append(-nll)
        return torch.cat(out)

    # ------------------------------------------------------------- streaming: how far into the sentence?
    def _equiv_map(self, lang: str) -> dict[int, set[int]]:
        equiv = {}
        for grp in EQUIV.get(lang, []):
            ids = {self.vocab[x] for x in grp if x in self.vocab}
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
        lp = self.log_probs(audio16k)
        covered = self.progress(lp, words, lang)
        ends = np.cumsum([len(ph) for _, ph in words])
        done = int(sum(e <= covered for e in ends))
        if done == 0:
            return {"partial": True, "done_words": 0, "n_words": len(words), "covered_phones": covered,
                    "words": [], "heard": self.heard(lp)}
        r = self.score(audio16k, text, lang, words=words[:done], lp=lp)
        r.update(partial=True, done_words=done, n_words=len(words), covered_phones=covered,
                 complete=done == len(words))
        return r

    # ------------------------------------------------------------- scoring
    def score(self, audio16k: np.ndarray, text: str, lang: str, words=None, lp=None) -> dict:
        words = words if words is not None else self.g2p(text, lang)
        canon = [self.vocab[p] for _, ph in words for p in ph]
        lp = lp if lp is not None else self.log_probs(audio16k)

        # Substitution candidates: phonemes with the most posterior mass in this audio + canonical ones.
        mass = lp.exp().sum(0)
        mass[list(self.special)] = 0
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
                # Mandarin: a competitor that differs only in tone tests the *tone*, not the segment.
                kind = "tone" if lang == "zh" and tone_of(cp) and base(qp) == base(cp) else "seg"
                variants.append(canon[:i] + [q] + canon[i + 1:]); owner.append((i, qp, kind))
        ll = self.seq_logp(lp, variants).cpu().numpy()
        base_ll = ll[0]

        best = {k: [(-np.inf, "")] * len(canon) for k in ("seg", "tone")}
        for (i, alt, kind), v in zip(owner[1:], ll[1:]):
            if v > best[kind][i][0]:
                best[kind][i] = (v, alt)

        cal = self.calib.get(lang, {})
        bias, temp = cal.get("bias", BIAS), cal.get("temp", TEMP)
        to100 = lambda g: round(100.0 / (1.0 + np.exp(-(g - bias) / temp)), 1)
        k, out_words = 0, []
        for w, ph in words:
            ps = []
            for p in ph:
                gop = float(base_ll - best["seg"][k][0])
                pscore = PhoneScore(p, to100(gop), round(gop, 2), "" if gop > 0 else best["seg"][k][1])
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
            "text": text, "lang": lang,
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
