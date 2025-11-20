import streamlit as st
import json
import os
import io
import re
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import httpx
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import faiss
from dataclasses import dataclass
import plotly.express as px
import google.generativeai as genai
from mistralai import Mistral

# -------------------------
# Basic logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resume_ranker_app")

# -------------------------
# Configuration dataclasses
# -------------------------
@dataclass
class Weights:
    dense: float = 0.15
    keyword: float = 0.05
    skill: float = 0.45
    experience: float = 0.20
    domain: float = 0.15
    recency: float = 0.10
    metadata: float = 0.05
    projects: float = 0.05
    education: float = 0.05

@dataclass
class PipelineConfig:
    # task-specific LLM models
    llm_model_job: str = "gemini-1.5-pro"
    llm_model_resume: str = "gemini-1.5-pro"
    llm_model_skill: str = "mistral-small-latest"
    llm_model_explain: str = "gemini-1.5-flash"

    # legacy fields kept so rest of code compiles (not used directly now)
    llm_model: str = "gemini-1.5-flash"
    llm_base_url: str = ""

    embed_model: str = "BAAI/bge-large-en-v1.5"
    embed_batch: int = 32
    weights: Weights = Weights()
    bm25_min_doc_freq: int = 1

cfg = PipelineConfig()

# -------------------------
# Cached resource loaders
# -------------------------
@st.cache_resource
def load_main_embedder(model_name: str = None):
    model = model_name or cfg.embed_model
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logger.info("Loading main embedder '%s' on device=%s", model, device)
    return SentenceTransformer(model, device=device)

@st.cache_resource
def load_skill_embedder(model_name: str = "BAAI/bge-small-en-v1.5"):
    # keep skill embedder on CPU to conserve GPU memory
    device = "cpu"
    logger.info("Loading skill embedder '%s' on device=%s", model_name, device)
    return SentenceTransformer(model_name, device=device)

@st.cache_resource
def create_faiss_index(dim: int):
    logger.info("Initializing FAISS IndexFlatIP (dim=%s)", dim)
    return faiss.IndexFlatIP(dim)

# -------------------------
# Hybrid Skill Normalizer (uses cached skill embedder)
# -------------------------
class HybridSkillNormalizer:
    def __init__(self, llm_model: str = None, embed_model: str = None, threshold: float = 0.82):
        self.llm_model = llm_model or cfg.llm_model
        self.threshold = threshold
        embed_model = embed_model or "BAAI/bge-small-en-v1.5"
        self.embedder = load_skill_embedder(embed_model)
        self.cache: Dict[str, np.ndarray] = {}
        self.alias_map: Dict[str, str] = {}
        self.reverse_groups: Dict[str, List[str]] = {}

    def _llm_normalize(self, skill: str) -> str:
        prompt = f"Normalize the following skill into a concise canonical skill token. Return only the canonical skill name:\n\"{skill}\""
        try:
            if not getattr(cfg, "llm_base_url", None):
                return skill.strip().lower()
            url = cfg.llm_base_url.rstrip("/") + "/api/chat"
            payload = {"model": self.llm_model, "messages": [{"role": "user", "content": prompt}], "stream": False}
            r = httpx.post(url, json=payload, timeout=20)
            r.raise_for_status()
            data = r.json()
            out = ""
            if isinstance(data, dict):
                if "message" in data and isinstance(data["message"], dict):
                    out = data["message"].get("content", "")
                elif "response" in data:
                    out = data.get("response", "")
                else:
                    out = str(data)
            else:
                out = str(data)
            out = out.splitlines()[0].strip().lower()
            out = re.sub(r"[^a-z0-9_\-\s\.]+", "", out)
            if out == "":
                return skill.strip().lower()
            return out
        except Exception as e:
            logger.debug("LLM normalization failed for '%s': %s", skill, e)
            return skill.strip().lower()

    def _embed(self, text: str) -> np.ndarray:
        v = self.embedder.encode([text], normalize_embeddings=True)
        return np.asarray(v[0], dtype=np.float32)

    def _find_similar(self, vec: np.ndarray):
        if not self.cache:
            return None, 0.0
        keys = list(self.cache.keys())
        mat = np.vstack([self.cache[k] for k in keys])
        sims = util.cos_sim(vec, mat)[0]
        best_idx = int(np.argmax(sims))
        return keys[best_idx], float(sims[best_idx])

    def normalize_skill(self, s: str) -> str:
        if not s or not isinstance(s, str):
            return ""
        original = s.strip().lower()
        if original in self.alias_map:
            return self.alias_map[original]
        cleaned = self._llm_normalize(original)
        vec = self._embed(cleaned)
        best, score = self._find_similar(vec)
        if best and score >= self.threshold:
            self.alias_map[original] = best
            self.reverse_groups.setdefault(best, []).append(original)
            return best
        self.cache[cleaned] = vec
        self.alias_map[original] = cleaned
        self.reverse_groups.setdefault(cleaned, []).append(original)
        return cleaned

    def normalize_list(self, skills: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for s in skills or []:
            if not isinstance(s, str):
                continue
            try:
                canon = self.normalize_skill(s)
            except Exception:
                canon = s.strip().lower()
            if canon and canon not in seen:
                seen.add(canon)
                out.append(canon)
        return out

@st.cache_resource
def get_skill_normalizer():
    return HybridSkillNormalizer(embed_model="BAAI/bge-small-en-v1.5")

SKILL_NORMALIZER: HybridSkillNormalizer = get_skill_normalizer()

# -------------------------
# Document reading helpers
# -------------------------
def read_document(blob: Dict[str, Any]) -> str:
    name = blob.get("name", "").lower()
    data = blob.get("bytes", b"")
    if not data:
        return ""
    if name.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    if name.endswith(".pdf"):
        try:
            stream = io.BytesIO(data)
            reader = PdfReader(stream)
            text = ""
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
            return text.strip()
        except Exception as e:
            logger.warning("PDF extraction error for %s: %s", name, e)
            return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""

# -------------------------
# LLM call helper
# -------------------------
def call_llm(prompt: str, model: str = None, timeout: int = 60) -> str:
    model = model or cfg.llm_model
    url = cfg.llm_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"num_predict": 512, "temperature": 0.1, "top_p": 0.9},
    }
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "")
            if "response" in data:
                return data["response"]
        return str(data)
    except Exception as e:
        logger.error("[LLM] call failed: %s", e)
        return ""

# -------------------------
# Safe JSON extraction
# -------------------------
def safe_json_extract(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    raw = raw.strip()
    if raw.startswith("```"):
        lines = []
        for line in raw.splitlines():
            if line.strip().startswith("```"):
                continue
            lines.append(line)
        raw = "\n".join(lines)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidate = raw[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            try:
                fixed = candidate.replace("'", '"')
                fixed = re.sub(r",\s*}", "}", fixed)
                fixed = re.sub(r",\s*]", "]", fixed)
                return json.loads(fixed)
            except Exception:
                return {}
    return {}

# -------------------------
# Job extraction
# -------------------------
def extract_job_struct(text: str) -> Dict[str, Any]:
    prompt = f"""
You are a JSON extractor. From the job posting text below extract a JSON object with fields:
- job_title (string)
- summary (short string)
- must (list of strings)
- important (list of strings)
- nice (list of strings)
- implicit (list of strings)
- domain (list of domains/tags)

Return only valid JSON. Text:
\"\"\"{text}\"\"\"
"""
    raw = call_llm_gemini(prompt, model=cfg.llm_model_resume, timeout=90)
    parsed = safe_json_extract(raw)
    raw_tiers = {
        "must": parsed.get("must", parsed.get("must_have", [])) or [],
        "important": parsed.get("important", parsed.get("important_skills", [])) or [],
        "nice": parsed.get("nice", parsed.get("nice_to_have", [])) or [],
        "implicit": parsed.get("implicit", []) or [],
    }
    norm_tiers = {tier: SKILL_NORMALIZER.normalize_list(raw_tiers.get(tier, [])) for tier in ["must", "important", "nice", "implicit"]}
    return {
        "job_title": parsed.get("job_title", "").strip(),
        "summary": parsed.get("summary", "").strip(),
        "must": norm_tiers["must"],
        "important": norm_tiers["important"],
        "nice": norm_tiers["nice"],
        "implicit": norm_tiers["implicit"],
        "domain": parsed.get("domain", []) or [],
    }

# -------------------------
# Basic regex resume parser
# -------------------------
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
YEARS_RE = re.compile(r"(\d{4})[–-](\d{4}|present|Present|Now|now)")

def regex_extract_basic(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    name = ""
    if lines:
        first = lines[0]
        if len(first.split()) <= 4 and re.search(r"[A-Za-z]", first):
            name = first
        else:
            for i, ln in enumerate(lines):
                if EMAIL_RE.search(ln) and i > 0:
                    name = lines[i - 1]
                    break
    email_match = EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else ""
    skill_candidates = set()
    for ln in lines:
        if re.search(r"\b(Skill|Skills|TECH|TECHNICAL|Technologies)\b", ln, re.I) or ("," in ln and len(ln.split(",")) <= 15):
            for token in re.split(r"[,:;\|\n]", ln):
                token = token.strip()
                if token and len(token) < 60:
                    skill_candidates.add(token)
    yrs = 0.0
    years = YEARS_RE.findall(text)
    if years:
        tot = 0
        count = 0
        for s, e in years:
            try:
                sy = int(s)
                ey = datetime.utcnow().year if re.match(r"(?i)present|now", e) else int(e)
                tot += max(0, ey - sy)
                count += 1
            except:
                pass
        if count:
            yrs = tot / count
    return {
        "name": name,
        "email": email,
        "skills": sorted(list(skill_candidates))[:200],
        "years_est": yrs,
        "raw": text,
    }

# -------------------------
# LLM-based resume extraction
# -------------------------
def extract_resume_struct(text: str, use_llm=True) -> Dict[str, Any]:
    basic = regex_extract_basic(text)
    if use_llm:
        prompt = f"""
You are an extractor. Given the resume text, return a JSON with:
- name
- email
- skills (list of strings)
- experience (list of {{company, title, start, end, years}})
- projects (list of short descriptions)
- domain (list of tags)
- last_active (year or string)
- years_est (float)
Return only JSON. Resume:
\"\"\"{text}\"\"\"
"""
        raw = call_llm(prompt, model=cfg.llm_model, timeout=90)
        parsed = safe_json_extract(raw)
        skills = parsed.get("skills") or basic.get("skills") or []
        skills = [s.strip() for s in skills if isinstance(s, str) and s.strip()]
        skills = SKILL_NORMALIZER.normalize_list(skills)
        skills_bool = {s: True for s in skills}
        return {
            "name": parsed.get("name") or basic.get("name") or "",
            "email": parsed.get("email") or basic.get("email") or "",
            "skills": skills,
            "skills_bool": skills_bool,
            "experience": parsed.get("experience", []),
            "projects": parsed.get("projects", []),
            "domain": parsed.get("domain", []),
            "last_active": parsed.get("last_active") or "",
            "years_est": parsed.get("years_est") or basic.get("years_est") or 0,
            "raw": text,
        }
    else:
        skills = SKILL_NORMALIZER.normalize_list(basic.get("skills", []))
        return {
            "name": basic.get("name", ""),
            "email": basic.get("email", ""),
            "skills": skills,
            "skills_bool": {s: True for s in skills},
            "experience": [],
            "projects": [],
            "domain": [],
            "last_active": "",
            "years_est": basic.get("years_est", 0),
            "raw": text,
        }

# -------------------------
# Embedding + FAISS services (cached)
# -------------------------
class EmbeddingService:
    def __init__(self, model_name: str = None):
        model_name = model_name or cfg.embed_model
        self.model = load_main_embedder(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dim: %s", self.dim)
        self._faiss_index = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self.model.encode(texts, batch_size=cfg.embed_batch, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype(np.float32)

    def create_faiss(self):
        if self._faiss_index is None:
            self._faiss_index = create_faiss_index(self.dim)
        return self._faiss_index

@st.cache_resource
def get_embedding_service():
    return EmbeddingService(cfg.embed_model)

EMB: EmbeddingService = get_embedding_service()

class FAISSService:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = create_faiss_index(dim)
        self.ids: List[str] = []
        self.meta: Dict[str, Dict[str, Any]] = {}

    def reset(self):
        self.index = create_faiss_index(self.dim)
        self.ids = []
        self.meta = {}

    def add(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]):
        if vectors is None or vectors.shape[0] == 0:
            return
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        self.index.add(vecs)
        for p in payloads:
            self.ids.append(p["id"])
            self.meta[p["id"]] = p

    def search(self, query_vec: np.ndarray, top_k: int = 10):
        if self.index.ntotal == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        D, I = self.index.search(q, top_k)
        hits = []
        for score, idx in zip(D[0], I[0]):
            # Guard against invalid indices that can appear in FAISS results
            if idx < 0 or idx >= len(self.ids):
                continue
            rid = self.ids[idx]
            hits.append({"id": rid, "payload": self.meta.get(rid, {}), "score": float(score)})
        return hits

FA = FAISSService(EMB.dim)

# -------------------------
# BM25 helper
# -------------------------
def safe_build_bm25(docs: List[str]):
    try:
        tokenized = [re.findall(r"\w+", d.lower()) for d in docs]
        if not any(tokenized):
            return None
        return BM25Okapi(tokenized)
    except Exception as e:
        logger.warning("BM25 build failed: %s", e)
        return None

# -------------------------
# Scoring helpers
# -------------------------
def normalize_scores(raw: List[float]) -> List[float]:
    arr = np.array(raw, dtype=np.float32)
    if arr.size == 0:
        return []
    minv, maxv = float(np.min(arr)), float(np.max(arr))
    if abs(maxv - minv) < 1e-9:
        return [1.0 for _ in arr.tolist()]
    norm = (arr - minv) / (maxv - minv)
    return norm.tolist()

def skill_tier_score(candidate_skills: List[str], job_tiers: Dict[str, List[str]]) -> float:
    if not candidate_skills:
        return 0.0
    cand = {SKILL_NORMALIZER.normalize_skill(s) for s in candidate_skills if s}
    cand.discard("")
    must = SKILL_NORMALIZER.normalize_list(job_tiers.get("must", []))
    imp = SKILL_NORMALIZER.normalize_list(job_tiers.get("important", []))
    nice = SKILL_NORMALIZER.normalize_list(job_tiers.get("nice", []))
    impl = SKILL_NORMALIZER.normalize_list(job_tiers.get("implicit", []))
    tier_weights = {"must": 1.0, "important": 0.7, "nice": 0.3, "implicit": 0.15}
    def coverage(tier_skills: List[str]) -> float:
        if not tier_skills:
            return 0.0
        present = sum(1.0 for s in tier_skills if s in cand)
        return present / max(1, len(tier_skills))
    must_cov = coverage(must)
    imp_cov = coverage(imp)
    nice_cov = coverage(nice)
    impl_cov = coverage(impl)
    total_weight = ((tier_weights["must"] if must else 0.0) + (tier_weights["important"] if imp else 0.0) + (tier_weights["nice"] if nice else 0.0) + (tier_weights["implicit"] if impl else 0.0))
    if total_weight <= 0:
        base_score = 0.0
    else:
        base_score = (must_cov * tier_weights["must"] + imp_cov * tier_weights["important"] + nice_cov * tier_weights["nice"] + impl_cov * tier_weights["implicit"]) / total_weight
    if must:
        missing = sum(1 for s in must if s not in cand)
        missing_fraction = missing / max(1, len(must))
        penalty = 0.5 * missing_fraction
        final_score = base_score * (1.0 - penalty)
    else:
        final_score = base_score
    return float(max(min(final_score, 1.0), 0.0))

def domain_score(candidate_domains: List[str], job_domains: List[str]) -> float:
    if not job_domains:
        return 0.0
    c = {d.lower() for d in (candidate_domains or [])}
    j = {d.lower() for d in (job_domains or [])}
    if not j:
        return 0.0
    return len(c & j) / max(1, len(j))

def experience_score(years: float) -> float:
    return min(max(float(years or 0), 0.0) / 10.0, 1.0)

def recency_score(last_active) -> float:
    try:
        y = int(str(last_active).strip())
        gap = max(0, datetime.utcnow().year - y)
        if gap <= 1:
            return 1.0
        if gap <= 3:
            return 0.8
        if gap <= 5:
            return 0.5
        return 0.2
    except:
        return 0.5

def metadata_score(structured: Dict[str, Any]) -> float:
    score = 0.0
    score += 0.5 if structured.get("email") else 0.0
    score += 0.5 if structured.get("projects") else 0.0
    return min(score, 1.0)

def projects_score(structured: Dict[str, Any]) -> float:
    projects = structured.get("projects") or []
    if not projects:
        return 0.0
    n = len(projects)
    return float(min(n / 5.0, 1.0))

def education_score(structured: Dict[str, Any]) -> float:
    text = (structured.get("raw") or "") + "\n" + " ".join(structured.get("projects", []))
    text = text.lower()
    if "phd" in text or "ph.d" in text:
        return 1.0
    if "master" in text or "msc" in text or "m.sc" in text:
        return 0.8
    if "bachelor" in text or "bsc" in text or "b.sc" in text:
        return 0.6
    if "diploma" in text or "associate" in text:
        return 0.4
    return 0.0

# -------------------------
# Explainability
# -------------------------
def generate_explanation(job_struct: Dict[str, Any], candidate: Dict[str, Any], components: Dict[str, float]) -> Dict[str, Any]:
    job_tiers = {
        "must": SKILL_NORMALIZER.normalize_list(job_struct.get("must", [])),
        "important": SKILL_NORMALIZER.normalize_list(job_struct.get("important", [])),
        "nice": SKILL_NORMALIZER.normalize_list(job_struct.get("nice", [])),
        "implicit": SKILL_NORMALIZER.normalize_list(job_struct.get("implicit", [])),
    }
    structured = candidate.get("structured", candidate)
    cand_skills = SKILL_NORMALIZER.normalize_list(structured.get("skills", []))
    cand_set = set(cand_skills)
    strengths = sorted([s for s in job_tiers.get("must", []) if s in cand_set])
    gaps = sorted([s for s in job_tiers.get("must", []) if s not in cand_set])
    projects = structured.get("projects", [])
    exp_summary = f"{structured.get('years_est', 0)} years; last active: {structured.get('last_active','N/A')}"
    breakdown = components
    weight_table = {
        "dense": cfg.weights.dense,
        "keyword": cfg.weights.keyword,
        "skill": cfg.weights.skill,
        "experience": cfg.weights.experience,
        "domain": cfg.weights.domain,
        "recency": cfg.weights.recency,
        "projects": cfg.weights.projects,
        "education": cfg.weights.education,
        "metadata": cfg.weights.metadata,
    }
    overall = sum(components[k] * weight_table.get(k, 0.0) for k in components)
    if overall >= 0.75:
        confidence = "high"
    elif overall >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    prompt = f"""
You are an experienced engineer and hiring panelist reviewing a candidate's RESUME.
The job description is context; your priority is the candidate's actual experience and how it matches this specific role.

Job (context):
- Title: {job_struct.get('job_title') or ''}
- Summary: {job_struct.get('summary') or ''}

Candidate (focus of the review):
- Name: {structured.get('name') or candidate.get('name') or ''}
- Normalized skills: {cand_skills}
- Projects (from resume): {projects}
- Experience & recency: {exp_summary}

Analysis data:
- Component scores (0–1): {breakdown}
- Weights: {weight_table}
- Overall fit score (0–1): {overall:.3f}
- Job must-have strengths (present in resume): {strengths}
- Job must-have gaps (missing from resume): {gaps}
- Confidence bucket: {confidence}

Write a short, job-specific review (3–5 sentences) as if you were giving feedback to a hiring manager:
1. Describe the candidate's profile based on the RESUME (tech/domain stack, type of projects, seniority).
2. Relate how this profile lines up with the job at a high level (strong match, partial match, or stretch) for THIS specific role.
3. Explicitly mention 2–4 concrete strengths from their resume and 1–2 key gaps relevant to the job.
4. Mention the overall fit score (0–1) and confidence level ({confidence}) in a natural way.
Avoid boilerplate phrases and be concise.
"""
    human_text = call_llm_gemini(prompt, model=cfg.llm_model_explain, timeout=45)
    structured_out = {
        "strengths": strengths,
        "gaps": gaps,
        "experience_summary": exp_summary,
        "projects": projects,
        "score_breakdown": breakdown,
        "weights": weight_table,
        "overall": float(overall),
        "confidence": confidence,
    }
    return {"structured": structured_out, "human": human_text}

# -------------------------
# Ranking pipeline
# -------------------------
def rank_hybrid(job: Dict[str, Any], resumes: List[Dict[str, Any]], cfg: PipelineConfig, top_n: int = 10):
    resume_texts = [r["raw"] for r in resumes]
    resume_vecs = EMB.encode(resume_texts) if len(resume_texts) else np.zeros((0, EMB.dim), dtype=np.float32)
    job_vec = EMB.encode([job["raw"]])[0] if job["raw"].strip() else np.zeros((EMB.dim,), dtype=np.float32)
    FA.reset()
    payloads = [{"id": r["id"], "structured": r["structured"], "skills": r.get("skills", [])} for r in resumes]
    FA.add(resume_vecs, payloads)
    dense_raw = [0.0] * len(resumes)
    hits = FA.search(job_vec, top_k=len(resumes))
    id2idx = {r["id"]: i for i, r in enumerate(resumes)}
    for h in hits:
        idx = id2idx.get(h["id"])
        if idx is not None:
            dense_raw[idx] = h["score"]
    dense_norm = normalize_scores(dense_raw)
    bm25 = safe_build_bm25(resume_texts)
    if bm25:
        job_tokens = re.findall(r"\w+", job["raw"].lower())
        bm_raw = bm25.get_scores(job_tokens)
    else:
        bm_raw = [0.0] * len(resumes)
    skill_raw, domain_raw, exp_raw, rec_raw, proj_raw, edu_raw, meta_raw = [], [], [], [], [], [], []
    for r in resumes:
        structured = r.get("structured", {})
        s = skill_tier_score(structured.get("skills", []), {
            "must": job["structured"].get("must", []),
            "important": job["structured"].get("important", []),
            "nice": job["structured"].get("nice", []),
            "implicit": job["structured"].get("implicit", []),
        })
        d = domain_score(r.get("domain", []), job["structured"].get("domain", []))
        e = experience_score(r.get("years_est", 0) or 0)
        rc = recency_score(r.get("last_active", "")) if r.get("last_active") else recency_score(r.get("years_est", 0))
        p = projects_score(structured)
        edu = education_score(structured)
        m = metadata_score(structured)
        skill_raw.append(s)
        domain_raw.append(d)
        exp_raw.append(e)
        rec_raw.append(rc)
        proj_raw.append(p)
        edu_raw.append(edu)
        meta_raw.append(m)
    dense = dense_norm or [0.0] * len(resumes)
    keyword = normalize_scores(bm_norm) or [0.0] * len(resumes)
    skill = normalize_scores(skill_raw) or [0.0] * len(resumes)
    domain = normalize_scores(domain_raw) or [0.0] * len(resumes)
    experience = normalize_scores(exp_raw) or [0.0] * len(resumes)
    recency = normalize_scores(rec_raw) or [0.0] * len(resumes)
    projects = normalize_scores(proj_raw) or [0.0] * len(resumes)
    education = normalize_scores(edu_raw) or [0.0] * len(resumes)
    metadata = normalize_scores(meta_raw) or [0.0] * len(resumes)
    final = []
    W = cfg.weights
    for i, r in enumerate(resumes):
        components = {
            "dense": float(dense[i]),
            "keyword": float(keyword[i]),
            "skill": float(skill[i]),
            "experience": float(experience[i]),
            "domain": float(domain[i]),
            "recency": float(recency[i]),
            "projects": float(projects[i]),
            "education": float(education[i]),
            "metadata": float(metadata[i]),
        }
        overall = (
            components["dense"] * W.dense
            + components["keyword"] * W.keyword
            + components["skill"] * W.skill
            + components["experience"] * W.experience
            + components["domain"] * W.domain
            + components["recency"] * W.recency
            + components["projects"] * W.projects
            + components["education"] * W.education
            + components["metadata"] * W.metadata
        )
        expl = generate_explanation(job["structured"], r, components)
        final.append(
            {
                "id": r["id"],
                "name": r["structured"].get("name", r.get("name", "")),
                "overall": float(overall),
                "components": components,
                "explanation": expl,
            }
        )
    final_sorted = sorted(final, key=lambda x: x["overall"], reverse=True)
    return final_sorted[:top_n]

# -------------------------
# Assemble entities from uploaded blobs
# -------------------------
def assemble_entities_from_blobs(jd_blob: Dict[str, Any], resume_blobs: List[Dict[str, Any]]):
    jd_text = read_document(jd_blob)
    jd_struct = extract_job_struct(jd_text)
    job = {
        "id": jd_blob.get("name", f"job_{uuid.uuid4()}"),
        "raw": jd_text,
        "structured": jd_struct,
    }
    resumes = []
    for rb in resume_blobs:
        txt = read_document(rb)
        parsed = extract_resume_struct(txt)
        resumes.append(
            {
                "id": rb.get("name", f"res_{uuid.uuid4()}"),
                "raw": txt,
                "skills": parsed.get("skills", []),
                "experiences": parsed.get("experience", []),
                "structured": parsed,
                "name": parsed.get("name", rb.get("name", "")),
                "domain": parsed.get("domain", []),
                "years_est": parsed.get("years_est", 0),
                "last_active": parsed.get("last_active", ""),
            }
        )
    return job, resumes

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Resume Ranking System", layout="wide")
st.title("🎯 Multi-Dimensional AI Resume Ranking System")
st.write("Upload a job description and resumes to generate ranked candidates")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📁 Input Files")
    jd_file = st.file_uploader("📄 Upload Job Description", type=["pdf", "txt"])
    res_files = st.file_uploader("🧑‍💼 Upload Candidate Resumes", type=["pdf", "txt"], accept_multiple_files=True)

    with st.expander("⚙️ Advanced Settings"):
        top_n = st.slider("Top N candidates to show:", 1, 20, 10)
        st.write("LLM model:", cfg.llm_model)
        st.write("Embedding model:", cfg.embed_model)
        st.markdown("**Weighting (0.0 - 1.0)**")
        cfg.weights.dense = st.slider("Dense similarity weight", 0.0, 1.0, cfg.weights.dense, 0.05)
        cfg.weights.keyword = st.slider("BM25 keyword weight", 0.0, 1.0, cfg.weights.keyword, 0.05)
        cfg.weights.skill = st.slider("Skill tier weight", 0.0, 1.0, cfg.weights.skill, 0.05)
        cfg.weights.experience = st.slider("Experience weight", 0.0, 1.0, cfg.weights.experience, 0.05)
        cfg.weights.domain = st.slider("Domain weight", 0.0, 1.0, cfg.weights.domain, 0.05)
        cfg.weights.recency = st.slider("Recency weight", 0.0, 1.0, cfg.weights.recency, 0.05)
        cfg.weights.projects = st.slider("Projects weight", 0.0, 1.0, cfg.weights.projects, 0.05)
        cfg.weights.education = st.slider("Education weight", 0.0, 1.0, cfg.weights.education, 0.05)
        cfg.weights.metadata = st.slider("Metadata weight", 0.0, 1.0, cfg.weights.metadata, 0.05)

    run_button = st.button("🚀 Run Candidate Ranking")

with col2:
    if run_button:
        if not jd_file:
            st.error("Please upload a job description.")
            st.stop()
        if not res_files:
            st.error("Please upload at least one resume.")
            st.stop()

        st.info("⏳ Running full hybrid pipeline (LLM extraction + embeddings + ranking)...")

        jd_blob = {"name": jd_file.name, "bytes": jd_file.read()}
        resume_blobs = [{"name": f.name, "bytes": f.read()} for f in res_files]

        try:
            job, resumes = assemble_entities_from_blobs(jd_blob, resume_blobs)
            ranked = rank_hybrid(job, resumes, cfg, top_n=top_n)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            logger.exception("Pipeline error")
            st.stop()

        st.success("✨ Ranking Completed!")

        st.subheader("📌 Extracted Job Structure")
        st.json(job.get("structured", {}))

        st.markdown("---")

        st.subheader("🏆 Ranked Candidates")
        df = pd.DataFrame([{"Rank": i + 1, "Name": r["name"], "Score": round(r["overall"], 3)} for i, r in enumerate(ranked)])
        st.dataframe(df, use_container_width=True)

        st.markdown("---")

        st.subheader("🔍 Candidate Explanations")
        for i, r in enumerate(ranked):
            with st.expander(f"#{i+1} — {r['name']} (Score: {round(r['overall'], 3)})"):
                comp_items = list(r["components"].items())
                comp_df = pd.DataFrame({"Component": [k for k, _ in comp_items], "Score": [float(v) for _, v in comp_items]})
                radar_fig = px.line_polar(comp_df, r="Score", theta="Component", line_close=True, range_r=[0, 1], title="Component Radar")
                radar_fig.update_traces(fill="toself")
                st.plotly_chart(radar_fig, use_container_width=True)

                overall_fit = r["explanation"]["structured"].get("overall", r["overall"])
                conf = r["explanation"]["structured"].get("confidence", "unknown")
                st.write(f"**Overall fit score:** {overall_fit:.3f}")
                st.write(f"**Confidence:** {conf}")

                st.markdown("### 📝 Candidate Review")
                st.info(r["explanation"].get("human", ""))

        st.markdown("---")

        st.subheader("📥 Download Results")
        st.download_button("Download JSON Report", json.dumps(ranked, indent=2, default=str), file_name="ranking_output.json", mime="application/json")
        st.download_button("Download CSV Rankings", df.to_csv(index=False), file_name="ranking_output.csv", mime="text/csv")

        st.caption("Built with Streamlit • FAISS • BGE Embeddings • LLM Explainability")

# End of file
