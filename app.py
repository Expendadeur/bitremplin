"""
BiTremplin - API de traduction (version legere, via HF Inference API Serverless)
=================================================================================
Contrairement a la version precedente qui chargeait le modele en RAM locale
(necessitant plusieurs Go), cette version appelle l'API Inference Serverless
de Hugging Face : c'est HF qui fait tourner le modele sur SES machines, ce
backend ne fait que relayer la requete. Resultat : aucune dependance torch/
transformers lourde, RAM necessaire minime -> tient dans le tier gratuit
Render (512 Mo).

Limite honnete a connaitre : un modele fine-tune personnel, peu utilise,
n'est pas toujours "chaud" sur l'infrastructure partagee de HF. Un premier
appel peut recevoir une erreur 503 (modele en cours de chargement cote HF) -
ce code reessaie automatiquement plusieurs fois avant d'abandonner. Si ca
reste instable en pratique, la solution Kaggle+ngrok (voir nos echanges
precedents) reste le filet de secours.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Expendadeur/nllb-kin-fr")

# Necessaire meme pour un modele public : identifie le compte aupres de HF et
# evite les limites de debit anonymes, plus severes. Cree un token (role
# "read" suffit) sur https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me")

BASE_DIR = Path(__file__).parent
LANG_FILE = BASE_DIR / "languages.json"
ADMIN_FILE = BASE_DIR / "admins.json"

MAX_RETRIES = 4
RETRY_DELAY_S = 5  # entre chaque tentative si le modele est "en train de charger" cote HF

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="BiTremplin Translation API (Inference API)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not HF_TOKEN:
    print("[BiTremplin] ATTENTION : HF_TOKEN non defini, risque de limites de debit basses.")

client = InferenceClient(model=MODEL_NAME, token=HF_TOKEN or None)
print(f"[BiTremplin] Client Inference API pret pour : {MODEL_NAME}")


# ---------------------------------------------------------------------------
# "Base de donnees" fichier JSON (langues + admins)
# ---------------------------------------------------------------------------

def _read_json(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default

def _write_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_languages() -> List[dict]:
    return _read_json(LANG_FILE, [])

def save_languages(langs: List[dict]):
    _write_json(LANG_FILE, langs)

def load_admins() -> dict:
    return _read_json(ADMIN_FILE, {"emails": []})

def save_admins(data: dict):
    _write_json(ADMIN_FILE, data)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TranslateRequest(BaseModel):
    text: str
    source: str
    target: str

class TranslateResponse(BaseModel):
    translation: str
    source: str
    target: str

class LanguageItem(BaseModel):
    code: str
    label: str
    flag: Optional[str] = None
    ttsLocale: Optional[str] = None
    modelCode: Optional[str] = None

class AdminEmailItem(BaseModel):
    email: str


def check_admin(x_admin_token: str = Header(default="")):
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Acces admin refuse")
    return True


# ---------------------------------------------------------------------------
# Routes publiques
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "mode": "hf-inference-api"}


@app.get("/languages", response_model=List[LanguageItem])
def get_languages():
    return load_languages()


@app.get("/is-admin")
def is_admin(email: str):
    data = load_admins()
    return {"is_admin": email.lower() in [e.lower() for e in data["emails"]]}


@app.post("/translate", response_model=TranslateResponse)
def translate(req: TranslateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texte vide")
    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="Texte trop long (max 5000 caracteres)")

    langs_by_code = {lang["code"]: lang for lang in load_languages()}
    if req.source not in langs_by_code:
        raise HTTPException(status_code=403, detail=f"Langue source non autorisee : {req.source}")
    if req.target not in langs_by_code:
        raise HTTPException(status_code=403, detail=f"Langue cible non autorisee : {req.target}")

    model_src = langs_by_code[req.source].get("modelCode") or req.source
    model_tgt = langs_by_code[req.target].get("modelCode") or req.target

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = client.translation(req.text, src_lang=model_src, tgt_lang=model_tgt)
            translation = result if isinstance(result, str) else result.translation_text
            return TranslateResponse(translation=translation, source=req.source, target=req.target)
        except HfHubHTTPError as e:
            last_error = e
            # 503 = modele en cours de "reveil" cote HF (cold start) -> on reessaie
            if "503" in str(e) and attempt < MAX_RETRIES:
                print(f"[BiTremplin] Modele en cours de chargement cote HF, tentative {attempt}/{MAX_RETRIES}...")
                time.sleep(RETRY_DELAY_S)
                continue
            break

    raise HTTPException(
        status_code=503,
        detail=f"Le modele n'a pas repondu apres {MAX_RETRIES} tentatives "
               f"(infrastructure partagee HF, reessaie dans une minute). Detail : {last_error}",
    )


# ---------------------------------------------------------------------------
# Routes admin
# ---------------------------------------------------------------------------

@app.post("/admin/languages", dependencies=[Depends(check_admin)])
def add_language(item: LanguageItem):
    langs = load_languages()
    if any(l["code"] == item.code for l in langs):
        raise HTTPException(status_code=400, detail="Cette langue est deja dans la liste")
    langs.append(item.dict())
    save_languages(langs)
    return {"status": "ajoutee", "languages": langs}


@app.delete("/admin/languages/{code}", dependencies=[Depends(check_admin)])
def remove_language(code: str):
    langs = [l for l in load_languages() if l["code"] != code]
    save_languages(langs)
    return {"status": "supprimee", "languages": langs}


@app.get("/admin/admins", dependencies=[Depends(check_admin)])
def get_admins():
    return load_admins()


@app.post("/admin/admins", dependencies=[Depends(check_admin)])
def add_admin(item: AdminEmailItem):
    data = load_admins()
    if item.email not in data["emails"]:
        data["emails"].append(item.email)
        save_admins(data)
    return data


@app.delete("/admin/admins/{email}", dependencies=[Depends(check_admin)])
def remove_admin(email: str):
    data = load_admins()
    data["emails"] = [e for e in data["emails"] if e != email]
    save_admins(data)
    return data
