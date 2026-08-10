"""
BiTremplin - API de traduction (version relais vers Kaggle+ngrok)
====================================================================
L'Inference API Serverless de Hugging Face ne supporte pas les modeles
fine-tunes personnels non adoptes par un fournisseur (erreur StopIteration
constatee en test). Cette version relaie donc les requetes de traduction
vers une session Kaggle qui fait tourner le vrai modele (voir
run_api_kaggle_ngrok.py), exposee via un tunnel ngrok.

Render reste le point d'entree stable pour le frontend (URL fixe), mais le
calcul reel depend d'une session Kaggle active -> renseigne KAGGLE_API_URL
(variable d'environnement sur Render) a chaque redemarrage de la session
Kaggle, avec la nouvelle URL ngrok affichee par le script.
"""

import os
import json
from pathlib import Path
from typing import Optional, List

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "Expendadeur/nllb-kin-fr")

# URL ngrok de la session Kaggle active (voir run_api_kaggle_ngrok.py).
# A METTRE A JOUR sur Render (Settings > Environment) a chaque redemarrage
# de la session Kaggle, car l'URL ngrok change a chaque fois (sauf domaine
# fixe reserve). Sans ca, /translate repond une erreur claire plutot que de
# planter silencieusement.
KAGGLE_API_URL = os.environ.get("KAGGLE_API_URL", "")

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "change-me")

BASE_DIR = Path(__file__).parent
LANG_FILE = BASE_DIR / "languages.json"
ADMIN_FILE = BASE_DIR / "admins.json"

REQUEST_TIMEOUT_S = 60  # le modele + generation peut prendre du temps sur CPU Kaggle

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="BiTremplin Translation API (relais Kaggle)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not KAGGLE_API_URL:
    print("[BiTremplin] ATTENTION : KAGGLE_API_URL non definie. /translate renverra une erreur claire tant qu'elle ne l'est pas.")
else:
    print(f"[BiTremplin] Relais configure vers : {KAGGLE_API_URL}")


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
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "mode": "relais-kaggle",
        "kaggle_configured": bool(KAGGLE_API_URL),
    }


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

    if not KAGGLE_API_URL:
        raise HTTPException(
            status_code=503,
            detail="KAGGLE_API_URL non configuree sur Render. Lance run_api_kaggle_ngrok.py "
                   "dans une session Kaggle, puis colle l'URL ngrok affichee dans les "
                   "variables d'environnement Render (Settings > Environment).",
        )

    # Relais simple : le script Kaggle fait deja tout le travail (verification
    # des langues autorisees, conversion code public -> modelCode, appel au
    # modele reel). Render se contente de transmettre et de renvoyer la reponse.
    try:
        resp = httpx.post(
            f"{KAGGLE_API_URL.rstrip('/')}/translate",
            json={"text": req.text, "source": req.source, "target": req.target},
            timeout=REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        return TranslateResponse(
            translation=data["translation"], source=req.source, target=req.target
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Impossible de joindre la session Kaggle. Verifie qu'elle est bien "
                   "active et que KAGGLE_API_URL est a jour (l'URL ngrok change a "
                   "chaque redemarrage de la session).",
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="La session Kaggle n'a pas repondu a temps.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur renvoyee par la session Kaggle ({e.response.status_code}) : {e.response.text}",
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
