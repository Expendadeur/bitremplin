---
title: BiTremplin Translation API
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# BiTremplin - API de traduction

API FastAPI qui sert le modele NLLB fine-tune (Kirundi <-> Francais) et supporte
en principe les ~200 langues du modele NLLB de base. Voir `/docs` pour la
documentation interactive (Swagger) une fois deploye.

Routes principales :
- `POST /translate` : traduire un texte
- `GET /languages` : liste des langues affichees dans le selecteur
- `POST /admin/languages` : ajouter une langue (protege par header `X-Admin-Token`)
