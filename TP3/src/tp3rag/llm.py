"""LLM local et contrôle observable de ses capacités (aucune API payante)."""
import base64
import io
import json
import time
import urllib.request

from PIL import Image, ImageDraw
from langchain_ollama import ChatOllama

from .config import Settings


def modele_local(settings=None):
    settings = settings or Settings()
    return ChatOllama(model=settings.model, base_url=settings.ollama_url, temperature=0,
                      seed=42, num_ctx=6144, num_predict=500, repeat_penalty=1.08, keep_alive="10m",
                      client_kwargs={"timeout": 240}, async_client_kwargs={"timeout": 240})


def api_ollama(settings, route, data):
    req = urllib.request.Request(settings.ollama_url + route,
        data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as reponse:
        return json.load(reponse)


def verifier_capacites(settings=None):
    settings = settings or Settings()
    debut = time.perf_counter()
    info = api_ollama(settings, "/api/show", {"model": settings.model})
    image = Image.new("RGB", (360, 240), "white")
    dessin = ImageDraw.Draw(image)
    dessin.rectangle((20, 30, 170, 210), fill="red")
    dessin.rectangle((190, 30, 340, 210), fill="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image64 = base64.b64encode(buf.getvalue()).decode()
    options = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 140}
    vision = api_ollama(settings, "/api/chat", {"model": settings.model, "stream": False,
        "messages": [{"role": "user", "content": "Nomme en français les couleurs des deux rectangles, de gauche à droite. Réponse très courte.",
                      "images": [image64]}], "options": options, "keep_alive": "10m"})
    outil = {"type": "function", "function": {"name": "additionner",
        "description": "Additionner deux entiers. Utiliser ce calculateur pour la somme demandée.",
        "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                       "required": ["a", "b"]}}}
    decision = api_ollama(settings, "/api/chat", {"model": settings.model, "stream": False,
        "messages": [{"role": "user", "content": "Appelle obligatoirement additionner avec a=7 et b=5. Ne calcule pas de tête."}],
        "tools": [outil], "options": options, "keep_alive": "10m"})
    calls = decision["message"].get("tool_calls", [])
    appels = []
    for call in calls:
        fonction = call["function"]
        args = fonction["arguments"]
        if fonction["name"] != "additionner" or set(args) != {"a", "b"} or any(type(v) is not int for v in args.values()):
            raise ValueError("Appel de contrôle inattendu.")
        appels.append({"nom": fonction["name"], "arguments": args, "resultat": args["a"] + args["b"]})
    reponse_vision = vision["message"]["content"]
    return {"modele": settings.model, "capacites_declarees": info.get("capabilities", []),
            "vision": {"reponse": reponse_vision, "succes": "rouge" in reponse_vision.lower() and "bleu" in reponse_vision.lower(),
                       "images_transmises": 1},
            "tool_calling": {"appels_reels": appels, "succes": any(c["resultat"] == 12 for c in appels)},
            "duree_secondes": time.perf_counter() - debut}
