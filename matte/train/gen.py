"""Render matting source images on a Qwen lane: subjects on chroma screens, and backdrop scenes.

    python3 matte/train/gen.py subjects --n 300 --url http://127.0.0.1:8792
    python3 matte/train/gen.py scenes --n 60 --url http://127.0.0.1:18080 --secret-file ...

Resumable: an id that already has an image is skipped. Prompts keep the key colour out of the
subject so the despill teacher cannot eat real colours.
"""
import argparse, base64, hashlib, json, os, random, sys, time, urllib.request

OUT = os.path.expanduser(os.environ.get("MATTE_DATA", "~/matte-data"))

SUBJECTS = [
    "a woman with very long curly frizzy {hair} hair, loose flyaway strands",
    "a man with a big messy {hair} afro and a short beard",
    "a girl with two long {hair} braids and wispy baby hairs",
    "an old man with thin wispy white hair and a white beard",
    "a woman with a {hair} messy bun, many loose strands around her face",
    "a man with shoulder-length wavy {hair} hair blowing in the wind",
    "a woman with short pixie-cut {hair} hair and fluffy edges",
    "a fluffy {fur} persian cat sitting",
    "a long-haired {fur} dog with flowing ears",
    "a fluffy {fur} rabbit",
    "a {fur} fox with a bushy tail",
    "a {fur} owl with soft feathers spread",
    "a dandelion seed head on a stem",
    "a person wearing a {cloth} lace shawl with fringe tassels",
    "a woman in a sheer {cloth} chiffon dress with a flowing veil",
    "a bouquet of {flower} flowers with thin stems",
    "a {cloth} feather boa draped over a chair",
    "a dancer with long {hair} hair mid-spin, hair flying",
    "a child with curly {hair} hair blowing bubbles",
    "a {fur} long-haired guinea pig",
]
HAIR = ["dark brown", "black", "blonde", "platinum blonde", "auburn", "red", "light brown", "grey", "copper", "chestnut"]
FUR = ["white", "orange", "grey", "black and white", "golden", "cream", "brown tabby", "silver"]
CLOTH = ["white", "black", "red", "cream", "purple", "pink", "beige", "orange", "navy blue", "gold"]
FLOWER = ["red", "white", "pink", "yellow", "purple", "orange"]
WEAR = ["white", "black", "red", "cream", "grey", "pink", "beige", "burgundy", "orange", "brown"]
SCREENS = {
    "green": "a bright saturated chroma key green screen backdrop",
    "blue": "a bright saturated chroma key blue screen backdrop",
}
LIGHT = [
    "soft even studio lighting",
    "strong backlight rim light, {key} light spill on the edges",
    "{key} light bouncing from the screen onto the subject's edges",
    "hard side light",
]
FRAMING = ["full body shot", "waist-up portrait", "close-up portrait", "three-quarter shot"]

SCENES = [
    "a cozy living room interior with a sofa and lamps", "a busy city street at night with neon signs",
    "a sunny beach with palm trees", "a dense green forest path", "a modern office with large windows",
    "a snowy mountain landscape", "a kitchen with white tiles", "a desert at sunset", "a library with tall bookshelves",
    "a rainy street with reflections", "a field of yellow flowers", "a cafe interior with warm lights",
    "a concrete skate park", "an underwater coral reef", "a cyberpunk alley", "a wheat field under blue sky",
    "a grey studio wall with soft gradient", "a bedroom with fairy lights", "a train station platform",
    "a lush jungle with ferns", "a brick wall with graffiti", "a ballroom with chandeliers", "a meadow with tall grass",
    "a stormy sky over the sea", "a japanese garden with maple trees", "a white marble hallway",
]


def subject_prompt(rng):
    tpl = rng.choice(SUBJECTS)
    key = rng.choice(list(SCREENS))
    s = tpl.format(hair=rng.choice(HAIR), fur=rng.choice(FUR), cloth=rng.choice(CLOTH), flower=rng.choice(FLOWER))
    wear = rng.choice(WEAR)
    light = rng.choice(LIGHT).format(key=key)
    return key, (f"studio photo of {s}, wearing {wear} clothes, {rng.choice(FRAMING)}, in front of {SCREENS[key]}, "
                 f"{light}, the whole subject is clearly separated from the backdrop, sharp focus, high detail, 85mm")


def post(url, body, secret, timeout=900):
    q = f"?secret={secret}" if secret else ""
    req = urllib.request.Request(url + "/v1/images/generations" + q, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["subjects", "scenes"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--url", default="http://127.0.0.1:8792")
    ap.add_argument("--secret-file", default="")
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--turbo", default="")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    secret = open(a.secret_file).read().strip() if a.secret_file else ""
    out = os.path.join(OUT, a.kind)
    os.makedirs(out, exist_ok=True)
    for i in range(a.start, a.start + a.n):
        sid = f"{a.kind[:2]}{i:05d}"
        path = os.path.join(out, sid + ".webp")
        if os.path.exists(path):
            continue
        rng = random.Random(i * 7919 + (1 if a.kind == "scenes" else 0))
        if a.kind == "subjects":
            key, prompt = subject_prompt(rng)
            size = rng.choice(["1024x1024", "896x1152", "1152x896"]) if a.size == "mixed" else a.size
        else:
            key, prompt = None, f"{rng.choice(SCENES)}, photo, natural light, high detail, no people"
            size = a.size
        body = {"prompt": prompt, "size": size, "seed": rng.randrange(1 << 30), "cache": False}
        if a.turbo:
            body["turbo"] = a.turbo == "1"
        t = time.perf_counter()
        try:
            d = post(a.url, body, secret)
            raw = base64.b64decode(d["data"][0]["b64_json"])
        except Exception as e:
            print(f"{sid} failed: {e}", flush=True)
            time.sleep(5)
            continue
        open(path + ".tmp", "wb").write(raw)
        os.replace(path + ".tmp", path)
        json.dump({"id": sid, "key": key, "prompt": prompt, "size": size, "seed": body["seed"]}, open(os.path.join(out, sid + ".json"), "w"))
        print(f"{sid} {size} {time.perf_counter() - t:.1f}s {prompt[:70]}", flush=True)


main()
