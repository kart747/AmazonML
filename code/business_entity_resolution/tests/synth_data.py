"""Tiny synthetic challenge dataset with the real file layout and noise patterns.

Used by the end-to-end smoke test so anyone can verify the whole pipeline in about a
minute without the 2.4 GB dataset. Mimics: S1 reference records; 0–4 noisy copies each
in S2 and S3 (typos, case, suffix drop/move, reorder, leetspeak, abbreviations, missing
address parts, empty addresses); ~6% singletons; ~25% distractor targets, including
"sibling" lookalikes (same name, nearby house number); an unseen country in test.
"""
import random
from pathlib import Path

WORDS = ("apex blue cedar delta eagle falcon global harbor iris jade krishna lotus metro nova orion "
         "prime quest raven sai shree summit tiger urban vista zenith coastal franklin ashutosh").split()
KINDS = "traders consulting motors foods logistics labs pharma textiles bakery realty".split()
LEGAL = {"US": ["Inc", "LLC", "Corp", "Co"], "India": ["Private Limited", "Pvt Ltd", "LLP"],
         "France": ["SARL", "SAS", "EURL"]}
STREETS = {"US": ["Main Street", "Oak Avenue", "Cedar Drive", "Park Road"],
           "India": ["MG Road", "Station Road", "Nehru Nagar", "Gandhi Marg"],
           "France": ["Rue de Paris", "Avenue Victor Hugo", "Boulevard Carnot", "Rue Nationale"]}
CITIES = {"US": [("Austin", "Texas", "TX"), ("Everson", "Washington", "WA"), ("Louisville", "Kentucky", "KY")],
          "India": [("Pune", "Maharashtra", "MH"), ("Mohali", "Punjab", "PB"), ("Delhi", "Delhi", "DL")],
          "France": [("Bordeaux", "Nouvelle-Aquitaine", "33"), ("Nantes", "Pays de la Loire", "44")]}


def _typo(r, s):
    if len(s) < 4:
        return s
    i = r.randrange(1, len(s) - 1)
    return s[:i] + s[i + 1] + s[i] + s[i + 2:] if r.random() < 0.5 else s[:i] + s[i + 1:]


def _entity(r, country):
    name = " ".join(w.title() for w in r.sample(WORDS, 2)) + " " + r.choice(KINDS).title() + " " + r.choice(LEGAL[country])
    city, state, code = r.choice(CITIES[country])
    return name, (r.randrange(1, 9999), r.choice(STREETS[country]), city, state, code)


def _addr(r, a, noisy):
    num, street, city, state, code = a
    if not noisy:
        return f"{num} {street}, {city}, {state}"
    if r.random() < 0.04:
        return ""
    parts = [f"{'#' if r.random() < 0.2 else ''}{num if r.random() < 0.9 else num + 1} {street.upper() if r.random() < 0.4 else street}",
             city, code if r.random() < 0.5 else state]
    r.shuffle(parts) if r.random() < 0.3 else None
    return ", ".join(p for p in parts if r.random() > 0.1)


def _name(r, n):
    toks = n.split()
    x = r.random()
    if x < 0.2:
        toks = toks[:-1]
    elif x < 0.3:
        toks = [toks[-1]] + toks[:-1]
    s = " ".join(toks)
    if r.random() < 0.3:
        s = _typo(r, s)
    if r.random() < 0.1:
        s = s.replace("o", "0", 1)
    return s.upper() if r.random() < 0.2 else s


def make(root: Path, n_train=3000, n_test=1500, seed=0):
    r = random.Random(seed)
    used = set()

    def uid(prefix):
        while True:
            i = r.randrange(1, 10**9)
            if (prefix, i) not in used:
                used.add((prefix, i))
                return f"{prefix}-{i}"

    for split, n, countries in (("train", n_train, ["US", "India"]), ("test", n_test, ["US", "India", "France"])):
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        s1, s2, s3, gt = [], [], [], []
        for _ in range(n):
            c = r.choice(countries)
            name, a = _entity(r, c)
            sid = uid("S1")
            s1.append((sid, name, _addr(r, a, False), c))
            matches = []
            if r.random() > 0.06:
                for src, rows, pre in ((2, s2, "S2"), (3, s3, "S3")):
                    for _ in range(r.randint(0, 3) + (1 if src == 2 else 0)):
                        tid = uid(pre)
                        rows.append((tid, _name(r, name), _addr(r, a, True), c))
                        matches.append(tid)
            if r.random() < 0.3:  # sibling lookalike distractor: same name, nearby number
                num, *rest = a
                s2.append((uid("S2"), _name(r, name), _addr(r, (num + r.randint(2, 9), *rest), True), c))
            gt.append((sid, ",".join(matches)))
        for _ in range(int(0.3 * n)):  # unrelated distractors
            c = r.choice(countries)
            name, a = _entity(r, c)
            pre, rows = r.choice([("S2", s2), ("S3", s3)])
            rows.append((uid(pre), _name(r, name), _addr(r, a, True), c))
        hdr = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        for k, rows in ((1, s1), (2, s2), (3, s3)):
            r.shuffle(rows)
            (d / f"{split}_source{k}.tsv").write_text(hdr + "".join("\t".join(x) + "\n" for x in rows))
        if split == "train":
            (d / "train_ground_truth.tsv").write_text(
                "source1_entity_id\tmatched_entity_ids\n" + "".join(f"{a}\t{b}\n" for a, b in gt))
    return root
