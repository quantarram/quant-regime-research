"""Minimal .xlsx reader (zip + XML): returns {sheet name: list of row lists}. Avoids adding a dependency for EIA's wholesale price workbooks."""
import re
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    rid = {r.get("Id"): r.get("Target") for r in rels}
    out = {}
    for sh in wb.find("m:sheets", NS):
        target = rid[sh.get("{%s}id" % NS["r"])]
        path_ = "xl/" + target.lstrip("/").replace("xl/", "")
        root = ET.fromstring(z.read(path_))
        rows = []
        for row in root.iter("{%s}row" % NS["m"]):
            vals = {}
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is None:
                    inl = c.find("m:is", NS)
                    vals[col_index(c.get("r"))] = "".join(t.text or "" for t in inl.iter("{%s}t" % NS["m"])) if inl is not None else None
                    continue
                t = c.get("t")
                vals[col_index(c.get("r"))] = shared[int(v.text)] if t == "s" else (v.text if t in ("str", "b", "e") else float(v.text))
            if vals:
                rows.append([vals.get(i) for i in range(max(vals) + 1)])
        out[sh.get("name")] = rows
    return out
