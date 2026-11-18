"""SQLite persistence: enrolled residents + Stage 5 audit log (neurogate_aas.db)."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import List, Optional

import pandas as pd

from neurogate.core.config import DB_PATH
from neurogate.core.utils import now_iso


class DB:
    def __init__(self, path=DB_PATH):
        self.path = path
        self.lock = threading.Lock()
        with self._c() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS residents(
                stem TEXT PRIMARY KEY, name TEXT, unit TEXT, img_path TEXT, wav_path TEXT,
                face_emb BLOB, face_norm REAL, voice_emb BLOB, synthetic INTEGER, enrolled_at TEXT)""")
            c.execute("""CREATE TABLE IF NOT EXISTS audit(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, attempt INTEGER,
                face_ids TEXT, cardinality INTEGER, voice_top TEXT, live INTEGER, decision TEXT,
                door TEXT, crossings INTEGER, rationale TEXT, trace TEXT)""")

    def _c(self):
        return sqlite3.connect(self.path, timeout=10)

    def unit_of(self, stem: str) -> Optional[str]:
        with self._c() as c:
            r = c.execute("SELECT unit FROM residents WHERE stem=?", (stem,)).fetchone()
        return r[0] if r else None

    def upsert_resident(self, r):
        with self.lock, self._c() as c:
            old = c.execute("SELECT enrolled_at FROM residents WHERE stem=?", (r.stem,)).fetchone()
            c.execute("INSERT OR REPLACE INTO residents VALUES(?,?,?,?,?,?,?,?,?,?)", (
                r.stem, r.name, r.unit, r.img_path, r.wav_path or "",
                r.face_e.tobytes() if r.face_e is not None else None, float(r.face_q),
                r.voice_e.tobytes() if r.voice_e is not None else None, int(r.synthetic),
                old[0] if old else now_iso()))

    def prune(self, stems: List[str]):
        with self.lock, self._c() as c:
            if stems:
                c.execute(f"DELETE FROM residents WHERE stem NOT IN ({','.join('?' * len(stems))})", stems)
            else:
                c.execute("DELETE FROM residents")

    def directory(self) -> pd.DataFrame:
        with self._c() as c:
            rows = c.execute("SELECT name, unit, ROUND(face_norm,1), CASE WHEN voice_emb IS NULL THEN 'missing' ELSE 'enrolled' END,"
                             " CASE synthetic WHEN 1 THEN 'synthetic demo' ELSE 'photo' END, enrolled_at FROM residents ORDER BY name").fetchall()
        return pd.DataFrame(rows, columns=["Resident", "Unit", "‖z‖₂", "Voiceprint", "Source", "Enrolled"])

    def log(self, **kw):
        with self.lock, self._c() as c:
            c.execute("INSERT INTO audit(ts,source,attempt,face_ids,cardinality,voice_top,live,decision,door,crossings,rationale,trace)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (now_iso(), kw["source"], kw["attempt"], kw["face_ids"], kw["C"], kw["voice_top"], int(kw["live"]),
                       kw["decision"], kw["door"], kw["crossings"], kw["rationale"], json.dumps(kw["trace"], default=str)[:20000]))

    def audit(self, limit=200) -> pd.DataFrame:
        with self._c() as c:
            rows = c.execute("SELECT id, ts, source, attempt, face_ids, cardinality, voice_top, live, decision, door, crossings, rationale"
                             " FROM audit ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return pd.DataFrame(rows, columns=["#", "Time", "Source", "Try", "Face match set F", "C", "Voice top j*",
                                           "Live", "Decision", "Door", "Crossings", "Rationale"])

    def clear_audit(self):
        with self.lock, self._c() as c:
            c.execute("DELETE FROM audit")

    def last_visit(self, name: str) -> Optional[str]:
        """Timestamp of the resident's most recent granted entry (for human-like greetings)."""
        with self._c() as c:
            r = c.execute("SELECT ts FROM audit WHERE decision IN ('allow','tailgate-alert') AND face_ids LIKE ? "
                          "ORDER BY id DESC LIMIT 1", (f"%{name}%",)).fetchone()
        return r[0] if r else None
