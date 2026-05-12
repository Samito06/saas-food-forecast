"""
app/i18n.py — translation helpers. All strings must go through T() or Tlist().
"""
import streamlit as st
from constants import TRANS, EN_DAYS_IDX


def T(key: str) -> str:
    lang = st.session_state.get("lang", "fr")
    val  = TRANS.get(lang, TRANS["fr"]).get(key) or TRANS["fr"].get(key, key)
    return val if isinstance(val, str) else str(val)


def Tlist(key: str) -> list:
    lang = st.session_state.get("lang", "fr")
    val  = TRANS.get(lang, TRANS["fr"]).get(key) or TRANS["fr"].get(key, [])
    return val if isinstance(val, list) else []


def Tdays() -> list[str]:
    lang = st.session_state.get("lang", "fr")
    return TRANS.get(lang, TRANS["fr"]).get("days", TRANS["fr"]["days"])


def day_label(en_name: str) -> str:
    days = Tdays()
    return days[EN_DAYS_IDX.get(en_name, 0)]
