"""
AyurKOSH-GNN - Simple Admin Access Control (Passkey Gate)
==========================================================
Protects the audit records and summary metrics with a secure Admin Passkey.
Students and experts can only review formulations and submit assessments.
Only the project lead with the passkey can view and download audit records.
"""

import os
import streamlit as st

DEFAULT_ADMIN_PASSKEY = "ayurkosh2026"

def get_admin_passkey():
    """Fetches configured passkey from st.secrets or environment variable."""
    try:
        if "admin_passkey" in st.secrets:
            return str(st.secrets["admin_passkey"])
    except Exception:
        pass
    return os.environ.get("ADMIN_PASSKEY", DEFAULT_ADMIN_PASSKEY)

def is_admin():
    """Returns True if the current session is unlocked as Admin."""
    return bool(st.session_state.get("is_admin", False))

def logout_admin():
    """Logs out the admin session."""
    st.session_state["is_admin"] = False
    st.rerun()

def render_admin_sidebar_controls():
    """Renders Admin login form or active session status in the sidebar."""
    st.sidebar.markdown("---")
    
    if is_admin():
        st.sidebar.success("🔓 **Admin Mode Active**\n\nAudit records & downloads unlocked.", icon="✅")
        if st.sidebar.button("🚪 Lock Admin Access", use_container_width=True):
            logout_admin()
    else:
        with st.sidebar.expander("🔒 Admin Portal", expanded=False):
            st.caption("Audit reports and records are restricted to the project lead.")
            passkey_input = st.text_input(
                "Enter Admin Passkey:",
                type="password",
                key="admin_passkey_box",
                help="Enter passkey to view and download audit registries."
            )
            if st.button("Unlock Admin Access", type="primary", use_container_width=True):
                expected = get_admin_passkey()
                if passkey_input and passkey_input == expected:
                    st.session_state["is_admin"] = True
                    st.success("Admin access granted!")
                    st.rerun()
                else:
                    st.error("Incorrect Admin Passkey.")
