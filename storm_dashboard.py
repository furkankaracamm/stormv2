"""
STORM Dashboard - Real-time monitoring for the research system
Run: streamlit run storm_dashboard.py
"""

import streamlit as st
import sqlite3
import pandas as pd
import time
import os
from pathlib import Path
from datetime import datetime

# Configuration
STORM_DATA_DIR = "storm_data"
BRAIN_DIR = "storm_persistent_brain"
METADATA_DB = os.path.join(BRAIN_DIR, "metadata.db")  # Use Brain's database
ACADEMIC_DB = os.path.join(STORM_DATA_DIR, "academic_brain.db")
LOG_FILE = "professor_live.log"

# Page config
st.set_page_config(
    page_title="🌪️ STORM Dashboard",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .gap-card {
        background: linear-gradient(135deg, #2d3436 0%, #636e72 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #00b894;
    }
    .log-container {
        background: #0d1117;
        color: #58a6ff;
        font-family: 'Consolas', monospace;
        padding: 15px;
        border-radius: 10px;
        height: 400px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


def get_db_connection(db_path):
    """Get SQLite connection with error handling."""
    try:
        if os.path.exists(db_path):
            return sqlite3.connect(db_path, check_same_thread=False)
    except:
        pass
    return None


def get_paper_stats():
    """Get statistics about processed papers."""
    stats = {"total": 0, "books": 0, "papers": 0, "theses": 0}
    conn = get_db_connection(METADATA_DB)
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM processed_files")
            stats["total"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE filename LIKE 'BOOK_%'")
            stats["books"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE filename LIKE 'PAPER_%'")
            stats["papers"] = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM processed_files WHERE filename LIKE 'THESIS_%'")
            stats["theses"] = cursor.fetchone()[0]
            conn.close()
        except:
            pass
    return stats


def get_claim_stats():
    """Get statistics about extracted claims."""
    stats = {"total_claims": 0, "hypothesis": 0, "finding": 0}
    conn = get_db_connection(ACADEMIC_DB)
    if conn:
        try:
            cursor = conn.cursor()
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='paper_claims'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM paper_claims")
                stats["total_claims"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM paper_claims WHERE claim_type='hypothesis'")
                stats["hypothesis"] = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM paper_claims WHERE claim_type='finding'")
                stats["finding"] = cursor.fetchone()[0]
            conn.close()
        except:
            pass
    return stats


def get_gaps():
    """Get detected research gaps."""
    gaps = []
    conn = get_db_connection(ACADEMIC_DB)
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detected_gaps'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT gap_type, suggestion, gap_score, detected_at 
                    FROM detected_gaps 
                    ORDER BY detected_at DESC 
                    LIMIT 20
                """)
                for row in cursor.fetchall():
                    gaps.append({
                        "type": row[0],
                        "suggestion": row[1],
                        "score": row[2],
                        "detected_at": row[3]
                    })
            conn.close()
        except:
            pass
    return gaps


def get_searched_queries():
    """Get cached search queries."""
    queries = []
    conn = get_db_connection(METADATA_DB)
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='searched_queries'")
            if cursor.fetchone():
                cursor.execute("SELECT query, searched_at, found FROM searched_queries ORDER BY searched_at DESC LIMIT 50")
                for row in cursor.fetchall():
                    queries.append({
                        "query": row[0],
                        "searched_at": row[1],
                        "found": "✅" if row[2] else "❌"
                    })
            conn.close()
        except:
            pass
    return queries


def get_recent_logs(lines=50):
    """Get recent log entries."""
    logs = []
    try:
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                logs = f.readlines()[-lines:]
    except:
        pass
    return logs


def main():
    # Header
    st.markdown('<p class="main-header">🌪️ STORM Research System</p>', unsafe_allow_html=True)
    st.markdown("**Real-time monitoring dashboard for autonomous academic research**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (sec)", 5, 60, 10)
        
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        stats = get_paper_stats()
        st.metric("Total Documents", stats["total"])
        
        st.markdown("---")
        st.markdown("## 🔗 Quick Links")
        st.markdown("- [arXiv](https://arxiv.org)")
        st.markdown("- [Semantic Scholar](https://semanticscholar.org)")
        st.markdown("- [Anna's Archive](https://annas-archive.org)")
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Research Gaps", "📜 Live Logs", "🔍 Search Cache"])
    
    # TAB 1: Overview
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        stats = get_paper_stats()
        claim_stats = get_claim_stats()
        
        with col1:
            st.metric(
                "📚 Total Documents",
                stats["total"],
                delta=None
            )
        with col2:
            st.metric(
                "📄 Papers",
                stats["papers"]
            )
        with col3:
            st.metric(
                "📖 Books",
                stats["books"]
            )
        with col4:
            st.metric(
                "💡 Claims Extracted",
                claim_stats["total_claims"]
            )
        
        st.markdown("---")
        
        # Document breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Document Distribution")
            if stats["total"] > 0:
                chart_data = pd.DataFrame({
                    'Type': ['Papers', 'Books', 'Theses', 'Other'],
                    'Count': [
                        stats["papers"], 
                        stats["books"], 
                        stats["theses"],
                        stats["total"] - stats["papers"] - stats["books"] - stats["theses"]
                    ]
                })
                st.bar_chart(chart_data.set_index('Type'))
            else:
                st.info("No documents processed yet")
        
        with col2:
            st.markdown("### 🧠 Claims by Type")
            if claim_stats["total_claims"] > 0:
                claim_data = pd.DataFrame({
                    'Type': ['Hypothesis', 'Finding', 'Other'],
                    'Count': [
                        claim_stats["hypothesis"],
                        claim_stats["finding"],
                        claim_stats["total_claims"] - claim_stats["hypothesis"] - claim_stats["finding"]
                    ]
                })
                st.bar_chart(claim_data.set_index('Type'))
            else:
                st.info("No claims extracted yet")
    
    # TAB 2: Research Gaps
    with tab2:
        st.markdown("### 🎯 Detected Research Gaps")
        st.markdown("*These are automatically identified research opportunities*")
        
        gaps = get_gaps()
        
        if gaps:
            for gap in gaps:
                gap_type = gap["type"].replace("_", " ").title()
                score = gap["score"] or 0
                
                # Color based on gap type
                colors = {
                    "topic_gap": "🔵",
                    "citation_gap": "🟢",
                    "contradiction_gap": "🔴",
                    "geographic_gap": "🟡",
                    "method_gap": "🟣"
                }
                emoji = colors.get(gap["type"], "⚪")
                
                with st.expander(f"{emoji} **{gap_type}** (Score: {score:.2f})"):
                    st.write(gap["suggestion"])
                    st.caption(f"Detected: {gap['detected_at']}")
        else:
            st.info("🔍 No gaps detected yet. Gaps are analyzed every 10 cycles.")
            st.markdown("""
            **Gap Types:**
            - 🔵 **Topic Gap**: Understudied topics in your collection
            - 🟢 **Citation Gap**: Similar papers not citing each other
            - 🔴 **Contradiction Gap**: Conflicting claims between papers
            - 🟡 **Geographic Gap**: Understudied regions
            - 🟣 **Method Gap**: Underused research methods
            """)
    
    # TAB 3: Live Logs
    with tab3:
        st.markdown("### 📜 Live Activity Stream")
        
        logs = get_recent_logs(100)
        
        if logs:
            log_text = "".join(logs)
            st.code(log_text, language="log")
        else:
            st.info("No log entries yet")
    
    # TAB 4: Search Cache
    with tab4:
        st.markdown("### 🔍 Search Query Cache")
        st.markdown("*Cached searches to avoid redundant queries*")
        
        queries = get_searched_queries()
        
        if queries:
            df = pd.DataFrame(queries)
            st.dataframe(df, use_container_width=True)
            st.metric("Total Cached Queries", len(queries))
        else:
            st.info("No cached queries yet")
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
