"""
Streamlit application for Decision-Making Under Uncertainty toolkit.
"""

import streamlit as st
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def main():
    st.set_page_config(
        page_title="M6 - Streamlit Application",
        page_icon="🤖",
    )

    st.sidebar.header("Laboratories")
    labs = [
        ("Bandit Laboratory", "Bandit Lab"),
        ("GridWorld Laboratory", "GridWorld Lab"),
        ("POMDP Laboratory", "POMDP Lab"),
        ("Transfer Learning Lab", "Transfer Learning Lab"),
        ("Results & History", "Experiment History"),
    ]
    
    st.sidebar.radio("Select a laboratory", labs)
    
    st.title("Decision-Making Under Uncertainty")
    st.markdown("""
    This application provides interactive laboratories for exploring decision-making algorithms
    under uncertainty. Each laboratory contains configurable experiments that you can run and compare.
    """)


if __name__ == "__main__":
    main()