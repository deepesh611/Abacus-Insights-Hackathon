"""
Streamlit Dashboard for Multi-Agent Fraud Detection System
Main application file
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui_utils import (
    load_data, 
    create_fraud_metrics, 
    plot_fraud_trend, 
    plot_fraud_by_rule, 
    plot_amount_distribution,
    get_orchestrator
)

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make chat input sticky at bottom
st.markdown("""
    <style>
    /* Make chat input sticky at bottom of viewport */
    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        width: 100% !important;
        max-width: 800px !important;
        background: white !important;
        z-index: 999 !important;
        padding: 1rem !important;
        border-top: 1px solid #e0e0e0 !important;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1) !important;
        border-radius: 8px 8px 0 0 !important;
    }
    
    /* Add padding to main content to prevent overlap with fixed input */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🛡️ Multi-Agent Fraud Detection System")

# Load data
with st.spinner("Loading data..."):
    df = load_data()
    orchestrator = get_orchestrator()

# Sidebar
st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date Range",
    [df['claim_date'].min(), df['claim_date'].max()]
)

# Filter data based on date
if len(date_range) == 2:
    mask = (df['claim_date'].dt.date >= date_range[0]) & (df['claim_date'].dt.date <= date_range[1])
    filtered_df = df.loc[mask]
else:
    filtered_df = df

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Fraud Explorer", "🕵️ Investigation", "💬 AI Assistant"])

# ===========================
# TAB 1: DASHBOARD
# ===========================
with tab1:
    # Custom CSS for dashboard cards
    st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.header("📊 Fraud Detection Dashboard")
    st.markdown("**Real-time analytics and insights into insurance claim fraud detection**")
    
    st.markdown("---")
    
    # ========================================
    # KEY METRICS - Enhanced Cards
    # ========================================
    st.subheader("🔑 Key Metrics")
    
    metrics = create_fraud_metrics(filtered_df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>📋 Total Claims</div>
                <div style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{:,}</div>
            </div>
        """.format(metrics['total_claims']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>🚨 Fraud Cases</div>
                <div style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{:,}</div>
            </div>
        """.format(metrics['fraud_claims']), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>📊 Fraud Rate</div>
                <div style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>{:.1f}%</div>
            </div>
        """.format(metrics['fraud_rate']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 1.5rem; border-radius: 10px; color: white; text-align: center;'>
                <div style='font-size: 0.9rem; opacity: 0.9;'>💰 Potential Savings</div>
                <div style='font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;'>${:,.0f}</div>
            </div>
        """.format(metrics['fraud_amount']), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========================================
    # CHARTS SECTION
    # ========================================
    st.subheader("📈 Analytics")
    
    # Row 1: Trend and Rules
    col_left, col_right = st.columns(2)
    
    with col_left:
        with st.container():
            st.plotly_chart(plot_fraud_trend(filtered_df), use_container_width=True)
        
    with col_right:
        with st.container():
            st.plotly_chart(plot_fraud_by_rule(filtered_df), use_container_width=True)
    
    # Row 2: Distribution
    st.plotly_chart(plot_amount_distribution(filtered_df), use_container_width=True)

# ===========================
# TAB 2: FRAUD EXPLORER
# ===========================
with tab2:
    st.header("Fraud Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_fraud_only = st.checkbox("Show Fraud Cases Only", value=True)
    with col2:
        min_score = st.slider("Minimum Fraud Score", 0, 100, 50)
    
    # Get unique fraud rules for filter
    with col3:
        all_rules = set()
        for rules in filtered_df[filtered_df['fraud_detected'] == 1]['rules_triggered'].dropna():
            all_rules.update([r.strip() for r in str(rules).split(',')])
        all_rules = sorted(list(all_rules))
        
        selected_rules = st.multiselect(
            "Filter by Rules",
            options=all_rules,
            default=[]
        )
    
    # Apply filters
    explorer_df = filtered_df.copy()
    if show_fraud_only:
        explorer_df = explorer_df[explorer_df['fraud_detected'] == 1]
    
    explorer_df = explorer_df[explorer_df['fraud_score'] >= min_score]
    
    # Apply rules filter
    if selected_rules:
        def has_selected_rule(rules_str):
            if pd.isna(rules_str):
                return False
            rules_list = [r.strip() for r in str(rules_str).split(',')]
            return any(rule in rules_list for rule in selected_rules)
        
        explorer_df = explorer_df[explorer_df['rules_triggered'].apply(has_selected_rule)]
    
    # Display table
    st.dataframe(
        explorer_df[[
            'claim_id', 'claim_date', 'provider_id', 'claim_amount', 
            'fraud_score', 'rules_triggered', 'explanation'
        ]].sort_values('fraud_score', ascending=False),
        use_container_width=True,
        hide_index=True
    )

# ===========================
# TAB 3: INVESTIGATION
# ===========================
with tab3:
    st.header("Deep Investigation")
    
    # Claim selector
    claim_id_input = st.text_input("Enter Claim ID to Investigate (e.g., C00034)")
    
    if st.button("Investigate Claim") and claim_id_input:
        with st.spinner(f"Agents analyzing claim {claim_id_input}..."):
            result = orchestrator.investigate_single_claim(claim_id_input)
            
            if "error" in result:
                st.error(result['error'])
            else:
                # Get claim details if available
                claim_data = result.get('claim_data', {})
                
                # ========================================
                # Claim Information Card
                # ========================================
                st.subheader("📋 Claim Information")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Claim ID", claim_id_input)
                with col2:
                    amount = claim_data.get('claim_amount', 'N/A')
                    st.metric("Amount", f"${amount:,.2f}" if isinstance(amount, (int, float)) else amount)
                with col3:
                    st.metric("Provider", claim_data.get('provider_id', 'N/A'))
                with col4:
                    fraud_score = result.get('fraud_score', 0)
                    st.metric("Fraud Score", f"{fraud_score}/100", delta=None)
                
                st.markdown("---")
                
                # ========================================
                # Detection Results
                # ========================================
                if result.get('fraud_detected', False):
                    # FRAUD DETECTED
                    st.error("🚨 FRAUD DETECTED")
                    
                    # Show rules triggered
                    rules = result.get('rules_triggered', [])
                    if rules:
                        st.warning(f"**Rules Triggered:** {', '.join(rules)}")
                    
                    # ========================================
                    # AI Analysis Tabs
                    # ========================================
                    st.subheader("🤖 AI Agent Analysis")
                    
                    analysis_tab1, analysis_tab2 = st.tabs(["💬 Explanation (Business)", "🔍 Investigation (Technical)"])
                    
                    with analysis_tab1:
                        st.markdown("### Explanation Agent Analysis")
                        st.info("📊 **Business-Friendly Summary**")
                        explanation = result.get('explanation', 'No explanation available')
                        st.markdown(explanation)
                    
                    with analysis_tab2:
                        st.markdown("### Investigation Agent Deep Dive")
                        
                        investigation = result.get('investigation', {})
                        analysis = investigation.get('analysis', 'No analysis available')
                        
                        # Parse investigation analysis if it contains structured data
                        st.info("🔬 **Expert Fraud Analysis**")
                        st.markdown(analysis)
                        
                        # Show additional investigation details if available
                        if 'likelihood' in investigation:
                            likelihood = investigation['likelihood']
                            if likelihood == 'HIGH':
                                st.error(f"**Fraud Likelihood:** {likelihood} ⚠️")
                            elif likelihood == 'MEDIUM':
                                st.warning(f"**Fraud Likelihood:** {likelihood} ⚡")
                            else:
                                st.info(f"**Fraud Likelihood:** {likelihood}")
                        
                        if 'red_flags' in investigation:
                            with st.expander("🚩 Red Flags Identified", expanded=True):
                                for flag in investigation['red_flags']:
                                    st.markdown(f"- {flag}")
                        
                        if 'priority' in investigation:
                            priority = investigation['priority']
                            st.metric("Investigation Priority", priority)
                
                else:
                    # NO FRAUD DETECTED
                    st.success("✅ NO FRAUD DETECTED")
                    st.info("This claim passed all fraud detection rules and appears to be legitimate.")
                    
                    # Show some positive indicators
                    st.markdown("**Verification Checks:**")
                    st.markdown("- ✅ No duplicate claims detected")
                    st.markdown("- ✅ Amount within normal range")
                    st.markdown("- ✅ Valid procedure-diagnosis codes")
                    st.markdown("- ✅ Normal provider billing patterns")
                    st.markdown("- ✅ No velocity fraud indicators")

# ===========================
# TAB 4: AI ASSISTANT
# ===========================
with tab4:
    st.header("💬 AI Assistant (Query Agent)")
    
    # Add CSS for proper column behavior
    st.markdown("""
        <style>
        /* Right column (FAQ) - sticky */
        div[data-testid="stVerticalBlock"] > div:nth-child(2) div[data-testid="column"]:nth-child(2) {
            position: sticky !important;
            top: 60px !important;
            max-height: calc(100vh - 200px) !important;
            overflow-y: auto !important;
        }
        
        /* Left column (chat) - scrollable */
        div[data-testid="stVerticalBlock"] > div:nth-child(2) div[data-testid="column"]:nth-child(1) {
            max-height: calc(100vh - 300px) !important;
            overflow-y: auto !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Chat history initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Check if there's a pending question from example button
    if 'pending_question' in st.session_state:
        prompt = st.session_state.pop('pending_question')
    else:
        prompt = None
    
    # ========================================
    # 2-COLUMN LAYOUT
    # ========================================
    col_chat, col_faq = st.columns([2, 1])
    
    # LEFT COLUMN: Chat (scrollable)
    with col_chat:
        # Display chat messages
        if len(st.session_state.messages) == 0:
            st.info("👋 **Welcome!** Ask me anything about fraud detection data. Use the FAQ on the right or type below.")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # RIGHT COLUMN: FAQ (sticky)
    with col_faq:
        st.subheader("💡 Frequently Asked")
        
        example_questions = [
            "What are the most common fraud patterns?",
            "Show me duplicate claim cases",
            "Which providers have the most fraud flags?",
            "How many fraud cases involve amounts over $50,000?",
            "List all provider specialties",
            "What are impossible scenario frauds?"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"faq_{i}", use_container_width=True):
                st.session_state['pending_question'] = question
                st.rerun()
        
        st.markdown("---")
        st.info("💡 **Tip:** Type your own question in the input box below.")

    # Chat input - this keeps it fixed at bottom
    if not prompt:
        prompt = st.chat_input("Type your question here... (e.g., 'Show me high value fraud cases')")
    
    # Process the prompt (from input or example button)
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🤔 Analyzing fraud data..."):
            response = orchestrator.query_system(prompt)
            answer = response['answer']
            sources = response['sources']  # Use all sources (now 10)
            
            # Format response with better styling
            full_response = f"{answer}\n\n**Sources:** {', '.join(sources)}"
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Rerun to display new messages
        st.rerun()

