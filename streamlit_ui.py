import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🔍 Fake News Detection System")
st.markdown("Analyze articles for credibility and potential misinformation")
st.markdown("---")

# API endpoint
API_URL = "http://localhost:5000/api/analyze"

# Sidebar for examples and settings
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input("API URL", value=API_URL)
    
    st.markdown("---")
    st.header("📝 Load Example")
    
    example_type = st.selectbox(
        "Select Example",
        ["None", "Suspicious Article", "Credible Article"]
    )
    
    if example_type == "Suspicious Article":
        example_title = "SHOCKING: You Won't Believe What Scientists Discovered!"
        example_content = """This unbelievable and devastating discovery will outrage you! 
        Sources say that unnamed officials allegedly confirmed the rumors. According to reports, 
        this miracle cure has been hidden from the public. You won't believe what happens next! 
        The crisis is worse than anyone thought. They don't want you to know about this secret!"""
        example_author = "Anonymous"
    elif example_type == "Credible Article":
        example_title = "New Climate Study Published in Nature Journal"
        example_content = """Researchers from MIT published a comprehensive study examining 
        climate patterns over the past decade. The study, which appears in the journal Nature, 
        analyzed data from over 100 weather stations worldwide. Dr. Jane Smith, lead researcher, 
        stated that "the data shows a clear upward trend in global temperatures." The findings 
        were peer-reviewed and include detailed methodology. According to the paper, temperature 
        increases were measured at an average of 0.2 degrees Celsius per year across the studied 
        regions. The research team, consisting of 15 scientists from five institutions, spent 
        three years collecting and analyzing the data. Independent verification of the results 
        has been conducted by teams at Oxford University and Stanford University."""
        example_author = "Dr. Jane Smith"
    else:
        example_title = ""
        example_content = ""
        example_author = ""
    
    st.markdown("---")
    st.info("💡 Make sure the API is running at the specified URL")

# Main form
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Article Information")
    
    title = st.text_input(
        "Article Title *", 
        value=example_title,
        placeholder="Enter article title..."
    )
    
    content = st.text_area(
        "Article Content *", 
        value=example_content,
        height=300,
        placeholder="Paste or type article content here..."
    )

with col2:
    st.subheader("Metadata")
    
    author = st.text_input(
        "Author", 
        value=example_author,
        placeholder="Author name (optional)"
    )
    
    date = st.date_input(
        "Publication Date",
        value=datetime.now()
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    analyze_button = st.button("🔍 Analyze Article", type="primary", use_container_width=True)
    clear_button = st.button("🗑️ Clear Form", use_container_width=True)

# Clear form logic
if clear_button:
    st.rerun()

# Analyze article
if analyze_button:
    if not title or not content:
        st.error("⚠️ Please provide both title and content")
    else:
        with st.spinner("🔄 Analyzing article..."):
            try:
                # Prepare data
                article_data = {
                    "title": title,
                    "content": content,
                    "author": author if author else "",
                    "date": date.isoformat()
                }
                
                # Make API request
                response = requests.post(api_url, json=article_data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Overall score and verdict
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Overall Score", f"{result['overall_score']}/100")
                    
                    with col2:
                        verdict = result['verdict']
                        if "Credible" in verdict and "Questionable" not in verdict:
                            st.success(f"✅ {verdict}")
                        elif "Questionable" in verdict or "Possibly" in verdict:
                            st.warning(f"⚠️ {verdict}")
                        else:
                            st.error(f"❌ {verdict}")
                    
                    with col3:
                        st.info(f"🎯 Confidence: {result['confidence']}")
                    
                    st.markdown("---")
                    
                    # Component scores
                    st.subheader("📈 Component Scores")
                    
                    score_cols = st.columns(4)
                    components = result['component_scores']
                    
                    for idx, (component, score) in enumerate(components.items()):
                        with score_cols[idx]:
                            # Color based on score
                            if score >= 80:
                                st.markdown(f"**{component.title()}**")
                                st.success(f"{score}/100")
                            elif score >= 60:
                                st.markdown(f"**{component.title()}**")
                                st.warning(f"{score}/100")
                            else:
                                st.markdown(f"**{component.title()}**")
                                st.error(f"{score}/100")
                    
                    st.markdown("---")
                    
                    # Issues detected
                    st.subheader("⚠️ Issues Detected")
                    
                    all_issues = result['issues']
                    has_issues = False
                    
                    for category, issues in all_issues.items():
                        if issues:
                            has_issues = True
                            with st.expander(f"**{category.title()}** ({len(issues)} issue{'s' if len(issues) > 1 else ''})"):
                                for issue in issues:
                                    st.markdown(f"- {issue}")
                    
                    if not has_issues:
                        st.success("✅ No major issues detected!")
                    
                    st.markdown("---")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    for rec in result['recommendations']:
                        st.info(f"• {rec}")
                    
                    # Raw JSON (collapsible)
                    with st.expander("🔍 View Raw JSON Response"):
                        st.json(result)
                
                else:
                    st.error(f"❌ API Error: {response.status_code}")
                    st.code(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure the API is running at " + api_url)
                st.info("Run the API with: `python app.py`")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Fake News Detection API v1.0 | Built with Streamlit</p>
        <p style='font-size: 12px;'>Always verify information from multiple trusted sources</p>
    </div>
    """,
    unsafe_allow_html=True
)