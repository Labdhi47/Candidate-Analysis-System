import streamlit as st
from interview_analysis import InterviewAnalysis

def main():
    """
    Main function to run the Streamlit application for interview analysis.
    It handles the user interface and interaction for uploading videos and displaying results.
    """
    # Set the page configuration for the Streamlit app
    st.set_page_config(page_title="Interview Analysis System", layout="wide")

    # Header of the application
    st.title("🎥 Interview Analysis System")
    st.write("Upload an interview video for comprehensive analysis")

    # Initialize the interview analysis system if not already done
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = InterviewAnalysis()

    # File upload section for the interview video
    st.subheader("📤 Upload Interview")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'mov', 'avi'],
        help="Upload an interview video file (MP4, MOV, or AVI format)"
    )

    # If a file is uploaded, process it
    if uploaded_file:
        with st.spinner("Processing interview content..."):
            try:
                # Get the transcript of the uploaded video
                transcript = st.session_state.analyzer.process_video(uploaded_file)

                # Create a vector store from the transcript for analysis
                vector_store = st.session_state.analyzer.create_vector_store(transcript)

                st.success("✅ Interview processed successfully!")

                # Analysis section header
                st.subheader("📊 Interview Analysis")

                # Create tabs for different types of analyses
                comm_tab, listen_tab, engage_tab = st.tabs([
                    "💬 Communication Style",
                    "👂 Active Listening",
                    "🤝 Engagement"
                ])

                # Communication Style Analysis
                with comm_tab:
                    with st.spinner("Analyzing communication style..."):
                        comm_analysis = st.session_state.analyzer.analyze_interview(
                            vector_store,
                            "communication"
                        )
                        st.markdown("### Communication Style")
                        st.write(comm_analysis)

                # Active Listening Analysis
                with listen_tab:
                    with st.spinner("Analyzing listening skills..."):
                        listen_analysis = st.session_state.analyzer.analyze_interview(
                            vector_store,
                            "active_listening"
                        )
                        st.markdown("### Active Listening")
                        st.write(listen_analysis)

                # Engagement Analysis
                with engage_tab:
                    with st.spinner("Analyzing engagement..."):
                        engage_analysis = st.session_state.analyzer.analyze_interview(
                            vector_store,
                            "engagement"
                        )
                        st.markdown("### Engagement")
                        st.write(engage_analysis)

                # Overall Summary Section
                st.subheader("📑 Overall Summary")
                summary_expander = st.expander("View Complete Analysis")
                with summary_expander:
                    st.markdown("### Communication Style")
                    st.write(comm_analysis)

                    st.markdown("### Active Listening")
                    st.write(listen_analysis)

                    st.markdown("### Engagement")
                    st.write(engage_analysis)

                # Download Report Button
                st.download_button(
                    label="📥 Download Analysis Report",
                    data=f"""
                    Interview Analysis Report

                    Communication Style:
                    {comm_analysis}

                    Active Listening:
                    {listen_analysis}

                    Engagement:
                    {engage_analysis}
                    """,
                    file_name="interview_analysis.txt",
                    mime="text/plain"
                )

            except Exception as e:
                # Display an error message if processing fails
                st.error(f"Error processing video: {str(e)}")

    # Sidebar with additional information about the application
    with st.sidebar:
        st.header("About")
        st.write("""
        This system analyzes interview videos and provides insights on:
        - Communication Style
        - Active Listening Skills
        - Candidate Engagement

        Upload a video to get started!
        """)

        st.header("📋 Analysis Criteria")
        with st.expander("Communication Style"):
            st.write("""
            - Clarity of expression
            - Professional language
            - Speaking pace and tone
            - Idea articulation
            """)

        with st.expander("Active Listening"):
            st.write("""
            - Response relevance
            - Question comprehension
            - Follow-up questions
            - Engagement with points
            """)

        with st.expander("Engagement"):
            st.write("""
            - Interaction quality
            - Enthusiasm
            - Professional rapport
            - Overall presence
            """)

# Entry point for the Streamlit application
if __name__ == "__main__":
    main()