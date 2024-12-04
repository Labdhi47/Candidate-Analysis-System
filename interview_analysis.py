import torch
from transformers import pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from moviepy.editor import VideoFileClip
from faster_whisper import WhisperModel
import tempfile
import os

# Set device for PyTorch (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InterviewAnalysis:
    def __init__(self):
        """Initialize the InterviewAnalysis system by setting up models and components."""
        self.setup_components()

    def setup_components(self):
        """Set up the necessary components for transcription, text generation, and LangChain."""
        # Initialize Whisper model for audio transcription
        self.whisper_model = WhisperModel("base")

        # Initialize text generation pipeline using a pre-trained model
        self.gen_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=device
        )

        # Setup LangChain components
        self.llm = HuggingFacePipeline(pipeline=self.gen_pipeline)  # Language model for analysis
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"  # Embedding model for vectorization
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Size of text chunks for processing
            chunk_overlap=100  # Overlap between chunks
        )

    def process_video(self, video_file):
        """
        Process the uploaded video file to extract and transcribe audio.

        Args:
            video_file: The uploaded video file.

        Returns:
            str: The transcribed text from the audio.
        """
        # Save temporary video file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video.write(video_file.read())
        temp_video.close()

        # Extract audio from the video
        video = VideoFileClip(temp_video.name)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        video.audio.write_audiofile(temp_audio.name, logger=None)

        # Transcribe the audio to text using the Whisper model
        segments, _ = self.whisper_model.transcribe(temp_audio.name)
        transcript = " ".join([segment.text for segment in segments])  # Combine segments into a single transcript

        # Cleanup temporary files
        os.unlink(temp_video.name)  # Delete temporary video file
        os.unlink(temp_audio.name)  # Delete temporary audio file
        video.close()  # Close the video file

        return transcript  # Return the transcribed text

    def create_vector_store(self, transcript):
        """
        Create a vector store from the transcribed text.

        Args:
            transcript: The transcribed text to be converted into vector embeddings.

        Returns:
            vector_store: The created vector store for the transcript.
        """
        # Split the transcript into manageable chunks
        chunks = self.text_splitter.split_text(transcript)

        # Create a vector store using the text chunks and embeddings
        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=self.embeddings
        )

        return vector_store  # Return the created vector store

    def analyze_interview(self, vector_store, category):
        """
        Analyze the interview based on the specified category using the vector store.

        Args:
            vector_store: The vector store containing the transcript embeddings.
            category: The category for analysis (e.g., communication, active listening, engagement).

        Returns:
            str: The detailed analysis result along with a score.
        """
        # Define analysis prompts for different categories
        prompts = {
            "communication": """
            Analyze the candidate's communication style based on the following interview segment:
            Context: {context}

            Focus on:
            1. Clarity of expression
            2. Professional language use
            3. Speaking pace and tone
            4. Ability to articulate ideas

            Provide a detailed analysis with specific examples and a score out of 10.
            """,

            "active_listening": """
            Evaluate the candidate's active listening skills based on the following interview segment:
            Context: {context}

            Focus on:
            1. Response relevance
            2. Question comprehension
            3. Follow-up questions
            4. Engagement with interviewer's points

            Provide a detailed analysis with specific examples and a score out of 10.
            """,

            "engagement": """
            Assess the candidate's engagement level based on the following interview segment:
            Context: {context}

            Focus on:
            1. Interaction quality
            2. Enthusiasm and interest
            3. Professional rapport
            4. Overall presence

            Provide a detailed analysis with specific examples and a score out of 10.
            """
        }

        # Create a QA chain for retrieving answers based on the vector store
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

        # Get analysis response based on the selected category
        response = qa_chain({"query": prompts[category]})

        # Format the output to include the analysis result and score
        analysis_result = response["result"]
        score = self.extract_score(analysis_result)  # Extract score from the analysis result
        return f"{analysis_result} Score: {score}/10"  # Return formatted analysis with score

    def extract_score(self, analysis_text):
        """
        Extract the score from the analysis text.

        Args:
            analysis_text: The text containing the analysis result.

        Returns:
            str: The extracted score as a string, or "0" if not found.
        """
        # Check if the score is present in the analysis text
        if "Score:" in analysis_text:
            score = analysis_text.split("Score:")[-1].strip().split("/")[0]  # Extract score
            return score if score.isdigit() else "0"  # Return score if it's a digit, otherwise return "0"
        return "0"  # Default score if not found
