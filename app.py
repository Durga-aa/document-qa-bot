import streamlit as st
from rag_pipeline import search_documents, generate_answer

st.set_page_config(
    page_title="Document Q&A Bot",
    page_icon="📚"
)

st.title("📚 Document Q&A Bot")
st.write("Ask questions about your PDF documents.")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is Machine Learning?"
)

if st.button("Ask"):

    if question.strip():

        documents, sources = search_documents(question)

        answer = generate_answer(question, documents)

        st.write("### Answer")
        st.write(answer)

        # Display sources only when relevant documents are found
        if documents and sources:

            st.write("### 📄 Sources")

            for i, source in enumerate(sources):

                st.write(f"**Source {i + 1}**")

                st.write(
                    f"**File:** {source.get('source', 'Unknown')}"
                )

                st.write(
                    f"**Page:** {source.get('page', 'Unknown')}"
                )

                with st.expander("View source chunk"):

                    st.write(documents[i])

    else:

       st.write("### 📄 Sources")
       st.write("no relevant sources founds in the provided documents.")