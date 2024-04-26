import streamlit as st
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

def retrieve_paper():
    # Replace 'file_path' with the actual path of the PDF file
    file_path = "/content/drive/MyDrive/nopaper.pdf"
    with open(file_path, "r", encoding="latin-1") as file:
        paper_text = file.read()
    return paper_text

# Function to encode text into embeddings
def encode_text(text):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    with torch.no_grad():
        embeddings = model.encoder(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

def generate_answers(question, paper_text):
    # Initialize T5 tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # Encode the question and paper text
    input_text = f"question: {question} context: {paper_text}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate answer using the model
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, early_stopping=True)
    
    # Decode and return the answer
    answer_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer_text




def main():
    st.title("Leave No Context Behind - Q&A System")
    st.write("Ask a question about the paper and get an answer!")

    # Retrieve paper text
    paper_text = retrieve_paper()

    # Get user query
    user_query = st.text_input("Enter your question:")
    
    # Generate answer when user submits query
    if st.button("Get Answer"):
        if not user_query:
            st.error("Please enter a question.")
        else:
            # Encode paper text into embeddings
            paper_embeddings = encode_text(paper_text)
            
            # Generate answer using LangChain
            answer = generate_answers(user_query, paper_embeddings)
            
            st.write(f"**Question:** {user_query}")
            st.write(f"**Answer:** {answer}")

  

if __name__ == "__main__":
    main()
