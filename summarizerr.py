from fastapi import FastAPI, HTTPException, Depends
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import nltk
app = FastAPI()

# Load the pre-trained models
checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def generate_summary(article: str) -> List[str]:
    # Tokenize and split the article into chunks
    sentences = nltk.tokenize.sent_tokenize(article)
    # ... (the rest of your chunking logic)
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk.strip()) # save the chunk
            
        else: 
            chunks.append(chunk.strip()) # save the chunk
            
            # reset 
            length = 0 
            chunk = ""

            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    # Generate summaries for each chunk
    summaries = []
    for chunk in chunks:
        input_ids = tokenizer(chunk, return_tensors="pt").input_ids
        output = model.generate(input_ids)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries

@app.post("/generate_summary/")
async def read_item(article: str):
    summaries = generate_summary(article)
    return {"summaries": summaries}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
