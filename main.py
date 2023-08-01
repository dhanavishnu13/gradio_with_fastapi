from sentence_transformers import SentenceTransformer, util

bi_encoder = SentenceTransformer('nq-distilbert-base-v1')

### Create corpus embeddings containing the wikipedia passages
### To keep things summaraized, we are not going to show the code for this part

import json
from sentence_transformers import SentenceTransformer, util
import time
import gzip
import os
import torch

base_directory = os.path.dirname(os.path.realpath(''))

# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
model_name = "nq-distilbert-base-v1"
bi_encoder = SentenceTransformer(model_name)
top_k = 5  # Number of passages we want to retrieve with the bi-encoder

# As dataset, we use Simple English Wikipedia. Compared to the full English wikipedia, it has only
# about 170k articles. We split these articles into paragraphs and encode them with the bi-encoder

wikipedia_filepath = os.path.join(base_directory, "data/simplewiki-2020-11-01.jsonl.gz")

if not os.path.exists(wikipedia_filepath):
    util.http_get(
        "http://sbert.net/datasets/simplewiki-2020-11-01.jsonl.gz", wikipedia_filepath
    )

passages = []
with gzip.open(wikipedia_filepath, "rt", encoding="utf8") as fIn:
    for line in fIn:
        data = json.loads(line.strip())
        for paragraph in data["paragraphs"]:
            # We encode the passages as [title, text]
            passages.append([data["title"], paragraph])

# If you like, you can also limit the number of passages you want to use
print("Passages:", len(passages))

# To speed things up, pre-computed embeddings are downloaded.
# The provided file encoded the passages with the model 'nq-distilbert-base-v1'
if model_name == "nq-distilbert-base-v1":
    embeddings_filepath = os.path.join(
        base_directory, "simplewiki-2020-11-01-nq-distilbert-base-v1.pt"
    )
    if not os.path.exists(embeddings_filepath):
        util.http_get(
            "http://sbert.net/datasets/simplewiki-2020-11-01-nq-distilbert-base-v1.pt",
            embeddings_filepath,
        )

    if torch.cuda.is_available():
        corpus_embeddings = torch.load(embeddings_filepath)
    else:
        corpus_embeddings = torch.load(embeddings_filepath, map_location="cpu")

    corpus_embeddings = corpus_embeddings.float()  # Convert embedding file to float

    if torch.cuda.is_available():
        corpus_embeddings = corpus_embeddings.to("cuda")

else:  # Here, we compute the corpus_embeddings from scratch (which can take a while depending on the GPU)
    corpus_embeddings = bi_encoder.encode(
        passages, convert_to_tensor=True, show_progress_bar=True
    )


def predict(query: str):
    # Encode the query using the bi-encoder and find potentially relevant passages
    start_time = time.time()
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]  # Get the hits for the first query

    end_time = time.time()

    # Output of top-k hits
    print("Input question:", query)
    print("Results (after {:.3f} seconds):".format(end_time - start_time))

    results = [
        {
            "score": hit["score"],
            "title": passages[hit["corpus_id"]][0],
            "text": passages[hit["corpus_id"]][1],
        }
        for hit in hits
    ]

    for hit in hits:
        print("\t{:.3f}\t{}".format(hit["score"], passages[hit["corpus_id"]]))

    print("\n\n========\n")

    return results

import gradio as gr

def gradio_predict(question: str):
    results = predict(question) # results is a list of dictionaries

    best_result = results[0]

    # return a tuple of the title and text as a string, and the score as a number
    return f"{best_result['title']}\n\n{best_result['text']}", best_result["score"]

demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(
        label="Ask a question", placeholder="What is the capital of France?"
    ),
    outputs=[gr.Textbox(label="Answer"), gr.Number(label="Score")],
    allow_flagging="never",
)

# demo.launch()

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from main import predict

app = FastAPI()


class Request(BaseModel):
    question: str


class Result(BaseModel):
    score: float
    title: str
    text: str


class Response(BaseModel):
    results: List[Result] # list of Result objects


@app.post("/predict", response_model=Response)
async def predict_api(request: Request):
    results = predict(request.question)
    return Response(
        results=[
            Result(score=r["score"], title=r["title"], text=r["text"])
            for r in results
        ]
    )

app = gr.mount_gradio_app(app, demo, path="/")