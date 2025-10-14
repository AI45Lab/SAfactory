from fastapi import FastAPI
import os
from .model import *
from .env_wrapper import server

app = FastAPI()

VISUAL = os.environ.get("VISUAL", "false").lower() == "true"
if VISUAL:
    print("Running in VISUAL mode")
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.get("/")
def hello():
    return "This is environment Trading."


@app.post("/create")
async def create(body: CreateRequestBody):
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    print(f"/step {body.id} {body.action}")
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    print(f"/reset {body.id}")
    return server.reset(body.id)


@app.get("/observation")
def get_observation(id: int):
    print(f"/observation {id}")
    return server.get_observation(id)


@app.get("/detail")
def get_detailed_info(id: int):
    return server.get_detailed_info(id)

@app.post("/close")
def close(body: CloseRequestBody):
    print(f"/close {body.id}")
    return server.close(body.id)
