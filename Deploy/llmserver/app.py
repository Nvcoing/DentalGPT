from fastapi import FastAPI
from generate import router as generate_router

app = FastAPI()
app.include_router(generate_router)
