from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router


app = FastAPI(
    title="API Iris - A3Data",
    openapi_url="/api/v1/openapi.json"
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router, prefix="/api/v1")



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app="main:app", host="0.0.0.0", port=8001, log_level="debug", reload=True)