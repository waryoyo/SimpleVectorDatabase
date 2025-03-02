from dotenv import load_dotenv
import app
import uvicorn


load_dotenv()

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8420, reload=True, access_log=True)
