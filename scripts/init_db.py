from backend.db import Base, engine
from backend.models import Document, Chunk

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created.")
