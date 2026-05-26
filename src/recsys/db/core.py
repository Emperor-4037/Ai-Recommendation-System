import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

# The default PostgreSQL connection string configured in docker-compose.yml
# In production or different environments, this should be configurable via env vars.
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://recsys_user:recsys_password@127.0.0.1:5433/recsys_db?ssl=disable"
)

# Async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    # Memory-conscious setup: prevent SQLAlchemy from loading too much in memory
    pool_pre_ping=True,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting an async database session."""
    async with AsyncSessionLocal() as session:
        yield session

async def init_db():
    """Initializes the database, creating all tables."""
    # We must import models here to ensure they are registered with Base
    from recsys.db.models import Base
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
