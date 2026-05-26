import asyncio
import sys

# Windows selector event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from recsys.db.core import init_db

async def main():
    await init_db()
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())
